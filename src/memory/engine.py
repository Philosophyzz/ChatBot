"""The memory subsystem facade.

One object owns the whole vertical slice — persistence, embeddings, retrieval,
extraction and consolidation — so the chat engine and the HTTP layer depend on a
single, high-level interface:

    engine = await MemoryEngine.create(config)
    context = await engine.build_context("你还记得我喜欢喝什么吗？", session_id=..., persona=...)
    await engine.remember_turn(session_id=..., user_message=..., assistant_message=...)

The background worker keeps extraction off the response path and runs
consolidation only when the system is idle, which matters on a single 16 GB GPU
where a maintenance pass and a chat completion compete for the same memory.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from core.types import Message, RetrievalQuery, RetrievalResult
from memory.consolidate import ConsolidationReport, Consolidator
from memory.db import Database
from memory.extract import ExtractionOutcome, MemoryExtractor
from memory.retrieve import HybridRetriever
from memory.store import MemoryStore

log = get_logger(__name__)


@dataclass
class MemoryContext:
    """The memory payload injected into one prompt."""

    rendered: str = ""
    hits: List[Any] = field(default_factory=list)
    profile_lines: List[str] = field(default_factory=list)
    tokens_used: int = 0
    debug: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.rendered.strip()

    def to_public(self, *, include_hits: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "tokens_used": self.tokens_used,
            "elapsed_ms": self.elapsed_ms,
            "hit_count": len(self.hits),
            "debug": self.debug,
        }
        if include_hits:
            payload["hits"] = [hit.to_public() for hit in self.hits]
        return payload


class MemoryEngine:
    """Wires the memory layers together and owns their lifecycle."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        db: Database,
        retriever: HybridRetriever,
        extractor: MemoryExtractor,
        consolidator: Consolidator,
        config: Any,
        embedder: Any = None,
        vectors: Any = None,
        reranker: Any = None,
        llm: Any = None,
    ) -> None:
        self.store = store
        self.db = db
        self.retriever = retriever
        self.extractor = extractor
        self.consolidator = consolidator
        self.config = config
        self.embedder = embedder
        self.vectors = vectors
        self.reranker = reranker
        self.llm = llm
        self._worker: Optional[asyncio.Task] = None
        self._queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=256)
        self._stop = asyncio.Event()
        self._last_activity = time.time()
        self._consolidation_lock = asyncio.Lock()
        self.stats: Dict[str, Any] = {"extracted": 0, "created": 0, "superseded": 0, "errors": 0}

    # ----------------------------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------------------------
    @classmethod
    async def create(
        cls,
        config: Any,
        *,
        llm: Any = None,
        db: Optional[Database] = None,
        embedder: Any = None,
        vectors: Any = None,
        reranker: Any = None,
    ) -> "MemoryEngine":
        """Build the engine, degrading gracefully when optional pieces are absent."""
        from core.registry import KIND_VECTOR_STORE, registry

        database = db or Database(config.paths.db_path)
        database.init()

        store = MemoryStore(database, default_scope="global")

        if embedder is None:
            try:
                from llm.embed import build_embedder

                embedder = build_embedder(config, role="primary")
                # Probe once: a configured-but-dead embedder should fall back now,
                # not silently degrade every retrieval later.
                if hasattr(embedder, "health"):
                    health = await embedder.health()
                    if not health.get("ok"):
                        log.warning("primary embedder unavailable, falling back", extra={"detail": health})
                        raise RuntimeError(health.get("error") or "embedder unhealthy")
            except Exception as exc:  # noqa: BLE001
                log.warning("using fallback embedder", extra={"error": str(exc)[:200]})
                from llm.embed import build_embedder as _build

                try:
                    embedder = _build(config, role="fallback")
                except Exception:  # pragma: no cover
                    from llm.embed import HashingEmbedder

                    embedder = HashingEmbedder(dim=config.memory.dim)

        if vectors is None:
            try:
                vectors = registry.create(KIND_VECTOR_STORE, "sqlite", database)
            except Exception:  # pragma: no cover
                from memory.vector import MemoryVectorStore

                vectors = MemoryVectorStore()

        if reranker is None:
            try:
                from llm.embed import build_reranker

                reranker = build_reranker(config)
            except Exception as exc:  # noqa: BLE001
                log.warning("reranker unavailable, using heuristic", extra={"error": str(exc)[:200]})
                from llm.embed import HeuristicReranker

                reranker = HeuristicReranker()

        retriever = HybridRetriever(store, embedder, vectors, reranker, config=config)
        extractor = MemoryExtractor(store, llm, embedder, vectors, config=config)
        consolidator = Consolidator(store, llm, embedder, vectors, config=config)

        return cls(
            store=store,
            db=database,
            retriever=retriever,
            extractor=extractor,
            consolidator=consolidator,
            config=config,
            embedder=embedder,
            vectors=vectors,
            reranker=reranker,
            llm=llm,
        )

    # ----------------------------------------------------------------------------------
    # Read path
    # ----------------------------------------------------------------------------------
    async def build_context(
        self,
        query: str,
        *,
        session_id: str,
        persona_id: Optional[str] = None,
        scope: str = "global",
        top_k: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> MemoryContext:
        """Retrieve and render the memory block for one user turn."""
        started = time.perf_counter()
        cfg = self.config.memory
        request = RetrievalQuery(
            text=query,
            session_id=session_id,
            persona_id=persona_id,
            scope=scope,
            top_k=int(top_k or cfg.top_k),
            token_budget=int(token_budget or cfg.token_budget),
        )
        try:
            result: RetrievalResult = await self.retriever.retrieve(request)
        except Exception as exc:  # noqa: BLE001 - memory failure must not block chat
            log.exception("memory retrieval failed")
            self.stats["errors"] += 1
            return MemoryContext(debug={"error": str(exc)[:200]}, elapsed_ms=(time.perf_counter() - started) * 1000)
        return MemoryContext(
            rendered=result.rendered,
            hits=result.hits,
            profile_lines=result.profile_lines,
            tokens_used=result.tokens_used,
            debug=result.debug,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def search_history(
        self, query: str, *, scope: str = "", conversation_id: str = "", limit: int = 20
    ) -> List[Dict[str, Any]]:
        return await self.retriever.search_history(
            query, scope=scope, conversation_id=conversation_id, limit=limit
        )

    # ----------------------------------------------------------------------------------
    # Write path
    # ----------------------------------------------------------------------------------
    async def remember_turn(
        self,
        *,
        session_id: str,
        user_message: Message,
        assistant_message: Optional[Message] = None,
        persona_id: Optional[str] = None,
        scope: str = "global",
        wait: bool = False,
    ) -> Optional[ExtractionOutcome]:
        """Queue (or directly run) extraction for one turn.

        Returns the outcome when ``wait`` is set (used by tests and the
        "/memory/extract" debug endpoint), otherwise ``None`` immediately.
        """
        self._last_activity = time.time()
        payload = {
            "session_id": session_id,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "persona_id": persona_id,
            "scope": scope,
        }
        if wait or not self.config.memory.extract_async:
            return await self._run_extraction(payload)
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            log.warning("memory extraction queue full, dropping turn", extra={"session": session_id})
            self.stats["errors"] += 1
        return None

    async def _run_extraction(self, payload: Dict[str, Any]) -> ExtractionOutcome:
        outcome = await self.extractor.process_turn(
            conversation_id=payload["session_id"],
            user_message=payload["user_message"],
            assistant_message=payload.get("assistant_message"),
            persona_id=payload.get("persona_id"),
            scope=payload.get("scope", "global"),
        )
        self.stats["extracted"] += 1
        self.stats["created"] += outcome.created
        self.stats["superseded"] += outcome.superseded
        if outcome.error:
            self.stats["errors"] += 1
        return outcome

    async def forget(self, memory_id: str, *, hard: bool = False) -> bool:
        return await self.store.delete_memory(memory_id, hard=hard)

    async def update_memory(self, memory_id: str, **fields: Any) -> bool:
        return await self.store.update_memory(memory_id, **fields)

    # ----------------------------------------------------------------------------------
    # Background worker
    # ----------------------------------------------------------------------------------
    def start_background(self) -> None:
        """Start the extraction consumer and the consolidation timer."""
        if self._worker is not None and not self._worker.done():
            return
        self._stop.clear()
        self._worker = asyncio.create_task(self._worker_loop(), name="memory-worker")
        log.info("memory background worker started")

    async def stop_background(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._worker = None

    async def _worker_loop(self) -> None:
        cfg = self.config.memory
        interval = max(60, int(cfg.consolidate_interval_s))
        idle_needed = max(30, int(cfg.consolidate_idle_s))
        next_consolidation = time.time() + interval
        while not self._stop.is_set():
            try:
                try:
                    payload = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    payload = None
                if payload is not None:
                    try:
                        await self._run_extraction(payload)
                    finally:
                        self._queue.task_done()

                if cfg.consolidate_enabled and time.time() >= next_consolidation:
                    if time.time() - self._last_activity >= idle_needed and not self._consolidation_lock.locked():
                        async with self._consolidation_lock:
                            await self.consolidator.run()
                    next_consolidation = time.time() + interval
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("memory worker iteration failed")
                self.stats["errors"] += 1
                await asyncio.sleep(5.0)

    async def consolidate_now(self) -> ConsolidationReport:
        async with self._consolidation_lock:
            return await self.consolidator.run(force=True)

    # ----------------------------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        stats = await self.store.stats()
        details: Dict[str, Any] = {}
        for label, component in (("embedder", self.embedder), ("reranker", self.reranker)):
            if component is not None and hasattr(component, "health"):
                try:
                    details[label] = await component.health()
                except Exception as exc:  # noqa: BLE001
                    details[label] = {"ok": False, "error": str(exc)[:160]}
        return {
            "ok": True,
            "store": stats,
            "embedder": getattr(self.embedder, "name", None),
            "reranker": getattr(self.reranker, "name", None),
            "vector_backend": getattr(self.vectors, "name", None),
            "pipeline": self.stats,
            "queue_depth": self._queue.qsize(),
            "components": details,
        }

    async def close(self) -> None:
        await self.stop_background()
        for component in (self.embedder, self.reranker):
            if component is not None and hasattr(component, "aclose"):
                try:
                    await component.aclose()
                except Exception:  # noqa: BLE001
                    pass
        await self.store.close()


__all__ = ["MemoryContext", "MemoryEngine"]
