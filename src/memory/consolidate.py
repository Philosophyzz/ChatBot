"""Sleep-time consolidation — the maintenance pass that keeps memory healthy.

An append-only memory store degrades: duplicates accumulate, old episodes drown
the prompt budget, stale facts keep their original importance, and anything
written while the embedding server was down stays unsearchable. This module is the
counterpart to the write path and runs on idle time, when the GPU is otherwise
free.

What one pass does, cheapest first:

1. **Backfill vectors** for memories that have none (recovers degraded writes).
2. **Merge near-duplicates** — keeps the highest-importance instance, folds the
   others into it, and retires them so retrieval cannot return the same claim
   three times.
3. **Summarize old episodes** into ``SUMMARY`` memories and retire the originals
   from the active set, which is what keeps recall over months from blowing up
   the token budget.
4. **Reflect** — when a cluster of related facts has grown, ask the model for a
   higher-order insight ("user prefers short replies in the morning but asks for
   detail at night") and store it as a ``REFLECTION``. This is the step that
   makes the assistant feel like it has *understood* rather than merely recorded.
5. **Decay** importance of memories that were never retrieved, so unused noise
   fades while frequently confirmed facts stay near the top.

Every step is bounded and idempotent, and the whole pass is a no-op when there is
nothing to do — it is safe to run on a timer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.logging import get_logger
from core.types import LLMRequest, MemoryItem, MemoryKind, Message, Role, now_ts
from core.utils import truncate_to_tokens
from llm.embed import cosine
from memory.store import MemoryStore, content_hash

log = get_logger(__name__)

_DAY = 86400.0

_CONSOLIDATION_CURSOR = "consolidation_cursor"
_LAST_RUN = "consolidation_last_run"


@dataclass
class ConsolidationReport:
    vectors_backfilled: int = 0
    duplicates_merged: int = 0
    episodes_summarized: int = 0
    reflections_created: int = 0
    importance_decayed: int = 0
    elapsed_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_public(self) -> Dict[str, Any]:
        return {
            "vectors_backfilled": self.vectors_backfilled,
            "duplicates_merged": self.duplicates_merged,
            "episodes_summarized": self.episodes_summarized,
            "reflections_created": self.reflections_created,
            "importance_decayed": self.importance_decayed,
            "elapsed_ms": self.elapsed_ms,
            "errors": list(self.errors),
        }


_SUMMARY_PROMPT = """下面是若干条较早的对话记忆片段。请把它们压缩成一段简洁的中文摘要：
- 保留具体的人名、时间、数字、决定和用户偏好。
- 丢弃寒暄和一次性信息。
- 不超过 200 字，不要分点，直接给一段话。
只输出摘要正文。"""

_REFLECTION_PROMPT = """下面是关于同一位用户的一组长期记忆。请提炼 1~2 条**更高层的洞察**（例如稳定的价值观、作息规律、沟通偏好、反复出现的困扰）。

要求：
- 每条是一句陈述句，必须由给定记忆直接支持，不要凭空推测。
- 如果这些记忆之间没有值得总结的规律，就返回空数组。
- 只输出 JSON：{{"reflections":[{{"content":"","importance":0.5,"confidence":0.6,"tags":[]}}]}}"""


class Consolidator:
    """Runs bounded maintenance passes over the memory store."""

    def __init__(
        self,
        store: MemoryStore,
        llm: Any = None,
        embedder: Any = None,
        vectors: Any = None,
        *,
        config: Any = None,
        scope: str = "global",
    ) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.vectors = vectors
        self.config = config
        self.scope = scope
        cfg = getattr(config, "memory", None)
        self.enabled = bool(getattr(cfg, "consolidate_enabled", True))
        self.min_episodes = int(getattr(cfg, "consolidate_min_episodes", 24))
        self.reflections_enabled = bool(getattr(cfg, "reflections_enabled", True))
        self.duplicate_threshold = float(getattr(cfg, "contradiction_threshold", 0.86)) + 0.08
        self.recent_window_s = 3 * _DAY

    # -- orchestration -----------------------------------------------------------------
    async def run(self, *, scope: str = "", force: bool = False) -> ConsolidationReport:
        started = time.perf_counter()
        scope = scope or self.scope
        report = ConsolidationReport()

        if not self.enabled and not force:
            report.errors.append("consolidation disabled by configuration")
            return report

        for step in (
            self._backfill_vectors,
            self._merge_duplicates,
            self._summarize_episodes,
            self._reflect,
            self._decay_importance,
        ):
            try:
                await step(scope, report)
            except Exception as exc:  # noqa: BLE001 - one failing step must not abort the pass
                log.warning("consolidation step failed", extra={"step": step.__name__, "error": str(exc)})
                report.errors.append(f"{step.__name__}: {exc}"[:200])

        report.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        try:
            await self.store.kv_set(_LAST_RUN, now_ts())
        except Exception:  # noqa: BLE001
            pass
        log.info("consolidation pass finished", extra=report.to_public())
        return report

    # -- steps -------------------------------------------------------------------------
    async def _backfill_vectors(self, scope: str, report: ConsolidationReport) -> None:
        if self.embedder is None or self.vectors is None:
            return
        stored = {mid for mid, _blob, _dim in await self.vectors.iter_all()}
        pending = [
            item
            for item in self.store.fetch_active_candidates(scope=scope, limit=4000)
            if item.id not in stored and len(item.content) >= 4
        ]
        if not pending:
            return
        batch = 16
        for start in range(0, len(pending), batch):
            chunk = pending[start : start + batch]
            try:
                vectors = await self.embedder.embed([item.content for item in chunk])
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"backfill embedding: {exc}"[:160])
                return
            ids = [item.id for item in chunk]
            await self.vectors.upsert(ids, vectors, [{"kind": item.kind.value} for item in chunk])
            report.vectors_backfilled += len(ids)

    async def _merge_duplicates(self, scope: str, report: ConsolidationReport) -> None:
        """Collapse near-identical memories that survived the write-time checks."""
        if self.vectors is None:
            return
        candidates = self.store.fetch_active_candidates(scope=scope, limit=2000)
        by_kind: Dict[str, List[MemoryItem]] = {}
        for item in candidates:
            kind = item.kind.value if isinstance(item.kind, MemoryKind) else str(item.kind)
            # Summaries and reflections are already distilled; merging them losses information.
            if kind in {MemoryKind.SUMMARY.value, MemoryKind.REFLECTION.value}:
                continue
            by_kind.setdefault(kind, []).append(item)

        vectors = await self.vectors.get([item.id for item in candidates])
        merged_ids: set = set()
        for kind, items in by_kind.items():
            # Only compare items sharing a subject: cross-subject duplicates are
            # almost always legitimately distinct.
            groups: Dict[str, List[MemoryItem]] = {}
            for item in items:
                if item.id in merged_ids:
                    continue
                groups.setdefault((item.subject or "").strip() or "_", []).append(item)
            for _subject, group in groups.items():
                if len(group) < 2:
                    continue
                ordered = sorted(group, key=lambda i: (i.importance, i.created_at), reverse=True)
                for index, keeper in enumerate(ordered):
                    if keeper.id in merged_ids or keeper.id not in vectors:
                        continue
                    keeper_vec = vectors[keeper.id]
                    for other in ordered[index + 1 :]:
                        if other.id in merged_ids or other.id not in vectors:
                            continue
                        if content_hash(keeper.content) == content_hash(other.content):
                            similarity = 1.0
                        else:
                            similarity = cosine(keeper_vec, vectors[other.id])
                        if similarity >= self.duplicate_threshold:
                            await self.store.update_memory(
                                keeper.id,
                                importance=min(1.0, max(keeper.importance, other.importance) + 0.02),
                            )
                            await self.store.retire_memory(other.id, superseded_by=keeper.id)
                            merged_ids.add(other.id)
                            report.duplicates_merged += 1

    async def _summarize_episodes(self, scope: str, report: ConsolidationReport) -> None:
        """Compress old episode memories into periodic summaries."""
        if self.llm is None:
            return
        cutoff = now_ts() - self.recent_window_s
        episodes = [
            item
            for item in self.store.fetch_active_candidates(scope=scope, limit=4000)
            if (item.kind == MemoryKind.EPISODE) and item.created_at < cutoff
        ]
        if len(episodes) < self.min_episodes:
            return
        episodes.sort(key=lambda item: item.created_at)
        # Process one bounded chunk per pass so a huge backlog cannot stall the loop.
        chunk = episodes[: max(self.min_episodes, 40)]
        transcript = "\n".join(f"- [{time.strftime('%Y-%m-%d', time.localtime(item.created_at))}] {item.content}" for item in chunk)
        request = LLMRequest(
            messages=[
                Message(role=Role.SYSTEM, content=_SUMMARY_PROMPT),
                Message(role=Role.USER, content=truncate_to_tokens(transcript, 2400)),
            ],
            temperature=0.2,
            max_tokens=400,
        )
        try:
            completion = await self.llm.complete(request)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"summary completion: {exc}"[:160])
            return
        summary_text = (completion.text or "").strip()
        if len(summary_text) < 10:
            return
        summary_id = await self.store.add_memory(
            content=summary_text,
            kind=MemoryKind.SUMMARY,
            importance=0.55,
            confidence=0.7,
            scope=scope,
            tags=["consolidated"],
            sources=[item.id for item in chunk],
            meta={"source_count": len(chunk), "period_start": chunk[0].created_at, "period_end": chunk[-1].created_at},
            dedupe=False,
        )
        if summary_id and self.embedder is not None and self.vectors is not None:
            try:
                vector = await self.embedder.embed_one(summary_text)
                if vector:
                    await self.vectors.upsert([summary_id], [vector], [{"kind": MemoryKind.SUMMARY.value}])
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"summary embedding: {exc}"[:160])
        # Retire the originals: the summary now carries their information, and the
        # raw text stays in the messages table for full-text history search.
        for item in chunk:
            await self.store.retire_memory(item.id, superseded_by=summary_id or None)
        report.episodes_summarized += len(chunk)
        await self.store.kv_set(_CONSOLIDATION_CURSOR, chunk[-1].created_at)

    async def _reflect(self, scope: str, report: ConsolidationReport) -> None:
        """Derive higher-order insights from clusters of related memories."""
        if not self.reflections_enabled or self.llm is None or self.vectors is None:
            return
        candidates = [
            item
            for item in self.store.fetch_active_candidates(scope=scope, limit=2000)
            if item.kind in {MemoryKind.FACT, MemoryKind.PREFERENCE, MemoryKind.PROFILE}
            and item.importance >= 0.45
        ]
        if len(candidates) < 8:
            return
        # Cluster by subject so the insight has a coherent topic.
        by_subject: Dict[str, List[MemoryItem]] = {}
        for item in candidates:
            by_subject.setdefault((item.subject or "综合").strip() or "综合", []).append(item)
        # Only the largest group per pass; keeps cost predictable.
        subject, group = max(by_subject.items(), key=lambda pair: len(pair[1]))
        if len(group) < 6:
            return

        existing = {
            content_hash(item.content)
            for item in self.store.fetch_active_candidates(scope=scope, limit=2000)
            if item.kind == MemoryKind.REFLECTION
        }
        listing = "\n".join(f"- {item.content}" for item in group[:24])
        request = LLMRequest(
            messages=[
                Message(role=Role.SYSTEM, content=_REFLECTION_PROMPT),
                Message(role=Role.USER, content=f"主题：{subject}\n记忆：\n{listing}"),
            ],
            temperature=0.3,
            max_tokens=400,
        )
        try:
            completion = await self.llm.complete(request)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"reflection completion: {exc}"[:160])
            return
        from memory.extract import parse_json_loose

        parsed = parse_json_loose(completion.text or "")
        if not parsed:
            return
        for entry in (parsed.get("reflections") or [])[:2]:
            if not isinstance(entry, dict):
                continue
            content = str(entry.get("content") or "").strip()
            if len(content) < 8 or content_hash(content) in existing:
                continue
            memory_id = await self.store.add_memory(
                content=content,
                kind=MemoryKind.REFLECTION,
                subject=subject,
                importance=min(1.0, max(0.4, float(entry.get("importance") or 0.55))),
                confidence=min(1.0, max(0.3, float(entry.get("confidence") or 0.6))),
                scope=scope,
                tags=["reflection", *[str(t)[:16] for t in (entry.get("tags") or [])][:4]],
                dedupe=False,
            )
            if memory_id:
                report.reflections_created += 1
                if self.embedder is not None:
                    try:
                        vector = await self.embedder.embed_one(content)
                        if vector:
                            await self.vectors.upsert([memory_id], [vector], [{"kind": MemoryKind.REFLECTION.value}])
                    except Exception:  # noqa: BLE001
                        pass

    async def _decay_importance(self, scope: str, report: ConsolidationReport) -> None:
        """Fade memories that were never retrieved; protect confirmed ones.

        Retrieval strength is the signal: the memory system already records
        ``access_count``, so unused items lose a little importance per pass while
        items that keep proving useful are left alone. This keeps the top of the
        ranking meaningful as the store grows.
        """
        candidates = self.store.fetch_active_candidates(scope=scope, limit=4000)
        now = now_ts()
        updates: List[Tuple[str, float]] = []
        for item in candidates:
            if item.kind in {MemoryKind.PROFILE, MemoryKind.REFLECTION}:
                continue
            age_days = (now - (item.valid_from or item.created_at)) / _DAY
            if age_days < 7:
                continue
            if item.access_count == 0 and item.importance > 0.25:
                updates.append((item.id, max(0.2, item.importance - 0.03)))
            elif item.access_count >= 3 and item.importance < 0.95:
                updates.append((item.id, min(0.99, item.importance + 0.02)))
        for memory_id, importance in updates:
            await self.store.update_memory(memory_id, importance=importance)
        report.importance_decayed = len(updates)


__all__ = ["ConsolidationReport", "Consolidator"]
