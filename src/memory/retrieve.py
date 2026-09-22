"""Hybrid, time-aware memory retrieval.

Retrieval is where a memory system either feels magical or useless, so the design
is deliberately multi-channel and explainable. For one query we fuse four kinds of
evidence:

======================  ==========================================================
channel                 what it catches that the others miss
======================  ==========================================================
dense (embedding)       paraphrases — "我喝咖啡不加糖" vs "美式，不要糖"
lexical (BM25/FTS5)     exact names, numbers, rare tokens embeddings blur away
graph (temporal KG)     multi-hop reasoning — the query names a *person*, so every
                        edge attached to them is a candidate
profile                 the always-inject "who am I talking to" block
======================  ==========================================================

Two properties make this more than a similarity search:

**Temporal validity.** Candidates are filtered to their validity window, so the
answer reflects what is true *now* (or at a requested ``as_of`` instant), not the
union of everything ever said. Superseded memories are recalled only if the query
explicitly asks for history.

**Score transparency.** Every hit carries its per-channel contributions. That is
what makes the system tunable without guesswork — you can see whether a wrong
answer came from a bad embedding, a missing graph edge, or an over-aggressive
recency decay. Retrieval quality problems are otherwise unsolvable by inspection.

The final ordering blends the fused retrieval score with importance and an
exponential recency prior, then optionally a cross-encoder rerank. Weights and the
half-life live in :class:`core.config.MemoryConfig`.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.logging import get_logger
from core.registry import KIND_RETRIEVER, register
from core.types import (
    MemoryHit,
    MemoryItem,
    MemoryKind,
    RetrievalQuery,
    RetrievalResult,
    now_ts,
)
from core.utils import estimate_tokens, truncate_to_tokens
from llm.embed import lexical_overlap
from memory.store import MemoryStore, row_to_item

log = get_logger(__name__)

_DAY = 86400.0

#: Kinds that are always worth injecting regardless of the query, because they
#: describe the user themselves. Without this, "你是谁" style small talk would
#: retrieve nothing about the person and the assistant would forget their name.
_ALWAYS_KINDS = (MemoryKind.PROFILE,)

#: Entity names that indicate the user themself; a query mentioning "我" should
#: expand the graph from the user node.
_SELF_ALIASES = {"我", "我的", "自己", "本人", "用户", "咱", "俺"}

#: Ask for history instead of present state ("以前", "之前", "曾经", "原来").
_HISTORY_MARKERS = ("以前", "之前", "曾经", "原来", "过去", "上次", "history", "used to")


def _minmax(values: Sequence[float]) -> List[float]:
    """Scale to [0, 1]; returns all-zeros when the spread is degenerate."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 1e-9:
        # Every candidate is equally good on this channel: contribute nothing
        # rather than a constant that would act as a hidden bias.
        return [0.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


@register(KIND_RETRIEVER, "hybrid")
class HybridRetriever:
    """Multi-channel memory retrieval with temporal and explainability guarantees."""

    name = "hybrid"

    def __init__(
        self,
        store: MemoryStore,
        embedder: Any = None,
        vectors: Any = None,
        reranker: Any = None,
        *,
        config: Any = None,
        scope: str = "global",
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.vectors = vectors
        self.reranker = reranker
        cfg = getattr(config, "memory", None)
        self.cfg = cfg
        self.w_dense = float(getattr(cfg, "w_dense", 1.0))
        self.w_lexical = float(getattr(cfg, "w_lexical", 0.7))
        self.w_graph = float(getattr(cfg, "w_graph", 0.6))
        self.w_importance = float(getattr(cfg, "w_importance", 0.35))
        self.w_recency = float(getattr(cfg, "w_recency", 0.25))
        self.w_rerank = float(getattr(cfg, "w_rerank", 1.2))
        self.half_life_days = max(1.0, float(getattr(cfg, "recency_half_life_days", 21.0)))
        self.default_scope = scope

    # ==================================================================================
    # Public entry points
    # ==================================================================================
    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        started = time.perf_counter()
        scope = query.scope or self.default_scope
        top_k = max(1, query.top_k)
        debug: Dict[str, Any] = {}

        wants_history = any(marker in query.text for marker in _HISTORY_MARKERS)
        channel_rank: Dict[str, Dict[str, float]] = {"dense": {}, "lexical": {}, "graph": {}}

        # ---- candidate pool: one DB read serves every channel ------------------------
        candidates = self.store.fetch_active_candidates(scope=scope, kinds=(), as_of=query.as_of, limit=4000)
        if not candidates:
            return RetrievalResult(debug={"reason": "no memories", "channels": {k: 0 for k in channel_rank}})
        by_id: Dict[str, MemoryItem] = {item.id: item for item in candidates}
        debug["candidate_pool"] = len(candidates)

        # ---- channel 1: lexical (BM25, exact terms) ----------------------------------
        for rank, (memory_id, score) in enumerate(
            self.store.lexical_search_items(query.text, scope=scope, limit=60)
        ):
            if memory_id in by_id:
                channel_rank["lexical"][memory_id] = score

        # ---- channel 2: dense (paraphrase) -------------------------------------------
        query_vector: Optional[List[float]] = None
        if self.embedder is not None and self.vectors is not None:
            try:
                query_vector = await self.embedder.embed_one(query.text)
                if query_vector:
                    # Score more candidates than we will keep: fusion and reranking
                    # both do better with a wider net, and the cost is linear.
                    pool_size = max(60, top_k * 6)
                    for memory_id, score in await self.vectors.search(query_vector, top_k=pool_size):
                        if memory_id in by_id:
                            channel_rank["dense"][memory_id] = float(score)
            except Exception as exc:  # noqa: BLE001
                debug["dense_error"] = str(exc)[:160]
                log.debug("dense retrieval failed", extra={"error": str(exc)})

        # ---- channel 3: graph (multi-hop, temporal) ----------------------------------
        anchor_ids = await self._anchor_entities(query.text, scope=scope)
        graph_hits: Dict[str, float] = {}
        if anchor_ids:
            relations = await self.store.neighbors(anchor_ids, scope=scope, limit=80, as_of=query.as_of)
            debug["graph_anchors"] = len(anchor_ids)
            for relation in relations:
                confidence = float(relation.get("confidence") or 0.5)
                for key in ("source_memory_id",):
                    memory_id = relation.get(key)
                    if memory_id and memory_id in by_id:
                        graph_hits[memory_id] = max(graph_hits.get(memory_id, 0.0), confidence)
                # Also surface memories that *mention* the connected entities: the
                # edge itself may have no backing memory (e.g. learned from a summary).
                for entity_name in (relation.get("subject"), relation.get("object")):
                    for memory_id, item in by_id.items():
                        if item.subject and entity_name and item.subject == entity_name:
                            graph_hits.setdefault(memory_id, confidence * 0.8)
            channel_rank["graph"] = graph_hits

        # ---- always-inject profile memories ------------------------------------------
        always_ids: Set[str] = {
            item.id
            for item in candidates
            if (item.kind if isinstance(item.kind, MemoryKind) else MemoryKind(item.kind)) in _ALWAYS_KINDS
            and item.importance >= 0.5
        }

        # ---- union of channels -------------------------------------------------------
        merged: Set[str] = set(always_ids)
        for bucket in channel_rank.values():
            merged.update(bucket)
        if not merged:
            return RetrievalResult(debug={**debug, "channels": {k: len(v) for k, v in channel_rank.items()}})

        # ---- fuse --------------------------------------------------------------------
        ids = list(merged)
        dense_scores = _minmax([channel_rank["dense"].get(i, 0.0) for i in ids])
        lexical_scores = _minmax([channel_rank["lexical"].get(i, 0.0) for i in ids])
        graph_scores = _minmax([channel_rank["graph"].get(i, 0.0) for i in ids])
        now = query.as_of or now_ts()

        hits: List[MemoryHit] = []
        for index, memory_id in enumerate(ids):
            item = by_id[memory_id]
            parts: Dict[str, float] = {}
            channels: List[str] = []

            if memory_id in channel_rank["dense"]:
                parts["dense"] = self.w_dense * dense_scores[index]
                channels.append("dense")
            if memory_id in channel_rank["lexical"]:
                parts["lexical"] = self.w_lexical * lexical_scores[index]
                channels.append("lexical")
            if memory_id in channel_rank["graph"]:
                parts["graph"] = self.w_graph * graph_scores[index]
                channels.append("graph")
            if memory_id in always_ids:
                parts["profile_prior"] = 0.45
                channels.append("profile")

            # Recency: exponential decay with a configurable half-life. Applied to
            # *event time* (valid_from) so a fact learned long ago about the
            # present is not unfairly punished for being stored late.
            reference = item.valid_from or item.updated_at or item.created_at
            age_days = max(0.0, (now - reference) / _DAY)
            recency = 0.5 ** (age_days / self.half_life_days)
            parts["importance"] = self.w_importance * float(item.importance)
            parts["recency"] = self.w_recency * recency
            parts["access"] = min(0.1, 0.02 * math.log1p(item.access_count))

            base = sum(parts.values())
            hits.append(MemoryHit(item=item, score=base, parts=parts, channels=channels))

        # ---- optional cross-encoder rerank ------------------------------------------
        rerank_pool = min(len(hits), max(top_k * 4, 24))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        head, tail = hits[:rerank_pool], hits[rerank_pool:]
        if self.reranker is not None and head:
            try:
                documents = [self._render_for_rerank(hit.item) for hit in head]
                ranked = await self.reranker.rerank(query.text, documents, top_k=len(documents))
                if ranked:
                    # ``ranked`` is ordered by the reranker's own score, so the
                    # normalized values must be computed in that same order before
                    # being attributed back to the original document indices.
                    normalized = _minmax([score for _index, score in ranked])
                    for (index, _raw), norm in zip(ranked, normalized):
                        if 0 <= index < len(head):
                            hit = head[index]
                            # Blend rather than replace: rerankers are strong but
                            # occasionally confidently wrong on short memories.
                            hit.parts["rerank"] = self.w_rerank * norm
                            hit.score = sum(hit.parts.values())
                            if "rerank" not in hit.channels:
                                hit.channels.append("rerank")
            except Exception as exc:  # noqa: BLE001
                debug["rerank_error"] = str(exc)[:160]
                log.debug("rerank failed", extra={"error": str(exc)})

        hits = head + tail
        hits.sort(key=lambda hit: hit.score, reverse=True)

        # ---- MMR-lite diversity pass -------------------------------------------------
        # Two near-identical memories in the prompt waste budget and bias the model.
        selected = self._diversify(hits, top_k)

        # ---- history handling -------------------------------------------------------
        if wants_history:
            historical = await self._historical_hits(query, scope=scope, limit=max(3, top_k // 3))
            known = {hit.item.id for hit in selected}
            selected.extend(hit for hit in historical if hit.item.id not in known)

        # ---- assemble the injected block within the token budget --------------------
        profile_lines = [f"{entry['key']}：{entry['value']}" for entry in await self.store.get_profile(scope=scope)]
        rendered, used, included = self._render_block(
            selected, profile_lines=profile_lines, budget=query.token_budget
        )
        if included:
            try:
                await self.store.touch_memories([hit.item.id for hit in included])
            except Exception as exc:  # noqa: BLE001
                log.debug("touching memories failed", extra={"error": str(exc)})

        debug["channels"] = {key: len(value) for key, value in channel_rank.items()}
        debug["merged"] = len(merged)
        debug["selected"] = len(included)
        debug["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        debug["half_life_days"] = self.half_life_days

        return RetrievalResult(
            hits=included,
            profile_lines=profile_lines,
            rendered=rendered,
            tokens_used=used,
            debug=debug,
        )

    # ==================================================================================
    # Internals
    # ==================================================================================
    async def _anchor_entities(self, text: str, *, scope: str, limit: int = 10) -> List[str]:
        """Find graph nodes named (directly or by alias) in the query."""
        mentions: List[str] = []
        for alias in _SELF_ALIASES:
            if alias in text:
                mentions.append("用户")
                break
        try:
            entities = await self.store.list_entities(scope=scope, limit=300, min_mentions=1)
        except Exception as exc:  # noqa: BLE001
            log.debug("entity listing failed", extra={"error": str(exc)})
            return []
        anchors: List[str] = []
        for entity in entities:
            names = [entity.name, *entity.aliases]
            if any(name and name in text for name in names):
                anchors.append(entity.id)
                if len(anchors) >= limit:
                    break
        # Resolve the self-mention to a concrete node id.
        if "用户" in mentions:
            resolved = await self.store.resolve_entities(["用户"], scope=scope)
            if resolved.get("用户"):
                anchors.append(resolved["用户"])
        return list(dict.fromkeys(anchors))

    async def _historical_hits(
        self, query: RetrievalQuery, *, scope: str, limit: int = 5
    ) -> List[MemoryHit]:
        """Retrieve superseded/expired memories when the user asks about the past."""
        rows = self.store.db.query(
            "SELECT * FROM memory_items WHERE scope=? AND superseded_by IS NOT NULL "
            "ORDER BY updated_at DESC LIMIT 200",
            (scope,),
        )
        out: List[MemoryHit] = []
        for row in rows:
            item = row_to_item(row)
            score = lexical_overlap(query.text, item.content)
            if score <= 0:
                continue
            out.append(
                MemoryHit(
                    item=item,
                    score=score,
                    parts={"historical": score},
                    channels=["history"],
                )
            )
        out.sort(key=lambda hit: hit.score, reverse=True)
        return out[:limit]

    @staticmethod
    def _render_for_rerank(item: MemoryItem) -> str:
        prefix = f"[{item.kind.value if isinstance(item.kind, MemoryKind) else item.kind}]"
        return f"{prefix} {item.content}"

    @staticmethod
    def _diversify(hits: List[MemoryHit], top_k: int, *, threshold: float = 0.9) -> List[MemoryHit]:
        """Greedy near-duplicate suppression over the final list."""
        selected: List[MemoryHit] = []
        for hit in hits:
            if len(selected) >= top_k:
                break
            duplicate = False
            for kept in selected:
                if hit.item.kind == kept.item.kind and lexical_overlap(hit.item.content, kept.item.content) > threshold:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(hit)
        # If dedup was too aggressive, top up from the remainder.
        if len(selected) < top_k:
            for hit in hits:
                if hit not in selected:
                    selected.append(hit)
                if len(selected) >= top_k:
                    break
        return selected

    def _render_block(
        self,
        hits: Sequence[MemoryHit],
        *,
        profile_lines: Sequence[str],
        budget: int,
    ) -> Tuple[str, int, List[MemoryHit]]:
        """Render the memory block, dropping the weakest items until it fits."""
        included: List[MemoryHit] = []
        used = 0
        lines: List[str] = []

        if profile_lines:
            header = "【关于用户的档案】"
            body = "\n".join(f"- {line}" for line in profile_lines[:20])
            block = f"{header}\n{body}"
            cost = estimate_tokens(block)
            if cost <= budget:
                lines.append(block)
                used += cost

        grouped: Dict[str, List[str]] = {}
        group_titles = {
            MemoryKind.FACT.value: "【长期记忆：事实】",
            MemoryKind.PREFERENCE.value: "【长期记忆：偏好】",
            MemoryKind.PROFILE.value: "【长期记忆：档案】",
            MemoryKind.RELATION.value: "【长期记忆：关系】",
            MemoryKind.EPISODE.value: "【长期记忆：相关往事】",
            MemoryKind.SUMMARY.value: "【长期记忆：往期摘要】",
            MemoryKind.REFLECTION.value: "【长期记忆：洞察】",
        }
        group_order = [
            MemoryKind.PROFILE.value,
            MemoryKind.PREFERENCE.value,
            MemoryKind.FACT.value,
            MemoryKind.RELATION.value,
            MemoryKind.EPISODE.value,
            MemoryKind.SUMMARY.value,
            MemoryKind.REFLECTION.value,
        ]

        for hit in hits:
            kind = hit.item.kind.value if isinstance(hit.item.kind, MemoryKind) else str(hit.item.kind)
            if kind == MemoryKind.PROFILE.value and profile_lines:
                # Already covered by the profile block unless it adds detail.
                if hit.item.content in " ".join(profile_lines):
                    continue
            line = f"- {hit.item.content}"
            cost = estimate_tokens(line) + 1
            if used + cost > budget:
                continue
            grouped.setdefault(kind, []).append(line)
            used += cost
            included.append(hit)

        for kind in group_order:
            entries = grouped.get(kind)
            if not entries:
                continue
            block = f"{group_titles.get(kind, '【长期记忆】')}\n" + "\n".join(entries)
            lines.append(block)

        rendered = "\n\n".join(lines).strip()
        return rendered, used, included

    # ==================================================================================
    # Chat-history search (the "search everything I ever said" feature)
    # ==================================================================================
    async def search_history(
        self, text: str, *, scope: str = "", conversation_id: str = "", limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Full-text search across stored messages, ranked semantically when possible."""
        messages = await self.store.search_messages(
            text, scope=scope, limit=max(limit * 3, 30), conversation_id=conversation_id
        )
        if not messages:
            return []
        if self.embedder is not None:
            try:
                vectors = await self.embedder.embed([text, *[m["content"] for m in messages]])
                if vectors:
                    query_vec, *rest = vectors
                    from llm.embed import cosine as _cos

                    for message, vector in zip(messages, rest):
                        message["semantic"] = round(_cos(query_vec, vector), 4)
                    messages.sort(key=lambda m: m.get("semantic", 0), reverse=True)
            except Exception as exc:  # noqa: BLE001
                log.debug("history semantic ranking failed", extra={"error": str(exc)})
        return messages[:limit]


__all__ = ["HybridRetriever"]
