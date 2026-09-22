"""Layered long-term memory: schema-aware persistence and retrieval access.

This module is the only place that knows the SQL. Everything above it
(extraction, retrieval, consolidation, the API) works with domain objects, which
is what makes the storage backend swappable: implementing the same methods over
Postgres/pgvector is a self-contained job.

The layers, and why each exists:

======================  ==========================================================
layer                   purpose
======================  ==========================================================
``messages`` + FTS      episodic record; answers "what did we actually say?"
``memory_items``        distilled facts / preferences / profile / summaries;
                        carries temporal validity + supersede chain
``entities``/``relations``  temporal knowledge graph; answers "how are these
                        things connected, and since when?"
``profiles``            the always-injected "who am I talking to" block
``kv_store``            cursor state for the consolidator, feature flags
======================  ==========================================================
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.logging import get_logger
from core.registry import KIND_MEMORY_STORE, register
from core.types import (
    Entity,
    MemoryItem,
    MemoryKind,
    Message,
    Role,
    new_id,
    now_ts,
)
from core.utils import estimate_tokens

log = get_logger(__name__)

_WS = re.compile(r"\s+")


def content_hash(text: str) -> str:
    """Normalized hash used for cheap exact-duplicate suppression."""
    normalized = _WS.sub(" ", (text or "").strip().lower())
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def row_to_item(row: Any) -> MemoryItem:
    """Build a :class:`MemoryItem` from a ``memory_items`` row.

    Public because the retriever needs to reconstruct retired memories that are
    not returned by the normal candidate queries.
    """
    return MemoryItem(
        id=row["id"],
        content=row["content"],
        kind=MemoryKind(row["kind"]) if row["kind"] in {k.value for k in MemoryKind} else MemoryKind.FACT,
        importance=float(row["importance"] or 0.5),
        confidence=float(row["confidence"] or 0.7),
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        last_access_at=float(row["last_access_at"]),
        access_count=int(row["access_count"] or 0),
        supersedes=row["supersedes"],
        superseded_by=row["superseded_by"],
        session_id=row["session_id"],
        persona_id=row["persona_id"],
        tags=list(json.loads(row["tags_json"] or "[]")),
        source_message_ids=list(json.loads(row["sources_json"] or "[]")),
    )


def _row_to_entity(row: Any, aliases: Sequence[str] = ()) -> Entity:
    return Entity(
        id=row["id"],
        name=row["name"],
        type=row["type"],
        aliases=tuple(aliases),
        summary=row["summary"] or "",
        importance=float(row["importance"] or 0.5),
        first_seen_at=float(row["first_seen_at"]),
        last_seen_at=float(row["last_seen_at"]),
        mention_count=int(row["mention_count"] or 1),
        scope=row["scope"],
    )


@register(KIND_MEMORY_STORE, "sqlite")
class MemoryStore:
    """All persistence for conversations, memories, the graph and the profile."""

    name = "sqlite"

    def __init__(self, db: Any, *, default_scope: str = "global") -> None:
        self.db = db
        self.default_scope = default_scope

    async def init(self) -> None:
        self.db.init()

    async def close(self) -> None:
        await self.db.aclose()

    # ==================================================================================
    # Conversations & messages
    # ==================================================================================
    async def ensure_conversation(
        self,
        conversation_id: str,
        *,
        title: str = "",
        persona_id: str = "",
        scope: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        scope = scope or self.default_scope
        ts = now_ts()

        def _write(conn: Any) -> None:
            conn.execute(
                "INSERT INTO conversations(id, title, persona_id, scope, created_at, updated_at, meta_json) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "persona_id=CASE WHEN excluded.persona_id<>'' THEN excluded.persona_id ELSE persona_id END, "
                "title=CASE WHEN excluded.title<>'' THEN excluded.title ELSE title END, "
                "updated_at=excluded.updated_at",
                (
                    conversation_id,
                    title,
                    persona_id,
                    scope,
                    ts,
                    ts,
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )

        await self.db.awrite(_write)
        return conversation_id

    async def add_message(
        self,
        conversation_id: str,
        message: Message,
        *,
        tokens: int = 0,
        persona_id: str = "",
        scope: str = "",
        audio_path: Optional[str] = None,
    ) -> str:
        scope = scope or self.default_scope
        tokens = tokens or estimate_tokens(message.content)
        meta_json = json.dumps(message.meta or {}, ensure_ascii=False, default=str)
        fts_enabled = self.db.fts_enabled

        def _write(conn: Any) -> None:
            conn.execute(
                "INSERT INTO messages(id, conversation_id, role, content, created_at, tokens, audio_path, persona_id, scope, meta_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
                (
                    message.id,
                    conversation_id,
                    message.role.value,
                    message.content,
                    message.created_at,
                    tokens,
                    audio_path,
                    persona_id,
                    scope,
                    meta_json,
                ),
            )
            if fts_enabled and message.content:
                conn.execute(
                    "INSERT INTO messages_fts(content, message_id, conversation_id, role, created_at) "
                    "VALUES(?,?,?,?,?)",
                    (message.content, message.id, conversation_id, message.role.value, message.created_at),
                )
            # turn_count counts *exchanges*, so only the user side increments it —
            # matching Session.turn_count (core/session.py) rather than doubling.
            conn.execute(
                "UPDATE conversations SET updated_at=?, turn_count=turn_count+?, token_total=token_total+? WHERE id=?",
                (
                    message.created_at,
                    1 if message.role == Role.USER else 0,
                    tokens,
                    conversation_id,
                ),
            )

        await self.db.awrite(_write)
        return message.id

    async def get_messages(
        self, conversation_id: str, *, limit: int = 200, before: Optional[float] = None
    ) -> List[Message]:
        if before is not None:
            rows = self.db.query(
                "SELECT * FROM messages WHERE conversation_id=? AND created_at<? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, before, limit),
            )
            rows = list(reversed(rows))
        else:
            rows = self.db.query(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            )
            rows = list(reversed(rows))
        return [
            Message(
                id=row["id"],
                role=Role(row["role"]),
                content=row["content"],
                created_at=float(row["created_at"]),
                meta=json.loads(row["meta_json"] or "{}"),
            )
            for row in rows
        ]

    async def recent_dialogue(
        self, conversation_id: str, turns: int = 6
    ) -> List[Tuple[str, str]]:
        """Return the last ``turns`` (user, assistant) pairs, oldest first.

        Used as extraction context so pronouns ("他", "那个") can be resolved
        against what was actually said.
        """
        rows = self.db.query(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
            (conversation_id, max(2, turns * 2)),
        )
        rows = list(reversed(rows))
        out: List[Tuple[str, str]] = []
        pending_user: Optional[str] = None
        for row in rows:
            if row["role"] == Role.USER.value:
                pending_user = row["content"]
            elif row["role"] == Role.ASSISTANT.value and pending_user is not None:
                out.append((pending_user, row["content"]))
                pending_user = None
        return out

    async def list_conversations(
        self, *, scope: str = "", limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        if scope:
            rows = self.db.query(
                "SELECT * FROM conversations WHERE scope=? AND archived=0 ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (scope, limit, offset),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM conversations WHERE archived=0 ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        return [
            {
                "id": row["id"],
                "title": row["title"] or "新对话",
                "persona_id": row["persona_id"],
                "scope": row["scope"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "turn_count": row["turn_count"],
                "token_total": row["token_total"],
                "summary": row["summary"],
            }
            for row in rows
        ]

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.query_one("SELECT * FROM conversations WHERE id=?", (conversation_id,))
        if row is None:
            return None
        return {
            "id": row["id"],
            "title": row["title"],
            "persona_id": row["persona_id"],
            "scope": row["scope"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "turn_count": row["turn_count"],
            "token_total": row["token_total"],
            "summary": row["summary"],
            "meta": json.loads(row["meta_json"] or "{}"),
        }

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        archived: Optional[bool] = None,
        persona_id: Optional[str] = None,
    ) -> None:
        fields: List[str] = []
        params: List[Any] = []
        if title is not None:
            fields.append("title=?")
            params.append(title)
        if summary is not None:
            fields.append("summary=?")
            params.append(summary)
        if archived is not None:
            fields.append("archived=?")
            params.append(1 if archived else 0)
        if persona_id is not None:
            fields.append("persona_id=?")
            params.append(persona_id)
        if not fields:
            return
        params.append(conversation_id)

        def _write(conn: Any) -> None:
            conn.execute(f"UPDATE conversations SET {', '.join(fields)} WHERE id=?", params)

        await self.db.awrite(_write)

    async def delete_conversation(self, conversation_id: str) -> None:
        def _write(conn: Any) -> None:
            if self.db.fts_enabled:
                conn.execute("DELETE FROM messages_fts WHERE conversation_id=?", (conversation_id,))
            conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))

        await self.db.awrite(_write)

    # ==================================================================================
    # Full-text search over chat history
    # ==================================================================================
    async def search_messages(
        self, query: str, *, scope: str = "", limit: int = 20, conversation_id: str = ""
    ) -> List[Dict[str, Any]]:
        """Full-text search over every stored message.

        Prefers FTS5 (ranked BM25); falls back to ``LIKE`` so search still works
        on SQLite builds without FTS5. CJK queries are additionally matched with a
        LIKE pass because ``unicode61`` tokenizes Chinese as one long token —
        without this, searching "咖啡" inside "我喜欢美式咖啡" would miss.
        """
        query = (query or "").strip()
        if not query:
            return []
        results: Dict[str, Dict[str, Any]] = {}
        fts_query = _fts_query(query)

        if self.db.fts_enabled and fts_query:
            try:
                sql = (
                    "SELECT m.*, bm25(messages_fts) AS rank FROM messages_fts "
                    "JOIN messages m ON m.id = messages_fts.message_id "
                    "WHERE messages_fts MATCH ?"
                )
                params: List[Any] = [fts_query]
                if scope:
                    sql += " AND m.scope=?"
                    params.append(scope)
                if conversation_id:
                    sql += " AND m.conversation_id=?"
                    params.append(conversation_id)
                sql += " ORDER BY rank LIMIT ?"
                params.append(limit)
                for row in self.db.query(sql, params):
                    results[row["id"]] = _message_row_to_hit(row, source="fts")
            except Exception as exc:  # noqa: BLE001 - malformed FTS syntax must not 500
                log.debug("fts query failed, falling back to LIKE", extra={"error": str(exc), "query": query})

        if len(results) < limit:
            sql = "SELECT * FROM messages WHERE content LIKE ?"
            params = [f"%{query}%"]
            if scope:
                sql += " AND scope=?"
                params.append(scope)
            if conversation_id:
                sql += " AND conversation_id=?"
                params.append(conversation_id)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            for row in self.db.query(sql, params):
                results.setdefault(row["id"], _message_row_to_hit(row, source="like"))

        ordered = sorted(results.values(), key=lambda item: item["created_at"], reverse=True)
        return ordered[:limit]

    # ==================================================================================
    # Memory items
    # ==================================================================================
    async def add_memory(
        self,
        *,
        content: str,
        kind: MemoryKind | str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        importance: float = 0.5,
        confidence: float = 0.7,
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        session_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        scope: str = "",
        tags: Optional[Sequence[str]] = None,
        sources: Optional[Sequence[str]] = None,
        supersedes: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        dedupe: bool = True,
    ) -> Optional[str]:
        """Insert a memory item, returning its id (or ``None`` when deduplicated).

        Deduplication collapses exact re-statements by ``content_hash``. It is
        skipped when the caller explicitly passes ``supersedes``: a replacement
        memory must always be created, even if its text matches (the two rows carry
        different validity windows, and collapsing them would erase the history the
        temporal chain exists to preserve).
        """
        content = (content or "").strip()
        if not content:
            return None
        kind_value = kind.value if isinstance(kind, MemoryKind) else str(kind)
        digest = content_hash(content)
        ts = now_ts()
        scope_value = scope or self.default_scope

        if dedupe and not supersedes:
            existing = self.db.query_one(
                "SELECT id FROM memory_items WHERE content_hash=? AND scope=? AND superseded_by IS NULL",
                (digest, scope_value),
            )
            if existing is not None:
                def _touch(conn: Any) -> None:
                    conn.execute(
                        "UPDATE memory_items SET access_count=access_count+1, updated_at=?, last_access_at=? WHERE id=?",
                        (ts, ts, existing["id"]),
                    )

                await self.db.awrite(_touch)
                return None

        item_id = new_id("mem_")
        fts_enabled = self.db.fts_enabled

        def _write(conn: Any) -> None:
            if supersedes:
                conn.execute(
                    "UPDATE memory_items SET superseded_by=?, valid_to=?, updated_at=? WHERE id=?",
                    (item_id, ts, ts, supersedes),
                )
            conn.execute(
                "INSERT INTO memory_items(id, kind, content, subject, predicate, object, importance, confidence, "
                "valid_from, valid_to, created_at, updated_at, last_access_at, access_count, supersedes, superseded_by, "
                "session_id, persona_id, scope, tags_json, sources_json, meta_json, content_hash) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item_id,
                    kind_value,
                    content,
                    subject,
                    predicate,
                    object,
                    float(importance),
                    float(confidence),
                    valid_from if valid_from is not None else ts,
                    valid_to,
                    ts,
                    ts,
                    ts,
                    0,
                    supersedes,
                    None,
                    session_id,
                    persona_id,
                    scope_value,
                    json.dumps(list(tags or []), ensure_ascii=False),
                    json.dumps(list(sources or []), ensure_ascii=False),
                    json.dumps(meta or {}, ensure_ascii=False),
                    digest,
                ),
            )
            if fts_enabled:
                conn.execute(
                    "INSERT INTO memory_fts(content, memory_id) VALUES(?,?)", (content, item_id)
                )

        await self.db.awrite(_write)
        return item_id

    async def update_memory(
        self,
        memory_id: str,
        *,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        superseded_by: Optional[str] = None,
        valid_to: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        fields: List[str] = []
        params: List[Any] = []
        if content is not None:
            fields.extend(["content=?", "content_hash=?"])
            params.extend([content, content_hash(content)])
        if importance is not None:
            fields.append("importance=?")
            params.append(float(importance))
        if confidence is not None:
            fields.append("confidence=?")
            params.append(float(confidence))
        if superseded_by is not None:
            fields.append("superseded_by=?")
            params.append(superseded_by)
        if valid_to is not None:
            fields.append("valid_to=?")
            params.append(valid_to)
        if tags is not None:
            fields.append("tags_json=?")
            params.append(json.dumps(list(tags), ensure_ascii=False))
        if meta is not None:
            fields.append("meta_json=?")
            params.append(json.dumps(meta, ensure_ascii=False))
        if not fields:
            return False
        fields.append("updated_at=?")
        params.append(now_ts())
        params.append(memory_id)
        fts_enabled = self.db.fts_enabled

        def _write(conn: Any) -> bool:
            # Honour the affected-row count: an UPDATE that matches nothing must
            # report failure so the API can answer 404 instead of a misleading 200.
            cursor = conn.execute(f"UPDATE memory_items SET {', '.join(fields)} WHERE id=?", params)
            if cursor.rowcount == 0:
                return False
            if fts_enabled and content is not None:
                conn.execute("DELETE FROM memory_fts WHERE memory_id=?", (memory_id,))
                conn.execute("INSERT INTO memory_fts(content, memory_id) VALUES(?,?)", (content, memory_id))
            return True

        return bool(await self.db.awrite(_write))

    async def retire_memory(self, memory_id: str, superseded_by: Optional[str] = None) -> bool:
        """Close a memory's validity window (soft delete) instead of erasing it.

        Keeping the closed window is what lets the system answer "what did I use
        to think?" and prevents a contradiction from silently reappearing.
        """
        ts = now_ts()

        def _write(conn: Any) -> None:
            conn.execute(
                "UPDATE memory_items SET valid_to=?, superseded_by=?, updated_at=? WHERE id=? AND valid_to IS NULL",
                (ts, superseded_by, ts, memory_id),
            )

        await self.db.awrite(_write)
        return True

    async def delete_memory(self, memory_id: str, *, hard: bool = False) -> bool:
        """Remove a memory. Soft by default (see :meth:`retire_memory`)."""
        if not hard:
            await self.retire_memory(memory_id)
            return True
        fts_enabled = self.db.fts_enabled

        def _write(conn: Any) -> None:
            conn.execute("DELETE FROM memory_vectors WHERE memory_id=?", (memory_id,))
            if fts_enabled:
                conn.execute("DELETE FROM memory_fts WHERE memory_id=?", (memory_id,))
            conn.execute("DELETE FROM memory_items WHERE id=?", (memory_id,))

        await self.db.awrite(_write)
        return True

    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        row = self.db.query_one("SELECT * FROM memory_items WHERE id=?", (memory_id,))
        return row_to_item(row) if row else None

    async def get_memories(self, ids: Sequence[str]) -> Dict[str, MemoryItem]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.db.query(f"SELECT * FROM memory_items WHERE id IN ({placeholders})", list(ids))
        return {row["id"]: row_to_item(row) for row in rows}

    async def list_memories(
        self,
        *,
        scope: str = "",
        kinds: Sequence[MemoryKind | str] = (),
        include_retired: bool = False,
        min_importance: float = 0.0,
        limit: int = 100,
        offset: int = 0,
        order: str = "recent",
    ) -> List[MemoryItem]:
        clauses: List[str] = []
        params: List[Any] = []
        if scope:
            clauses.append("scope=?")
            params.append(scope)
        if kinds:
            values = [k.value if isinstance(k, MemoryKind) else str(k) for k in kinds]
            clauses.append(f"kind IN ({','.join('?' for _ in values)})")
            params.extend(values)
        if not include_retired:
            clauses.append("superseded_by IS NULL AND (valid_to IS NULL OR valid_to > ?)")
            params.append(now_ts())
        if min_importance > 0:
            clauses.append("importance >= ?")
            params.append(float(min_importance))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = {
            "recent": "created_at DESC",
            "importance": "importance DESC, created_at DESC",
            "accessed": "last_access_at DESC",
        }.get(order, "created_at DESC")
        params.extend([limit, offset])
        rows = self.db.query(
            f"SELECT * FROM memory_items {where} ORDER BY {order_sql} LIMIT ? OFFSET ?", params
        )
        return [row_to_item(row) for row in rows]

    async def count_memories(self, *, scope: str = "", include_retired: bool = False) -> Dict[str, int]:
        clause = ""
        params: List[Any] = []
        if scope:
            clause = "WHERE scope=?"
            params.append(scope)
        if not include_retired:
            clause = (clause + " AND " if clause else "WHERE ") + "superseded_by IS NULL"
        rows = self.db.query(f"SELECT kind, COUNT(*) AS n FROM memory_items {clause} GROUP BY kind", params)
        counts = {row["kind"]: int(row["n"]) for row in rows}
        counts["total"] = sum(counts.values())
        return counts

    # -- candidate generation for hybrid retrieval -------------------------------------
    def fetch_active_candidates(
        self,
        *,
        scope: str = "",
        kinds: Sequence[str] = (),
        as_of: Optional[float] = None,
        limit: int = 4000,
    ) -> List[MemoryItem]:
        """Everything plausibly retrievable, for lexical/vector/graph scoring.

        ``as_of`` performs a *point-in-time* query: only memories whose validity
        window covers that instant. That is what makes the graph temporal rather
        than merely timestamped.
        """
        clauses: List[str] = ["superseded_by IS NULL"] if as_of is None else []
        params: List[Any] = []
        if as_of is not None:
            clauses.append("(valid_from IS NULL OR valid_from <= ?)")
            params.append(as_of)
            clauses.append("(valid_to IS NULL OR valid_to > ?)")
            params.append(as_of)
        else:
            clauses.append("(valid_to IS NULL OR valid_to > ?)")
            params.append(now_ts())
        if scope:
            clauses.append("scope=?")
            params.append(scope)
        if kinds:
            clauses.append(f"kind IN ({','.join('?' for _ in kinds)})")
            params.extend(list(kinds))
        params.append(limit)
        rows = self.db.query(
            f"SELECT * FROM memory_items WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        return [row_to_item(row) for row in rows]

    def lexical_search_items(self, query: str, *, scope: str = "", limit: int = 60) -> List[Tuple[str, float]]:
        """BM25-ranked item search through FTS5 (falls back to LIKE)."""
        if not query.strip():
            return []
        out: List[Tuple[str, float]] = []
        fts_query = _fts_query(query)
        if self.db.fts_enabled and fts_query:
            try:
                sql = (
                    "SELECT m.memory_id AS mid, bm25(memory_fts) AS rank FROM memory_fts "
                    "JOIN memory_items m ON m.id = memory_fts.memory_id "
                    "WHERE memory_fts MATCH ? AND m.superseded_by IS NULL AND m.valid_to IS NULL"
                )
                params: List[Any] = [fts_query]
                if scope:
                    sql += " AND m.scope=?"
                    params.append(scope)
                sql += " ORDER BY rank LIMIT ?"
                params.append(limit)
                for row in self.db.query(sql, params):
                    # bm25() is negative-is-better; convert to a positive 0..1-ish score.
                    out.append((row["mid"], 1.0 / (1.0 + abs(float(row["rank"])))))
            except Exception as exc:  # noqa: BLE001
                log.debug("memory fts failed", extra={"error": str(exc)})
        if len(out) < limit:
            sql = "SELECT id FROM memory_items WHERE content LIKE ? AND superseded_by IS NULL AND valid_to IS NULL"
            params = [f"%{query}%"]
            if scope:
                sql += " AND scope=?"
                params.append(scope)
            sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
            params.append(limit)
            seen = {mid for mid, _ in out}
            for row in self.db.query(sql, params):
                if row["id"] not in seen:
                    out.append((row["id"], 0.4))
        return out[:limit]

    async def touch_memories(self, ids: Sequence[str]) -> None:
        """Record access — the signal that drives usage-based strengthening."""
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        ts = now_ts()

        def _write(conn: Any) -> None:
            conn.execute(
                f"UPDATE memory_items SET access_count=access_count+1, last_access_at=? WHERE id IN ({placeholders})",
                [ts, *ids],
            )

        await self.db.awrite(_write)

    # ==================================================================================
    # Entities & relations (the graph layer)
    # ==================================================================================
    async def upsert_entity(
        self,
        name: str,
        *,
        type: str = "concept",
        aliases: Sequence[str] = (),
        summary: str = "",
        importance: float = 0.5,
        scope: str = "",
    ) -> str:
        name = (name or "").strip()
        if not name:
            raise ValueError("entity name must not be empty")
        scope_value = scope or self.default_scope
        ts = now_ts()
        entity_id = new_id("ent_")

        def _write(conn: Any) -> Dict[str, Any]:
            row = conn.execute(
                "SELECT id, mention_count, summary FROM entities WHERE scope=? AND name=?",
                (scope_value, name),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO entities(id, name, type, summary, importance, first_seen_at, last_seen_at, mention_count, scope) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (entity_id, name, type, summary, float(importance), ts, ts, 1, scope_value),
                )
                resolved = entity_id
            else:
                resolved = row["id"]
                merged_summary = summary or row["summary"] or ""
                # SQLite's two-argument max() keeps the stronger importance signal.
                conn.execute(
                    "UPDATE entities SET last_seen_at=?, mention_count=mention_count+1, "
                    "importance=MAX(importance, ?), summary=?, type=CASE WHEN ?<>'' THEN ? ELSE type END WHERE id=?",
                    (ts, float(importance), merged_summary, type, type, resolved),
                )
            for alias in {*(aliases or ()), name}:
                alias = (alias or "").strip()
                if not alias:
                    continue
                conn.execute(
                    "INSERT INTO entity_aliases(entity_id, alias, weight) VALUES(?,?,1.0) "
                    "ON CONFLICT(entity_id, alias) DO NOTHING",
                    (resolved, alias),
                )
            return {"id": resolved, "created": row is None}

        result = await self.db.awrite(_write)
        return str(result["id"])

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        row = self.db.query_one("SELECT * FROM entities WHERE id=?", (entity_id,))
        if row is None:
            return None
        aliases = [
            r["alias"] for r in self.db.query("SELECT alias FROM entity_aliases WHERE entity_id=?", (entity_id,))
        ]
        return _row_to_entity(row, aliases)

    async def resolve_entities(self, names: Sequence[str], *, scope: str = "") -> Dict[str, str]:
        """Map free-text mentions to entity ids (exact and alias match)."""
        scope_value = scope or self.default_scope
        out: Dict[str, str] = {}
        for name in names:
            name = (name or "").strip()
            if not name:
                continue
            row = self.db.query_one(
                "SELECT id FROM entities WHERE scope=? AND name=?", (scope_value, name)
            )
            if row is None:
                row = self.db.query_one(
                    "SELECT e.id AS id FROM entity_aliases a JOIN entities e ON e.id=a.entity_id "
                    "WHERE a.alias=? AND e.scope=? LIMIT 1",
                    (name, scope_value),
                )
            if row is not None:
                out[name] = row["id"]
        return out

    async def list_entities(
        self, *, scope: str = "", limit: int = 200, min_mentions: int = 1, query: str = ""
    ) -> List[Entity]:
        clauses = ["mention_count >= ?"]
        params: List[Any] = [min_mentions]
        if scope:
            clauses.append("scope=?")
            params.append(scope)
        if query:
            clauses.append("(name LIKE ? OR summary LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        params.append(limit)
        rows = self.db.query(
            f"SELECT * FROM entities WHERE {' AND '.join(clauses)} ORDER BY importance DESC, mention_count DESC, last_seen_at DESC LIMIT ?",
            params,
        )
        entities: List[Entity] = []
        for row in rows:
            aliases = [
                r["alias"]
                for r in self.db.query("SELECT alias FROM entity_aliases WHERE entity_id=?", (row["id"],))
            ]
            entities.append(_row_to_entity(row, aliases))
        return entities

    async def add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        *,
        confidence: float = 0.7,
        valid_from: Optional[float] = None,
        source_memory_id: Optional[str] = None,
        scope: str = "",
    ) -> str:
        """Add a graph edge, strengthening it when the same triple reappears."""
        scope_value = scope or self.default_scope
        ts = now_ts()
        predicate = (predicate or "").strip() or "related_to"
        relation_id = new_id("rel_")

        def _write(conn: Any) -> str:
            row = conn.execute(
                "SELECT id, confidence FROM relations WHERE scope=? AND subject_id=? AND predicate=? AND object_id=?",
                (scope_value, subject_id, predicate, object_id),
            ).fetchone()
            if row is not None:
                # Repeated observation increases confidence (bounded).
                new_conf = min(0.99, max(float(row["confidence"]), float(confidence)) + 0.02)
                conn.execute(
                    "UPDATE relations SET confidence=?, updated_at=?, valid_to=NULL WHERE id=?",
                    (new_conf, ts, row["id"]),
                )
                return str(row["id"])
            conn.execute(
                "INSERT INTO relations(id, subject_id, predicate, object_id, confidence, valid_from, created_at, updated_at, "
                "source_memory_id, scope) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    relation_id,
                    subject_id,
                    predicate,
                    object_id,
                    float(confidence),
                    valid_from if valid_from is not None else ts,
                    ts,
                    ts,
                    source_memory_id,
                    scope_value,
                ),
            )
            return relation_id

        return await self.db.awrite(_write)

    async def neighbors(
        self, entity_ids: Sequence[str], *, scope: str = "", limit: int = 60, as_of: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """One-hop graph expansion, returning both directions."""
        if not entity_ids:
            return []
        scope_value = scope or self.default_scope
        # The entity id list is bound twice (subject side and object side), hence
        # the duplicated parameters in the SQL below.
        id_list = list(entity_ids)
        placeholders = ",".join("?" for _ in id_list)
        params: List[Any] = [*id_list, *id_list, scope_value]
        sql = (
            "SELECT r.*, es.name AS subject_name, eo.name AS object_name FROM relations r "
            "JOIN entities es ON es.id=r.subject_id JOIN entities eo ON eo.id=r.object_id "
            f"WHERE (r.subject_id IN ({placeholders}) OR r.object_id IN ({placeholders})) AND r.scope=?"
        )
        if as_of is not None:
            sql += " AND (r.valid_from IS NULL OR r.valid_from <= ?) AND (r.valid_to IS NULL OR r.valid_to > ?)"
            params.extend([as_of, as_of])
        else:
            sql += " AND (r.valid_to IS NULL OR r.valid_to > ?)"
            params.append(now_ts())
        sql += " ORDER BY r.confidence DESC, r.updated_at DESC LIMIT ?"
        params.append(limit)
        return [
            {
                "id": row["id"],
                "subject_id": row["subject_id"],
                "subject": row["subject_name"],
                "predicate": row["predicate"],
                "object_id": row["object_id"],
                "object": row["object_name"],
                "confidence": float(row["confidence"]),
                "valid_from": row["valid_from"],
                "valid_to": row["valid_to"],
                "updated_at": row["updated_at"],
                "source_memory_id": row["source_memory_id"],
            }
            for row in self.db.query(sql, params)
        ]

    async def relations_for_entities(
        self, entity_ids: Sequence[str], *, scope: str = ""
    ) -> List[Dict[str, Any]]:
        return await self.neighbors(entity_ids, scope=scope, limit=200)

    # ==================================================================================
    # Profile (always-injected facts about the user)
    # ==================================================================================
    async def upsert_profile(
        self,
        key: str,
        value: str,
        *,
        subject: str = "user",
        confidence: float = 0.7,
        importance: float = 0.6,
        scope: str = "",
        source_memory_id: Optional[str] = None,
    ) -> None:
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return
        scope_value = scope or self.default_scope
        ts = now_ts()
        profile_id = new_id("prf_")

        def _write(conn: Any) -> None:
            conn.execute(
                "INSERT INTO profiles(id, scope, subject, key, value, confidence, importance, created_at, updated_at, source_memory_id) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(scope, subject, key) DO UPDATE SET value=excluded.value, "
                "confidence=MAX(profiles.confidence, excluded.confidence), updated_at=excluded.updated_at, "
                "source_memory_id=excluded.source_memory_id",
                (
                    profile_id,
                    scope_value,
                    subject,
                    key,
                    value,
                    float(confidence),
                    float(importance),
                    ts,
                    ts,
                    source_memory_id,
                ),
            )

        await self.db.awrite(_write)

    async def get_profile(self, *, scope: str = "", subject: str = "user", limit: int = 40) -> List[Dict[str, Any]]:
        scope_value = scope or self.default_scope
        rows = self.db.query(
            "SELECT * FROM profiles WHERE scope=? AND subject=? ORDER BY importance DESC, updated_at DESC LIMIT ?",
            (scope_value, subject, limit),
        )
        return [
            {
                "key": row["key"],
                "value": row["value"],
                "confidence": float(row["confidence"]),
                "importance": float(row["importance"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        ]

    async def delete_profile(self, key: str, *, scope: str = "", subject: str = "user") -> None:
        def _write(conn: Any) -> None:
            conn.execute(
                "DELETE FROM profiles WHERE scope=? AND subject=? AND key=?",
                (scope or self.default_scope, subject, key),
            )

        await self.db.awrite(_write)

    # ==================================================================================
    # Key/value bookkeeping
    # ==================================================================================
    async def kv_set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        ts = now_ts()

        def _write(conn: Any) -> None:
            conn.execute(
                "INSERT INTO kv_store(key, value_json, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
                (key, payload, ts),
            )

        await self.db.awrite(_write)

    async def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.db.query_one("SELECT value_json FROM kv_store WHERE key=?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:  # pragma: no cover
            return default

    # ==================================================================================
    # Turn bookkeeping (idempotent extraction)
    # ==================================================================================
    def turn_processed(self, message_id: str) -> bool:
        row = self.db.query_one("SELECT 1 AS n FROM turn_log WHERE message_id=?", (message_id,))
        return row is not None

    async def mark_turn(
        self,
        *,
        conversation_id: str,
        message_id: str,
        assistant_id: Optional[str] = None,
        items_created: int = 0,
        status: str = "done",
        error: str = "",
    ) -> None:
        ts = now_ts()

        def _write(conn: Any) -> None:
            conn.execute(
                "INSERT INTO turn_log(id, conversation_id, message_id, assistant_id, processed_at, items_created, status, error) "
                "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(message_id) DO UPDATE SET "
                "items_created=excluded.items_created, status=excluded.status, error=excluded.error, processed_at=excluded.processed_at",
                (new_id("turn_"), conversation_id, message_id, assistant_id, ts, items_created, status, error[:500]),
            )

        await self.db.awrite(_write)

    async def stats(self) -> Dict[str, Any]:
        return {
            "conversations": int(self.db.scalar("SELECT COUNT(*) FROM conversations", default=0) or 0),
            "messages": int(self.db.scalar("SELECT COUNT(*) FROM messages", default=0) or 0),
            "memories": int(
                self.db.scalar("SELECT COUNT(*) FROM memory_items WHERE superseded_by IS NULL", default=0) or 0
            ),
            "retired_memories": int(
                self.db.scalar("SELECT COUNT(*) FROM memory_items WHERE superseded_by IS NOT NULL", default=0) or 0
            ),
            "entities": int(self.db.scalar("SELECT COUNT(*) FROM entities", default=0) or 0),
            "relations": int(self.db.scalar("SELECT COUNT(*) FROM relations", default=0) or 0),
            "profile_keys": int(self.db.scalar("SELECT COUNT(*) FROM profiles", default=0) or 0),
            "vectors": int(self.db.scalar("SELECT COUNT(*) FROM memory_vectors", default=0) or 0),
            "fts_enabled": self.db.fts_enabled,
            "db_path": str(getattr(self.db, "path", "")),
        }


def _message_row_to_hit(row: Any, *, source: str) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "role": row["role"],
        "content": row["content"],
        "created_at": float(row["created_at"]),
        "scope": row["scope"],
        "meta": json.loads(row["meta_json"] or "{}"),
        "source": source,
    }


#: Characters that would break FTS5 MATCH syntax if passed through verbatim.
_FTS_SPECIAL = re.compile(r'["*()\[\]{}:^~\\\-]')


def _fts_query(text: str) -> str:
    """Turn a natural-language query into a safe FTS5 MATCH expression.

    Quoted terms are OR-ed, which matches user expectation ("coffee" OR "咖啡")
    better than the implicit AND: a memory containing either word is relevant.
    """
    text = (text or "").strip()
    if not text:
        return ""
    cleaned = _FTS_SPECIAL.sub(" ", text)
    terms = [t for t in cleaned.split() if t]
    if not terms:
        return ""
    # CJK runs are kept whole; long latin tokens get a prefix wildcard so
    # "archi" still finds "architecture".
    rendered: List[str] = []
    for term in terms[:12]:
        if re.fullmatch(r"[A-Za-z0-9_]+", term):
            rendered.append(f'"{term}"*')
        else:
            rendered.append(f'"{term}"')
    return " OR ".join(rendered)


__all__ = ["MemoryStore", "content_hash", "row_to_item"]
