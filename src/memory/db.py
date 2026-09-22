"""SQLite connection management, schema and migrations.

Design notes
------------
**Why SQLite.** The requirement is a single-machine, single-user assistant whose
state must be inspectable, backup-able (copy one file) and dependency-free. Every
serious "agent memory" service (Mem0, Zep) is a database wrapped in retrieval
logic; here SQLite *is* the database and the retrieval logic is ours.

**What is stored.**

* ``conversations`` / ``messages`` / ``messages_fts`` — the episodic record.
  FTS5 gives us true full-text search over the whole chat history, including
  Chinese (``unicode61`` tokenizer + trigram fallback for CJK substrings).
* ``memory_items`` — the semantic layer: facts, preferences, profile attributes,
  relation edges, summaries and reflections. Carries an *event-time validity
  window* (``valid_from``/``valid_to``) plus a supersede chain, which is what
  turns a flat fact table into a temporal knowledge graph.
* ``entities`` / ``entity_aliases`` / ``relations`` — the graph layer, with
  confidence and validity so edges can expire instead of being deleted.
* ``profiles`` — the distilled "who is this person" projection injected into
  every system prompt.
* ``memory_vectors`` — float32 blobs, one row per memory item, scored with numpy
  when available. Keeping vectors in the same file as the metadata means one
  transaction keeps them consistent; the ``VectorStore`` protocol allows moving
  to pgvector/FAISS without touching memory logic.
* ``turn_log`` — extraction bookkeeping so a retry never double-inserts.

**Migrations.** ``schema_version`` plus an ordered list of DDL steps. Adding a
column later is one tuple; user databases upgrade in place on start.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from core.logging import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 1

_BASE_DDL: Tuple[str, ...] = (
    # ---------------------------------------------------------------- conversations
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id            TEXT PRIMARY KEY,
        title         TEXT NOT NULL DEFAULT '',
        persona_id    TEXT NOT NULL DEFAULT '',
        scope         TEXT NOT NULL DEFAULT 'global',
        created_at    REAL NOT NULL,
        updated_at    REAL NOT NULL,
        turn_count    INTEGER NOT NULL DEFAULT 0,
        token_total   INTEGER NOT NULL DEFAULT 0,
        archived      INTEGER NOT NULL DEFAULT 0,
        summary       TEXT NOT NULL DEFAULT '',
        meta_json     TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_conv_scope ON conversations(scope, updated_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS messages (
        id            TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role          TEXT NOT NULL,
        content       TEXT NOT NULL,
        created_at    REAL NOT NULL,
        tokens        INTEGER NOT NULL DEFAULT 0,
        audio_path    TEXT,
        persona_id    TEXT NOT NULL DEFAULT '',
        scope         TEXT NOT NULL DEFAULT 'global',
        meta_json     TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_msg_created ON messages(created_at DESC)",
    # ---------------------------------------------------------------- memory items
    """
    CREATE TABLE IF NOT EXISTS memory_items (
        id             TEXT PRIMARY KEY,
        kind           TEXT NOT NULL,
        content        TEXT NOT NULL,
        subject        TEXT,
        predicate      TEXT,
        object         TEXT,
        importance     REAL NOT NULL DEFAULT 0.5,
        confidence     REAL NOT NULL DEFAULT 0.7,
        valid_from     REAL,
        valid_to       REAL,
        created_at     REAL NOT NULL,
        updated_at     REAL NOT NULL,
        last_access_at REAL NOT NULL,
        access_count   INTEGER NOT NULL DEFAULT 0,
        supersedes     TEXT,
        superseded_by  TEXT,
        session_id     TEXT,
        persona_id     TEXT,
        scope          TEXT NOT NULL DEFAULT 'global',
        tags_json      TEXT NOT NULL DEFAULT '[]',
        sources_json   TEXT NOT NULL DEFAULT '[]',
        meta_json      TEXT NOT NULL DEFAULT '{}',
        content_hash   TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_mem_scope_kind ON memory_items(scope, kind, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_mem_subject ON memory_items(subject)",
    "CREATE INDEX IF NOT EXISTS idx_mem_session ON memory_items(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_mem_hash ON memory_items(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_mem_active ON memory_items(superseded_by, valid_to)",
    # ---------------------------------------------------------------- entities / graph
    """
    CREATE TABLE IF NOT EXISTS entities (
        id            TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        type          TEXT NOT NULL DEFAULT 'concept',
        summary       TEXT NOT NULL DEFAULT '',
        importance    REAL NOT NULL DEFAULT 0.5,
        first_seen_at REAL NOT NULL,
        last_seen_at  REAL NOT NULL,
        mention_count INTEGER NOT NULL DEFAULT 1,
        scope         TEXT NOT NULL DEFAULT 'global',
        meta_json     TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_scope_name ON entities(scope, name)",
    """
    CREATE TABLE IF NOT EXISTS entity_aliases (
        entity_id TEXT NOT NULL,
        alias     TEXT NOT NULL,
        weight    REAL NOT NULL DEFAULT 1.0,
        PRIMARY KEY(entity_id, alias),
        FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_alias_alias ON entity_aliases(alias)",
    """
    CREATE TABLE IF NOT EXISTS relations (
        id           TEXT PRIMARY KEY,
        subject_id   TEXT NOT NULL,
        predicate    TEXT NOT NULL,
        object_id    TEXT NOT NULL,
        confidence   REAL NOT NULL DEFAULT 0.7,
        valid_from   REAL,
        valid_to     REAL,
        created_at   REAL NOT NULL,
        updated_at   REAL NOT NULL,
        source_memory_id TEXT,
        scope        TEXT NOT NULL DEFAULT 'global',
        meta_json    TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_triple ON relations(scope, subject_id, predicate, object_id)",
    "CREATE INDEX IF NOT EXISTS idx_rel_subject ON relations(subject_id)",
    "CREATE INDEX IF NOT EXISTS idx_rel_object ON relations(object_id)",
    # ---------------------------------------------------------------- profile
    """
    CREATE TABLE IF NOT EXISTS profiles (
        id             TEXT PRIMARY KEY,
        scope          TEXT NOT NULL DEFAULT 'global',
        subject        TEXT NOT NULL DEFAULT 'user',
        key            TEXT NOT NULL,
        value          TEXT NOT NULL,
        confidence     REAL NOT NULL DEFAULT 0.7,
        importance     REAL NOT NULL DEFAULT 0.6,
        created_at     REAL NOT NULL,
        updated_at     REAL NOT NULL,
        source_memory_id TEXT
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_profile_key ON profiles(scope, subject, key)",
    # ---------------------------------------------------------------- vectors
    # Deliberately NO foreign key to memory_items/entities. A vector is a derived
    # index entry, and it must be writable even when the parent row is not visible
    # to this connection (e.g. written by another thread's transaction), and
    # writable *before* the parent exists when a caller only wants to probe the
    # index. Consistency is maintained by the writer and repaired by
    # consolidation, which deletes orphaned vectors.
    """
    CREATE TABLE IF NOT EXISTS memory_vectors (
        memory_id  TEXT PRIMARY KEY,
        dim        INTEGER NOT NULL,
        vec        BLOB NOT NULL,
        model      TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entity_vectors (
        entity_id  TEXT PRIMARY KEY,
        dim        INTEGER NOT NULL,
        vec        BLOB NOT NULL,
        model      TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL
    )
    """,
    # ---------------------------------------------------------------- bookkeeping
    """
    CREATE TABLE IF NOT EXISTS turn_log (
        id            TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        message_id    TEXT NOT NULL,
        assistant_id  TEXT,
        processed_at  REAL NOT NULL,
        items_created INTEGER NOT NULL DEFAULT 0,
        status        TEXT NOT NULL DEFAULT 'done',
        error         TEXT NOT NULL DEFAULT ''
    )
    """,
    # The unique index on message_id is what makes the extraction upsert
    # (ON CONFLICT(message_id)) legal and the per-message idempotency guarantee real.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_turn_message ON turn_log(message_id)",
    "CREATE INDEX IF NOT EXISTS idx_turn_conv ON turn_log(conversation_id, processed_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS kv_store (
        key        TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
)

#: FTS5 is created separately because it needs its own failure handling: some
#: SQLite builds (rare, but they exist in embedded Python distributions) lack it.
_FTS_DDL: Tuple[str, ...] = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        content,
        message_id UNINDEXED,
        conversation_id UNINDEXED,
        role UNINDEXED,
        created_at UNINDEXED,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        content,
        memory_id UNINDEXED,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
)


class Database:
    """A thread-confined SQLite handle wrapped in async-friendly helpers.

    SQLite serializes writes anyway; what matters is that we never block the event
    loop for long. Reads run inline (they are sub-millisecond on this data size),
    writes go through ``asyncio.to_thread`` so a burst of memory inserts cannot
    stall a streaming chat response.
    """

    def __init__(self, path: Path, *, timeout_s: float = 15.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self._local = threading.local()
        self._write_lock = threading.Lock()
        # Every connection handed out, across threads. Thread-local storage alone cannot
        # be enumerated, and shutdown happens on a *different* thread than the one that
        # served requests (``aclose`` -> ``asyncio.to_thread``), so closing "this
        # thread's" connection closed nothing and left the file locked.
        self._live: set[sqlite3.Connection] = set()
        self._live_lock = threading.Lock()
        self._fts_enabled = False
        self._initialized = False

    # -- connection --------------------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        """One connection per thread; ``sqlite3`` objects are not shareable."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            with self._live_lock:
                if conn in self._live:
                    return conn
            # Closed underneath us by another thread (shutdown): drop the stale handle
            # instead of handing out a connection that raises "Cannot operate on a
            # closed database" on first use.
        conn = sqlite3.connect(
            str(self.path),
            timeout=self.timeout_s,
            isolation_level=None,  # autocommit; we manage transactions explicitly
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        self._configure(conn)
        self._local.conn = conn
        with self._live_lock:
            self._live.add(conn)
        return conn

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        # WAL: readers never block the writer, which keeps chat streaming smooth
        # while the background extractor writes memories.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA cache_size=-32000")  # ~32 MB page cache
        conn.execute("PRAGMA temp_store=MEMORY")

    # -- lifecycle ---------------------------------------------------------------------
    def init(self) -> None:
        if self._initialized:
            return
        conn = self.conn
        for statement in _BASE_DDL:
            conn.execute(statement)
        self._fts_enabled = self._try_create_fts(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        current = self._get_schema_version(conn)
        if current == 0:
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        elif current < SCHEMA_VERSION:
            self._migrate(conn, current)
        self._initialized = True
        log.info(
            "memory database ready",
            extra={"path": str(self.path), "fts": self._fts_enabled, "schema": SCHEMA_VERSION},
        )

    def _try_create_fts(self, conn: sqlite3.Connection) -> bool:
        try:
            for statement in _FTS_DDL:
                conn.execute(statement)
            return True
        except sqlite3.OperationalError as exc:
            log.warning(
                "FTS5 unavailable, falling back to LIKE search (chat history search will be slower)",
                extra={"error": str(exc)},
            )
            return False

    @staticmethod
    def _get_schema_version(conn: sqlite3.Connection) -> int:
        try:
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()
            return int(row["value"]) if row else 0
        except sqlite3.OperationalError:
            return 0

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Apply ordered migration steps. Currently a no-op hook for v1."""
        log.info("migrating memory database", extra={"from": from_version, "to": SCHEMA_VERSION})
        # Future: `if from_version < 2: conn.execute(...)`.
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    def close(self) -> None:
        """Close every connection this database opened, from any thread.

        Called via :meth:`aclose` on shutdown, which runs in an executor thread. Closing
        only the *calling* thread's connection (the previous behaviour) was a silent
        no-op: the request-serving thread's handle stayed open, so on Windows
        ``data/memory.sqlite3`` remained locked after the service stopped, the WAL was
        never checkpointed, and a restart could not clean up the directory.
        """
        with self._live_lock:
            conns = list(self._live)
            self._live.clear()
        for conn in conns:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - closing must never raise on shutdown
                pass
        # Only clears the calling thread's slot; other threads notice their handle is
        # gone on the next ``conn`` access and transparently reconnect.
        self._local.conn = None
        self._initialized = False

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    @property
    def fts_enabled(self) -> bool:
        return self._fts_enabled

    # -- convenience -------------------------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        row = self.query_one(sql, params)
        if row is None:
            return default
        value = row[0]
        return default if value is None else value

    def write(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run ``fn`` inside an IMMEDIATE transaction under a process-wide lock."""
        with self._write_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(conn)
                conn.execute("COMMIT")
                return result
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:  # pragma: no cover
                    pass
                raise

    async def awrite(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        return await asyncio.to_thread(self.write, fn)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        self.conn.executemany(sql, rows)


def transaction(db: Database) -> Any:
    """Context manager for a write transaction (used by scripts and tests)."""

    @contextmanager
    def _tx():
        with db._write_lock:
            conn = db.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:  # pragma: no cover
                    pass
                raise

    return _tx()


__all__ = ["Database", "SCHEMA_VERSION", "transaction"]
