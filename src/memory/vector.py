"""Vector index implementations.

The retrieval pipeline talks to the ``VectorStore`` protocol, never to a specific
engine. Two backends ship:

* :class:`SqliteVectorStore` — vectors stored as float32 blobs inside the same
  SQLite file as the memory metadata. One transaction keeps text and vector
  consistent, backups stay a single file, and there is nothing extra to run.
  Scoring uses NumPy when present (a 1024-d dot product over a few thousand rows
  is well under a millisecond) and falls back to pure Python.
* :class:`MemoryVectorStore` — a dict-backed index for tests and for very small
  datasets.

Brute-force search is the *right* choice at this scale: a personal assistant
accumulates thousands, not billions, of memories, and exact search avoids an
approximate-index recall problem that would silently drop the memory the user
cares about. The protocol is the seam for swapping in FAISS/pgvector later if the
dataset ever justifies it.
"""

from __future__ import annotations

import array
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.logging import get_logger
from core.registry import KIND_VECTOR_STORE, register

log = get_logger(__name__)

try:
    import numpy as _np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy is a declared dependency
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


def pack_vector(vector: Sequence[float]) -> bytes:
    """Serialize to float32 little-endian bytes (4 bytes per dimension)."""
    return array.array("f", [float(x) for x in vector]).tobytes()


def unpack_vector(blob: bytes) -> List[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)


@register(KIND_VECTOR_STORE, "sqlite")
class SqliteVectorStore:
    """Persistent brute-force vector index over the shared memory database."""

    name = "sqlite"

    def __init__(self, db: Any, table: str = "memory_vectors", id_column: str = "memory_id", model: str = "") -> None:
        self.db = db
        self.table = table
        self.id_column = id_column
        self.model = model

    async def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metas: Sequence[Dict[str, Any]],
    ) -> None:
        if not ids:
            return
        import time as _time

        rows = [
            (str(item_id), len(vector), pack_vector(vector), self.model, _time.time())
            for item_id, vector in zip(ids, vectors)
            if vector
        ]
        if not rows:
            return

        def _write(conn: Any) -> None:
            conn.executemany(
                f"INSERT INTO {self.table}({self.id_column}, dim, vec, model, created_at) "
                f"VALUES(?,?,?,?,?) ON CONFLICT({self.id_column}) DO UPDATE SET "
                f"dim=excluded.dim, vec=excluded.vec, model=excluded.model",
                rows,
            )

        await self.db.awrite(_write)

    async def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)

        def _write(conn: Any) -> None:
            conn.execute(f"DELETE FROM {self.table} WHERE {self.id_column} IN ({placeholders})", list(ids))

        await self.db.awrite(_write)

    async def count(self) -> int:
        return int(self.db.scalar(f"SELECT COUNT(*) FROM {self.table}", default=0) or 0)

    async def get(self, ids: Sequence[str]) -> Dict[str, List[float]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.db.query(
            f"SELECT {self.id_column} AS rid, vec FROM {self.table} WHERE {self.id_column} IN ({placeholders})",
            list(ids),
        )
        return {row["rid"]: unpack_vector(row["vec"]) for row in rows}

    async def iter_all(self) -> List[Tuple[str, bytes, int]]:
        """Return ``(id, blob, dim)`` for every row. Used by consolidation scoring."""
        rows = self.db.query(f"SELECT {self.id_column} AS rid, vec, dim FROM {self.table}")
        return [(row["rid"], row["vec"], int(row["dim"])) for row in rows]

    async def search(
        self,
        vector: Sequence[float],
        top_k: int,
        flt: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        if not vector:
            return []
        allowed: Optional[set] = None
        if flt and flt.get("ids"):
            allowed = {str(i) for i in flt["ids"]}

        rows = self.db.query(f"SELECT {self.id_column} AS rid, vec, dim FROM {self.table}")
        if not rows:
            return []

        if allowed is not None:
            rows = [row for row in rows if row["rid"] in allowed]
            if not rows:
                return []

        if _HAS_NUMPY:
            query = _np.asarray(vector, dtype=_np.float32)
            query_norm = float(_np.linalg.norm(query))
            ids: List[str] = []
            blobs: List[bytes] = []
            dims: List[int] = []
            for row in rows:
                ids.append(row["rid"])
                blobs.append(row["vec"])
                dims.append(int(row["dim"]))
            # Vectors are stored L2-normalized by the embedder, but we normalize
            # defensively so a non-normalizing backend still scores correctly.
            count = len(blobs)
            max_dim = max(dims) if dims else 0
            if max_dim == 0:
                return []
            matrix = _np.zeros((count, max_dim), dtype=_np.float32)
            for index, (blob, dim) in enumerate(zip(blobs, dims)):
                if dim != max_dim:
                    continue  # dimension mismatch: skip rather than mis-score
                matrix[index] = _np.frombuffer(blob, dtype=_np.float32, count=dim)
            norms = _np.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1.0
            if query_norm == 0 or query.shape[0] != max_dim:
                return []
            scores = (matrix @ query) / (norms * query_norm)
            k = min(max(1, top_k), count)
            top = _np.argpartition(-scores, k - 1)[:k] if k < count else _np.arange(count)
            ordered = sorted(((int(i), float(scores[i])) for i in top), key=lambda pair: pair[1], reverse=True)
            return [(ids[i], score) for i, score in ordered]

        # Pure-Python fallback.
        query_norm = math.sqrt(sum(x * x for x in vector))
        if query_norm == 0:
            return []
        scored: List[Tuple[str, float]] = []
        for row in rows:
            vec = unpack_vector(row["vec"])
            if len(vec) != len(vector):
                continue
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            dot = sum(a * b for a, b in zip(vec, vector))
            scored.append((row["rid"], dot / (norm * query_norm)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(1, top_k)]


@register(KIND_VECTOR_STORE, "memory")
class MemoryVectorStore:
    """In-process index for tests, demos and ephemeral sessions."""

    name = "memory"

    def __init__(self, **_ignored: Any) -> None:
        self._data: Dict[str, List[float]] = {}

    async def upsert(
        self, ids: Sequence[str], vectors: Sequence[Sequence[float]], metas: Sequence[Dict[str, Any]]
    ) -> None:
        for item_id, vector in zip(ids, vectors):
            if vector:
                self._data[str(item_id)] = [float(x) for x in vector]

    async def delete(self, ids: Sequence[str]) -> None:
        for item_id in ids:
            self._data.pop(str(item_id), None)

    async def count(self) -> int:
        return len(self._data)

    async def search(
        self, vector: Sequence[float], top_k: int, flt: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        if not vector:
            return []
        allowed = {str(i) for i in flt["ids"]} if flt and flt.get("ids") else None
        query_norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        scored: List[Tuple[str, float]] = []
        for item_id, vec in self._data.items():
            if allowed is not None and item_id not in allowed:
                continue
            if len(vec) != len(vector):
                continue
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            dot = sum(a * b for a, b in zip(vec, vector))
            scored.append((item_id, dot / (norm * query_norm)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(1, top_k)]

    async def get(self, ids: Sequence[str]) -> Dict[str, List[float]]:
        return {str(i): self._data[str(i)] for i in ids if str(i) in self._data}

    async def iter_all(self) -> List[Tuple[str, bytes, int]]:
        return [(key, pack_vector(value), len(value)) for key, value in self._data.items()]


__all__ = ["MemoryVectorStore", "SqliteVectorStore", "pack_vector", "unpack_vector"]
