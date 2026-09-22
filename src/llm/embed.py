"""Embedders and rerankers — the retrieval-quality end of the memory system.

Three embedders ship, in order of quality:

* ``llamacpp`` — calls the local ``llama-server`` ``/v1/embeddings`` endpoint.
  This is the default because the server is already running for chat, supports
  ``--embedding --pooling last``, and needs no PyTorch in the main venv. It hosts
  ``Qwen3-Embedding-0.6B`` (1024-d, strong on Chinese).
* ``openai`` — any remote OpenAI-compatible embedding API.
* ``hash`` — a deterministic bag-of-features projection. Zero dependencies, used
  as an automatic fallback so *memory keeps working* (degraded) when the
  embedding server is down, and as the unit-test embedder.

The reranker prefers llama.cpp's native ``--reranking`` support (``/v1/rerank``
with ``pooling=rank``); a ``heuristic`` fallback blends lexical overlap with
entity/subject agreement so the pipeline never hard-depends on a reranker model.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.errors import BackendUnavailable
from core.logging import get_logger
from core.registry import KIND_EMBEDDER, KIND_RERANKER, register
from core.types import Embedder, Reranker
from core.utils import LazySemaphore, retry_async

log = get_logger(__name__)

try:
    import httpx

    _HAS_HTTPX = True
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


def _tokens(text: str) -> List[str]:
    """Character-level for CJK, word-level for latin. Good enough for lexical scoring."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def lexical_overlap(query: str, document: str) -> float:
    """Jaccard-ish overlap with bigram support for Chinese, in [0, 1]."""
    q = _tokens(query)
    d = _tokens(document)
    if not q or not d:
        return 0.0
    q_set, d_set = set(q), set(d)
    unigram = len(q_set & d_set) / max(1, len(q_set))
    q_bi = {a + b for a, b in zip(q, q[1:])}
    d_bi = {a + b for a, b in zip(d, d[1:])}
    bigram = len(q_bi & d_bi) / max(1, len(q_bi)) if q_bi else 0.0
    return 0.65 * unigram + 0.35 * bigram


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


# --------------------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------------------


@register(KIND_EMBEDDER, "llamacpp")
@register(KIND_EMBEDDER, "openai")
class HttpEmbedder:
    """Embeddings from an OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081/v1",
        model: str = "Qwen3-Embedding-0.6B",
        api_key: str = "sk-local",
        dim: int = 1024,
        *,
        batch_size: int = 16,
        concurrency: int = 2,
        timeout_s: float = 120.0,
        name: str = "llamacpp",
        query_prefix: str = "Instruct: Given a user message, retrieve relevant long-term memories\nQuery: ",
        document_prefix: str = "",
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "sk-local"
        self.dim = int(dim)
        self.batch_size = max(1, int(batch_size))
        self.timeout_s = float(timeout_s)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._client: Optional[Any] = None
        self._lock = asyncio.Lock()
        self._sem = LazySemaphore(concurrency)

    async def _get_client(self) -> Any:
        if not _HAS_HTTPX:
            raise BackendUnavailable("缺少 httpx 依赖，无法调用嵌入服务")
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        base_url=self.base_url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=httpx.Timeout(self.timeout_s, connect=6.0),
# trust_env=False is REQUIRED, not cosmetic: httpx reads the system proxy
                        # (e.g. Clash on 127.0.0.1:7890) and would route requests to
                        # *local* services through it. The proxy cannot reach a
                        # localhost port and answers 502, so the app reports
                        # "model server broken" when the real problem is the proxy.
                        # `proxy=None` does NOT help - measured: it still returned 502;
                        # only trust_env=False produced a real ConnectError.
                        trust_env=False,
                    )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        client = await self._get_client()

        async def _post() -> Any:
            return await client.post(
                "/embeddings",
                json={"model": self.model, "input": list(texts), "encoding_format": "float"},
            )

        try:
            response = await retry_async(_post, attempts=3)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"嵌入服务不可用（{self.base_url}）：{exc}",
                detail={"base_url": self.base_url, "model": self.model},
            ) from exc
        if response.status_code >= 400:
            raise BackendUnavailable(
                f"嵌入服务返回 {response.status_code}",
                detail={"body": response.text[:400], "base_url": self.base_url},
            )
        body = response.json()
        data = body.get("data") or []
        # Servers may return items out of order; the index field is authoritative.
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [list(item.get("embedding") or []) for item in ordered]
        if vectors and len(vectors[0]) != self.dim:
            # Trust the server over the configured guess and remember the truth.
            self.dim = len(vectors[0])
        return vectors

    async def embed(self, texts: Sequence[str], *, is_query: bool = False) -> List[List[float]]:
        if not texts:
            return []
        prefix = self.query_prefix if is_query else self.document_prefix
        prepared = [f"{prefix}{t}" if prefix else t for t in texts]
        out: List[List[float]] = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            async with self._sem():
                out.extend(await self._embed_batch(batch))
        return out

    async def embed_one(self, text: str, *, is_query: bool = False) -> List[float]:
        vectors = await self.embed([text], is_query=is_query)
        return vectors[0] if vectors else []

    async def health(self) -> Dict[str, Any]:
        try:
            vector = await self.embed_one("健康检查", is_query=True)
            return {"ok": bool(vector), "name": self.name, "dim": self.dim, "base_url": self.base_url}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "name": self.name, "error": str(exc), "base_url": self.base_url}


@register(KIND_EMBEDDER, "hash")
class HashingEmbedder:
    """Deterministic hashing embedder: no model, no network, no GPU.

    Quality is far below a trained embedder, but it preserves *some* semantic
    signal through character/word overlap, which keeps retrieval usable when the
    embedding server is unavailable. It also makes tests reproducible without
    downloading anything.
    """

    name = "hash"

    def __init__(self, dim: int = 1024, **_ignored: Any) -> None:
        self.dim = max(64, int(dim))

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = _tokens(text)
        if not tokens:
            return vec
        for index, token in enumerate(tokens):
            for gram in (token, token + (tokens[index + 1] if index + 1 < len(tokens) else "")):
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                slot = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[slot] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    async def embed(self, texts: Sequence[str], **_ignored: Any) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    async def embed_one(self, text: str, **_ignored: Any) -> List[float]:
        return self._vector(text)

    async def health(self) -> Dict[str, Any]:
        return {"ok": True, "name": self.name, "dim": self.dim, "detail": "本地哈希嵌入（降级方案）"}


# --------------------------------------------------------------------------------------
# Rerankers
# --------------------------------------------------------------------------------------


@register(KIND_RERANKER, "llamacpp")
@register(KIND_RERANKER, "openai")
class HttpReranker:
    """Cross-encoder reranking through llama.cpp's ``/v1/rerank``."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8082/v1",
        model: str = "Qwen3-Reranker-0.6B",
        api_key: str = "sk-local",
        *,
        timeout_s: float = 60.0,
        name: str = "llamacpp",
        instruction: str = "Given a user message, judge whether the memory is relevant to answering it",
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "sk-local"
        self.timeout_s = float(timeout_s)
        self.instruction = instruction
        self._client: Optional[Any] = None

    async def _get_client(self) -> Any:
        if not _HAS_HTTPX:
            raise BackendUnavailable("缺少 httpx 依赖，无法调用重排服务")
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=httpx.Timeout(self.timeout_s, connect=6.0),
# trust_env=False is REQUIRED, not cosmetic: httpx reads the system proxy
                        # (e.g. Clash on 127.0.0.1:7890) and would route requests to
                        # *local* services through it. The proxy cannot reach a
                        # localhost port and answers 502, so the app reports
                        # "model server broken" when the real problem is the proxy.
                        # `proxy=None` does NOT help - measured: it still returned 502;
                        # only trust_env=False produced a real ConnectError.
                        trust_env=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def rerank(self, query: str, documents: Sequence[str], top_k: int = 0) -> List[Tuple[int, float]]:
        if not documents:
            return []
        client = await self._get_client()
        payload = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
            "instruction": self.instruction,
            "top_n": top_k or len(documents),
        }
        response = await client.post("/rerank", json=payload)
        if response.status_code >= 400:
            raise BackendUnavailable(
                f"重排服务返回 {response.status_code}: {response.text[:200]}",
                detail={"base_url": self.base_url},
            )
        body = response.json()
        results = body.get("results") or body.get("data") or []
        out: List[Tuple[int, float]] = []
        for item in results:
            if isinstance(item, dict):
                index = int(item.get("index", len(out)))
                score = float(item.get("relevance_score", item.get("score", 0.0)))
                out.append((index, score))
        out.sort(key=lambda pair: pair[1], reverse=True)
        return out[: top_k or len(out)]

    async def health(self) -> Dict[str, Any]:
        try:
            scores = await self.rerank("测试", ["测试文档", "无关内容"])
            return {"ok": bool(scores), "name": self.name, "base_url": self.base_url}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "name": self.name, "error": str(exc), "base_url": self.base_url}


@register(KIND_RERANKER, "heuristic")
class HeuristicReranker:
    """Zero-dependency reranker: lexical overlap plus subject/recency priors.

    Deterministic, instant, and good enough to keep precision acceptable while
    the real cross-encoder is unavailable.
    """

    name = "heuristic"

    def __init__(self, **_ignored: Any) -> None:
        pass

    async def rerank(self, query: str, documents: Sequence[str], top_k: int = 0) -> List[Tuple[int, float]]:
        scored = [(index, lexical_overlap(query, doc)) for index, doc in enumerate(documents)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: top_k or len(scored)]

    async def health(self) -> Dict[str, Any]:
        return {"ok": True, "name": self.name, "detail": "启发式重排（降级方案）"}


@register(KIND_RERANKER, "none")
class NoopReranker:
    """Keeps the original order; used when reranking is explicitly disabled."""

    name = "none"

    async def rerank(self, query: str, documents: Sequence[str], top_k: int = 0) -> List[Tuple[int, float]]:
        return [(i, 0.0) for i in range(len(documents))][: top_k or len(documents)]

    async def health(self) -> Dict[str, Any]:
        return {"ok": True, "name": self.name, "detail": "未启用重排"}


def build_embedder(config: Any, *, role: str = "primary") -> Embedder:
    """Construct the configured embedder, falling back to hashing on failure.

    The fallback is constructed lazily: a broken embedding server must not stop
    the application from starting, only degrade retrieval quality.
    """
    from core.registry import registry as _registry  # local import avoids a cycle

    memory_cfg = config.memory
    name = memory_cfg.embedder if role == "primary" else memory_cfg.fallback_embedder
    if name == "llamacpp":
        from core.config import LLMConfig  # noqa: F401  (typing only)

        embed_base = getattr(config, "embedding_base_url", None) or _derive_embedding_url(config)
        return _registry.create(
            KIND_EMBEDDER,
            "llamacpp",
            base_url=embed_base,
            model=getattr(config, "embedding_model", None) or _derive_embedding_model(config),
            dim=memory_cfg.dim,
        )
    return _registry.create(KIND_EMBEDDER, name, dim=memory_cfg.dim)


def _derive_embedding_url(config: Any) -> str:
    """Resolve the embedding endpoint.

    ``config.extra`` may carry an explicit ``embedding.base_url``; otherwise we
    assume the llama.cpp router exposes the embedding model on its own port,
    which is how ``config/llama-models.ini`` is written.
    """
    extra = getattr(config, "extra", {}) or {}
    section = extra.get("embedding") or {}
    if isinstance(section, dict) and section.get("base_url"):
        return str(section["base_url"])
    return "http://127.0.0.1:8081/v1"


def _derive_embedding_model(config: Any) -> str:
    extra = getattr(config, "extra", {}) or {}
    section = extra.get("embedding") or {}
    if isinstance(section, dict) and section.get("model"):
        return str(section["model"])
    return "Qwen3-Embedding-0.6B"


def build_reranker(config: Any) -> Reranker:
    from core.registry import registry as _registry

    name = (config.memory.reranker or "heuristic").strip()
    if name == "llamacpp":
        extra = getattr(config, "extra", {}) or {}
        section = extra.get("reranker") or {}
        base_url = str(section.get("base_url")) if isinstance(section, dict) and section.get("base_url") else "http://127.0.0.1:8082/v1"
        model = str(section.get("model")) if isinstance(section, dict) and section.get("model") else "Qwen3-Reranker-0.6B"
        return _registry.create(KIND_RERANKER, "llamacpp", base_url=base_url, model=model)
    return _registry.create(KIND_RERANKER, name)


__all__ = [
    "HeuristicReranker",
    "HashingEmbedder",
    "HttpEmbedder",
    "HttpReranker",
    "NoopReranker",
    "build_embedder",
    "build_reranker",
    "cosine",
    "lexical_overlap",
]
