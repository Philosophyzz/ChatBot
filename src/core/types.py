"""Core domain types and the extension contracts (Protocols) of the whole system.

Every capability in this project is expressed as a small ``typing.Protocol`` so a
new implementation only has to satisfy a structural interface and register itself.
No implementation module is imported here on purpose: this file is the stable
contract surface that plugins, the engine and the memory subsystem all depend on.

Design rules
------------
* Protocols are ``runtime_checkable`` so the plugin registry can sanity-check a
  candidate before it is mounted.
* Data classes are frozen where they cross a boundary (persisted, cached or sent
  over the wire) to make accidental mutation a loud error instead of a bug.
* Async is used for anything that can touch a model, disk or the network.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

# --------------------------------------------------------------------------------------
# Identifiers and small helpers
# --------------------------------------------------------------------------------------


def new_id(prefix: str = "") -> str:
    """Return a short, sortable-ish unique id, optionally namespaced."""
    raw = uuid.uuid4().hex[:16]
    return f"{prefix}{raw}" if prefix else raw


def now_ts() -> float:
    """Unix timestamp in seconds (UTC). Single source of truth for time."""
    return time.time()


# --------------------------------------------------------------------------------------
# Conversation primitives
# --------------------------------------------------------------------------------------


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """One chat turn. ``meta`` carries channel info, latency, model name, memory refs."""

    role: Role
    content: str
    id: str = field(default_factory=lambda: new_id("msg_"))
    created_at: float = field(default_factory=now_ts)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> Dict[str, Any]:
        """Render for any OpenAI-compatible /chat/completions payload."""
        return {"role": self.role.value, "content": self.content}

    def to_public(self) -> Dict[str, Any]:
        """Render for the browser/API surface."""
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "created_at": self.created_at,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class TokenEvent:
    """A streaming delta emitted by an LLM backend.

    ``kind`` is deliberately open: "text", "reasoning", "tool", "usage", "done".
    Consumers must ignore kinds they do not understand so a backend can add new
    event types without breaking the engine.
    """

    kind: str
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMStats:
    """Token accounting for one completion (filled by the backend when available)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    model: str = ""
    #: Raw server-side speed counters (llama.cpp ``timings``), kept verbatim so
    #: the /diagnostics endpoint can show tokens/s without re-deriving it.
    timings: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> Optional[float]:
        rate = self.timings.get("predicted_per_second") if self.timings else None
        if isinstance(rate, (int, float)) and rate > 0:
            return float(rate)
        if self.completion_tokens and self.total_ms and self.ttft_ms is not None:
            gen_ms = max(1.0, self.total_ms - self.ttft_ms)
            return round(self.completion_tokens / (gen_ms / 1000.0), 2)
        return None

    def to_public(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "model": self.model,
            "tokens_per_second": self.tokens_per_second,
            "timings": dict(self.timings),
        }


@dataclass(frozen=True)
class LLMRequest:
    """A completion request, independent of any vendor SDK shape."""

    messages: Sequence[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    stop: Sequence[str] = ()
    #: When set, the backend must constrain output to this JSON Schema if it can.
    json_schema: Optional[Dict[str, Any]] = None
    #: Free-form backend hints (e.g. {"cache_prompt": True, "n_keep": 64}).
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Completion:
    """A fully materialized (non-streaming) completion."""

    text: str
    message: Message
    stats: LLMStats = field(default_factory=LLMStats)


# --------------------------------------------------------------------------------------
# Memory primitives
# --------------------------------------------------------------------------------------


class MemoryKind(str, Enum):
    """The layer a memory item belongs to.

    This mirrors the layered architecture: a raw utterance, a distilled fact, a
    stable profile attribute, and a graph edge are stored side by side but are
    retrieved with different weights and lifetimes.
    """

    EPISODE = "episode"      # what happened / was said, verbatim-ish
    FACT = "fact"            # a distilled, checkable statement about the world
    PROFILE = "profile"      # a durable attribute of a person/entity
    PREFERENCE = "preference"
    RELATION = "relation"    # a (subject, predicate, object) edge
    SUMMARY = "summary"      # a rolled-up digest of older episodes
    REFLECTION = "reflection"  # a higher-order insight produced during consolidation


@dataclass
class MemoryItem:
    """One retrievable unit of long-term memory."""

    id: str
    content: str
    kind: MemoryKind
    #: 0..1 how important this is for the persona's future behaviour.
    importance: float = 0.5
    #: 0..1 model confidence in the extraction.
    confidence: float = 0.7
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    #: Validity window — the core of the "temporal" part of the knowledge graph.
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)
    last_access_at: float = field(default_factory=now_ts)
    access_count: int = 0
    source_message_ids: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    persona_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    #: Id of the memory this one supersedes (conflict resolution chain).
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    #: Retrieval bookkeeping, not persisted.
    score: float = 0.0
    score_parts: Dict[str, float] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.superseded_by is None

    def to_public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind.value if isinstance(self.kind, MemoryKind) else self.kind,
            "importance": round(self.importance, 4),
            "confidence": round(self.confidence, 4),
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "session_id": self.session_id,
            "persona_id": self.persona_id,
            "tags": list(self.tags),
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "score": round(self.score, 4),
            "score_parts": {k: round(v, 4) for k, v in self.score_parts.items()},
        }


@dataclass(frozen=True)
class Entity:
    """A node in the temporal knowledge graph."""

    id: str
    name: str
    type: str = "concept"
    aliases: Tuple[str, ...] = ()
    summary: str = ""
    importance: float = 0.5
    first_seen_at: float = field(default_factory=now_ts)
    last_seen_at: float = field(default_factory=now_ts)
    mention_count: int = 1
    #: Namespacing for multi-user / multi-persona isolation.
    scope: str = "global"

    def to_public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "aliases": list(self.aliases),
            "summary": self.summary,
            "importance": round(self.importance, 4),
            "mention_count": self.mention_count,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "scope": self.scope,
        }


@dataclass
class MemoryHit:
    """A retrieval candidate with a full score breakdown (explainability)."""

    item: MemoryItem
    score: float
    parts: Dict[str, float] = field(default_factory=dict)
    channels: List[str] = field(default_factory=list)

    def to_public(self) -> Dict[str, Any]:
        body = self.item.to_public()
        body["score"] = round(self.score, 4)
        body["score_parts"] = {k: round(v, 4) for k, v in self.parts.items()}
        body["channels"] = list(self.channels)
        return body


@dataclass(frozen=True)
class RetrievalQuery:
    """Everything the retriever needs; explicitly carries its own budget."""

    text: str
    session_id: str
    persona_id: Optional[str] = None
    scope: str = "global"
    top_k: int = 12
    #: Token budget for the assembled memory block injected into the prompt.
    token_budget: int = 1200
    #: Restrict to certain layers (empty = all).
    kinds: Tuple[MemoryKind, ...] = ()
    #: Include verbatim episode hits (needed for "search my chat history").
    include_episodes: bool = True
    #: Retrieval "as of" time, enabling point-in-time graph queries.
    as_of: Optional[float] = None


@dataclass
class RetrievalResult:
    hits: List[MemoryHit] = field(default_factory=list)
    profile_lines: List[str] = field(default_factory=list)
    rendered: str = ""
    tokens_used: int = 0
    debug: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Speech primitives
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioClip:
    """Raw audio plus its format, so backends can convert only when needed."""

    data: bytes
    mime: str = "audio/wav"
    sample_rate: int = 16000
    channels: int = 1
    duration_s: Optional[float] = None

    def to_public(self) -> Dict[str, Any]:
        return {
            "mime": self.mime,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_s": self.duration_s,
            "bytes": len(self.data),
        }


@dataclass(frozen=True)
class Transcript:
    text: str
    language: Optional[str] = None
    confidence: float = 1.0
    segments: Tuple[Dict[str, Any], ...] = ()
    duration_s: Optional[float] = None

    def to_public(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "confidence": round(self.confidence, 4),
            "duration_s": self.duration_s,
            "segments": list(self.segments),
        }


@dataclass(frozen=True)
class VoiceSpec:
    """How one persona should sound.

    ``backend`` pins a specific TTS implementation; leaving it None lets the
    router pick by capability (local-first, online fallback).
    """

    backend: Optional[str] = None
    voice_id: str = "default"
    #: Reference audio for zero-shot cloning, resolved relative to the project.
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    speed: float = 1.0
    #: Expressive hint: "sweet", "calm", "cheerful", "serious", ...
    emotion: str = "neutral"
    #: 0..1 emotion intensity for backends that support it.
    emotion_alpha: float = 0.8
    pitch_shift: float = 0.0
    style: Dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "voice_id": self.voice_id,
            "reference_audio": self.reference_audio,
            "speed": self.speed,
            "emotion": self.emotion,
            "emotion_alpha": self.emotion_alpha,
            "pitch_shift": self.pitch_shift,
            "style": dict(self.style),
        }


@dataclass(frozen=True)
class SpeechChunk:
    """A synthesized audio segment, streamable for low-latency playback."""

    data: bytes
    mime: str = "audio/wav"
    index: int = 0
    is_final: bool = False
    text: str = ""

    def to_public(self) -> Dict[str, Any]:
        return {"mime": self.mime, "index": self.index, "is_final": self.is_final, "text": self.text}


# --------------------------------------------------------------------------------------
# Persona
# --------------------------------------------------------------------------------------


@dataclass
class Persona:
    """A switchable character: prompt, voice, memory scope and sampling all move together."""

    id: str
    name: str
    system_prompt: str
    description: str = ""
    voice: VoiceSpec = field(default_factory=VoiceSpec)
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    #: Which memory scope this persona reads/writes.
    memory_scope: str = "global"
    #: Extra behaviour switches consumed by the engine (e.g. verbose_memory).
    options: Dict[str, Any] = field(default_factory=dict)
    #: Opening line offered by the UI.
    greeting: str = ""
    avatar: str = "🙂"
    builtin: bool = True

    def to_public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "voice": self.voice.to_public(),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "memory_scope": self.memory_scope,
            "options": dict(self.options),
            "greeting": self.greeting,
            "avatar": self.avatar,
            "builtin": self.builtin,
        }


# --------------------------------------------------------------------------------------
# Extension contracts
# --------------------------------------------------------------------------------------


@runtime_checkable
class LLMBackend(Protocol):
    """Any chat-completion source: llama.cpp, Ollama, vLLM, a hosted API, a mock."""

    name: str

    async def stream(self, request: LLMRequest) -> AsyncIterator[TokenEvent]:  # pragma: no cover
        ...

    async def complete(self, request: LLMRequest) -> Completion:  # pragma: no cover
        ...

    async def health(self) -> Dict[str, Any]:  # pragma: no cover
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Must be deterministic for a given input."""

    name: str
    dim: int

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover
        ...

    async def embed_one(self, text: str) -> List[float]:  # pragma: no cover
        ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder style rescoring: higher means more relevant."""

    name: str

    async def rerank(  # pragma: no cover
        self, query: str, documents: Sequence[str], top_k: int = 0
    ) -> List[Tuple[int, float]]:
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Pluggable vector index so SQLite can be swapped for pgvector/FAISS/etc."""

    name: str

    async def upsert(  # pragma: no cover
        self, ids: Sequence[str], vectors: Sequence[Sequence[float]], metas: Sequence[Dict[str, Any]]
    ) -> None:
        ...

    async def search(  # pragma: no cover
        self, vector: Sequence[float], top_k: int, flt: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        ...

    async def delete(self, ids: Sequence[str]) -> None:  # pragma: no cover
        ...

    async def count(self) -> int:  # pragma: no cover
        ...


@runtime_checkable
class STTBackend(Protocol):
    name: str

    async def transcribe(  # pragma: no cover
        self, audio: AudioClip, language: Optional[str] = None, prompt: Optional[str] = None
    ) -> Transcript:
        ...

    async def health(self) -> Dict[str, Any]:  # pragma: no cover
        ...


@runtime_checkable
class TTSBackend(Protocol):
    name: str
    #: True when this backend can synthesize expressive/cloned voices.
    supports_cloning: bool
    #: False when the backend requires the network (used by the local-first router).
    offline: bool

    async def synthesize(  # pragma: no cover
        self, text: str, voice: VoiceSpec, *, stream: bool = False
    ) -> AsyncIterator[SpeechChunk]:
        ...

    async def health(self) -> Dict[str, Any]:  # pragma: no cover
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Persistence seam for the layered memory system."""

    name: str

    async def init(self) -> None: ...  # pragma: no cover

    async def close(self) -> None: ...  # pragma: no cover


@runtime_checkable
class Channel(Protocol):
    """An input/output surface (web socket, REST, CLI, future desktop shell)."""

    name: str

    async def send(self, session_id: str, event: Dict[str, Any]) -> None: ...  # pragma: no cover


# Registry key -> factory type. The registry is intentionally untyped at the
# value level: plugins register callables, and the engine resolves them by name.
PluginFactory = Callable[..., Any]
