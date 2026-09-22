"""Layered configuration.

Precedence (lowest to highest):
    1. built-in defaults declared here
    2. ``config/config.yaml`` (or ``config/config.json``)
    3. ``config/local.yaml`` — machine-specific overrides, never committed
    4. environment variables prefixed ``CHATBOT_`` using ``__`` as nesting separator
       (e.g. ``CHATBOT_LLM__TEMPERATURE=0.4``)

Everything the system needs to locate on disk lives under :class:`Paths`, whose
single root is the project directory. Model files are required to live under the
project root (D: drive), which is enforced by :meth:`Paths.assert_on_project_root`.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from core.errors import ConfigError

#: Re-exported so existing imports keep working; the canonical definition lives in
#: :mod:`core.errors` (a second same-named class here shadowed it and confused
#: ``except`` clauses).
__all__ = [
    "AppConfig",
    "ConfigError",
    "ExtractionConfig",
    "LLMConfig",
    "MemoryConfig",
    "Paths",
    "PROJECT_ROOT",
    "ServerConfig",
    "SpeechConfig",
    "load_config",
    "load_yaml_or_json",
]

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

#: The project lives on D:. Resolved from this file's location so the tree can be moved
#: as a whole, but an explicit env var always wins.
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("CHATBOT_ROOT", _DEFAULT_ROOT)).resolve()


@dataclass(frozen=True)
class Paths:
    """Every filesystem location the application is allowed to write to.

    Keeping them in one frozen object makes the "everything on D:" requirement
    auditable: a reviewer reads this class and knows the whole write surface.
    """

    root: Path = PROJECT_ROOT
    config_dir: Path = field(default=None)  # type: ignore[assignment]
    data_dir: Path = field(default=None)  # type: ignore[assignment]
    models_dir: Path = field(default=None)  # type: ignore[assignment]
    gguf_dir: Path = field(default=None)  # type: ignore[assignment]
    tts_dir: Path = field(default=None)  # type: ignore[assignment]
    voices_dir: Path = field(default=None)  # type: ignore[assignment]
    cache_dir: Path = field(default=None)  # type: ignore[assignment]
    logs_dir: Path = field(default=None)  # type: ignore[assignment]
    bin_dir: Path = field(default=None)  # type: ignore[assignment]
    venvs_dir: Path = field(default=None)  # type: ignore[assignment]
    web_dir: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        root = Path(self.root)
        derived = {
            "config_dir": root / "config",
            "data_dir": root / "data",
            "models_dir": root / "models",
            "gguf_dir": root / "models" / "gguf",
            "tts_dir": root / "models" / "tts",
            "voices_dir": root / "models" / "voices",
            "cache_dir": root / "models" / "cache",
            "logs_dir": root / "logs",
            "bin_dir": root / "bin",
            "venvs_dir": root / "venvs",
            "web_dir": root / "web",
        }
        for key, value in derived.items():
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)

    def ensure(self) -> "Paths":
        """Create every directory. Safe to call repeatedly."""
        for key, value in self.as_dict().items():
            Path(value).mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> Dict[str, Path]:
        return {
            "root": self.root,
            "config_dir": self.config_dir,
            "data_dir": self.data_dir,
            "models_dir": self.models_dir,
            "gguf_dir": self.gguf_dir,
            "tts_dir": self.tts_dir,
            "voices_dir": self.voices_dir,
            "cache_dir": self.cache_dir,
            "logs_dir": self.logs_dir,
            "bin_dir": self.bin_dir,
            "venvs_dir": self.venvs_dir,
            "web_dir": self.web_dir,
        }

    def assert_on_project_root(self, path: Path) -> Path:
        """Fail loudly if a model path escapes the project root (the D: requirement)."""
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:  # pragma: no cover - guard rail
            raise ValueError(
                f"model path {resolved} is outside the project root {self.root}; "
                "the deployment contract requires all model files on D:"
            ) from exc
        return resolved

    @property
    def db_path(self) -> Path:
        return self.data_dir / "memory.sqlite3"

    @property
    def sessions_db_path(self) -> Path:
        return self.data_dir / "sessions.sqlite3"

    @property
    def personas_file(self) -> Path:
        return self.config_dir / "personas.yaml"

    @property
    def llama_models_ini(self) -> Path:
        return self.config_dir / "llama-models.ini"


# --------------------------------------------------------------------------------------
# Config sections
# --------------------------------------------------------------------------------------


@dataclass
class LLMConfig:
    #: Which registered backend to use: "llamacpp" | "ollama" | "openai" | "mock".
    backend: str = "llamacpp"
    #: ``mock`` runs the whole app with a scripted model — no GPU, no download.
    #: Kept as an explicit flag so the setup wizard can verify the pipeline before
    #: the 15 GB model finishes downloading.
    mock: bool = False
    #: OpenAI-compatible base URL of the local server.
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "sk-local"
    model: str = "local-model"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    #: Hard cap on prompt tokens; the engine trims history to respect it.
    context_tokens: int = 16384
    #: Reserve for the answer, so history trimming never starves generation.
    reserve_tokens: int = 1024
    request_timeout_s: float = 300.0
    #: Retries for transport-level failures (never for a partial stream).
    max_retries: int = 2


@dataclass
class MemoryConfig:
    store: str = "sqlite"
    embedder: str = "llamacpp"
    #: Fallback embedder used when the primary one is unreachable.
    fallback_embedder: str = "hash"
    reranker: str = "llamacpp"
    dim: int = 1024
    #: Retrieval
    top_k: int = 14
    token_budget: int = 1400
    #: Weights of the hybrid score. They are normalized at use time.
    w_dense: float = 1.0
    w_lexical: float = 0.7
    w_graph: float = 0.6
    w_importance: float = 0.35
    w_recency: float = 0.25
    w_rerank: float = 1.2
    #: Recency half-life in days for the exponential decay term.
    recency_half_life_days: float = 21.0
    #: Extraction
    extract_async: bool = True
    extract_every_n_turns: int = 1
    extract_min_chars: int = 8
    max_items_per_turn: int = 8
    #: Conflict resolution: cosine above which two facts are considered the same claim.
    contradiction_threshold: float = 0.86
    #: Consolidation ("sleep-time compute")
    consolidate_enabled: bool = True
    consolidate_interval_s: int = 1800
    consolidate_idle_s: int = 300
    consolidate_min_episodes: int = 24
    #: Reflections: synthesize higher-order insights from clusters of facts.
    reflections_enabled: bool = True


@dataclass
class SpeechConfig:
    stt_backend: str = "faster_whisper"
    tts_backend: str = "router"
    #: Ordered preference for the TTS router; first healthy match wins.
    tts_preference: List[str] = field(default_factory=lambda: ["indextts", "edge"])
    #: Local-first policy: never let a network backend win over a healthy local one.
    prefer_offline: bool = True
    stt_model: str = "large-v3-turbo"
    stt_device: str = "cuda"
    stt_compute_type: str = "int8_float16"
    stt_language: str = "zh"
    stt_idle_unload_s: int = 900
    #: 显存守卫：加载 whisper / IndexTTS2 之前，如果空闲显存不够，就先把另一个卸掉。
    #: 这台 16GB 卡上「对话模型 + whisper + IndexTTS2」三者同时驻留会超；而超了的后果
    #: 不是 Python 异常，是进程被原生代码 fail-fast 干掉（Windows 记录 0xc0000409），
    #: 表现为网页「Failed to fetch」、桌宠「连不上本地服务」。关掉它就得自己保证显存。
    vram_guard: bool = True
    #: 守卫留出的余量（GB）：CUDA 上下文与显存碎片。
    vram_margin_gb: float = 0.8
    #: HuggingFace endpoint for the Whisper download on first use. The mirror is the
    #: default because huggingface.co is unreliable here; set to "" to use the official
    #: host. Xet is disabled alongside it (the mirror does not proxy Xet and returns 401).
    stt_endpoint: str = "https://hf-mirror.com"
    #: 额外搜索 CUDA 运行库（cublas64_12.dll 等）的目录。留空=自动在同类 conda 环境的
    #: torch\lib 里找——本机驱动是 CUDA 13，CTranslate2 需要的却是 CUDA 12 的 cuBLAS，
    #: 不挂上就会报 "Library cublas64_12.dll is not found"。
    stt_cuda_dll_dirs: List[str] = field(default_factory=list)
    #: Voice-activity trimming before transcription (saves time on silence).
    vad_filter: bool = True
    max_utterance_s: float = 60.0
    #: Target characters per TTS chunk. Speech is synthesized *and sent* chunk by
    #: chunk, so this number is the user's time-to-first-word: a whole reply in one
    #: request meant waiting for every sentence (17s measured with edge-tts on an
    #: 80-character answer). ~40 characters is 2–4 seconds of speech.
    tts_chunk_chars: int = 40
    #: How many chunks may be synthesized in parallel. Edge-tts costs 1.5–5s per
    #: request in this network, so two in flight roughly halves total time while
    #: staying far below the rate that triggers its "NoAudioReceived" throttling.
    tts_chunk_concurrency: int = 2


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8077
    #: Allow LAN access (phone as a microphone/speaker). Must be explicit.
    allow_lan: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["http://127.0.0.1:8077", "http://localhost:8077"])
    #: Max upload size for a voice clip, in bytes.
    max_audio_bytes: int = 24 * 1024 * 1024
    #: Serve the static web UI from web/ without a build step.
    serve_static: bool = True


@dataclass
class ExtractionConfig:
    """Model + budget used for the background memory extraction pass."""

    enabled: bool = True
    model: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 900
    #: How many past turns to include as context for pronoun/entity resolution.
    context_turns: int = 6
    #: Skip extraction for turns shorter than this.
    min_chars: int = 6


@dataclass
class AppConfig:
    paths: Paths = field(default_factory=Paths)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    #: Registered persona used when a session does not name one.
    default_persona: str = "sweet_companion"
    log_level: str = "INFO"
    #: Turn on to expose score breakdowns and retrieved memories in API responses.
    debug: bool = False
    #: Extra raw sections for plugins that want their own namespace.
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> Dict[str, Any]:
        """Redacted view for the settings endpoint (never leaks api_key)."""
        return {
            "llm": {
                "backend": self.llm.backend,
                "base_url": self.llm.base_url,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "context_tokens": self.llm.context_tokens,
            },
            "memory": {
                "store": self.memory.store,
                "embedder": self.memory.embedder,
                "reranker": self.memory.reranker,
                "dim": self.memory.dim,
                "top_k": self.memory.top_k,
                "token_budget": self.memory.token_budget,
                "recency_half_life_days": self.memory.recency_half_life_days,
                "consolidate_enabled": self.memory.consolidate_enabled,
            },
            "speech": {
                "stt_backend": self.speech.stt_backend,
                "tts_backend": self.speech.tts_backend,
                "tts_preference": list(self.speech.tts_preference),
                "prefer_offline": self.speech.prefer_offline,
                "stt_model": self.speech.stt_model,
                "stt_language": self.speech.stt_language,
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "allow_lan": self.server.allow_lan,
            },
            "default_persona": self.default_persona,
            "debug": self.debug,
        }


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _mini_yaml_load(text: str) -> Dict[str, Any]:
    """Tiny YAML subset parser used only when PyYAML is unavailable.

    Supports nested mappings by indentation, ``key: value`` scalars, ``- `` lists,
    comments and blank lines. It is deliberately small: the project ships a real
    parser dependency, this exists so configuration never becomes an install-time
    hard failure.

    The one subtlety is distinguishing an empty mapping from a list: ``key:``
    followed by ``- item`` lines is a list, ``key:`` followed by ``child: v`` is a
    mapping. Since the value is empty at parse time, the next non-blank line decides.
    """
    root: Dict[str, Any] = {}
    stack: List[tuple] = [(-1, root)]
    lines = text.splitlines()
    for position, raw in enumerate(lines):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                continue
            parent.append(_coerce(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            child: Any = [] if _next_line_is_list_item(lines, position, indent) else {}
            parent[key] = child
            stack.append((indent, child))
        elif value in ("|", ">"):
            parent[key] = ""
        else:
            parent[key] = _coerce(value)
    return root


def _next_line_is_list_item(lines: List[str], position: int, indent: int) -> bool:
    """Look ahead for the first meaningful line and report whether it starts a list."""
    for candidate in lines[position + 1 :]:
        stripped = candidate.split("#", 1)[0].strip()
        if not stripped:
            continue
        candidate_indent = len(candidate) - len(candidate.lstrip(" "))
        # A list item must be indented deeper than its key.
        return candidate_indent > indent and stripped.startswith("- ")
    return False


def _coerce(value: str) -> Any:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    low = value.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_coerce(p.strip()) for p in inner.split(",")]
    return value


def load_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text) or {}
    try:  # real parser when available
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return _mini_yaml_load(text)


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _env_overrides(prefix: str = "CHATBOT_") -> Dict[str, Any]:
    """``CHATBOT_LLM__TEMPERATURE=0.4`` -> ``{"llm": {"temperature": 0.4}}``."""
    result: Dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower().split("__")
        if not path or not path[0]:
            continue
        cursor = result
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):  # pragma: no cover - pathological env
                break
        else:
            cursor[path[-1]] = _coerce(raw)
    return result


def _apply_section(target: Any, data: Mapping[str, Any]) -> None:
    """Copy matching keys onto a dataclass instance, ignoring unknown keys.

    Unknown keys are collected by the caller into ``AppConfig.extra`` so a plugin
    can own its own config namespace without patching this loader.
    """
    for key, value in data.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(current, Paths) and isinstance(value, Mapping):
            continue
        if isinstance(current, list) and isinstance(value, list):
            setattr(target, key, list(value))
        elif isinstance(current, float) and isinstance(value, (int, float)):
            setattr(target, key, float(value))
        elif isinstance(current, int) and isinstance(value, (int, float)) and not isinstance(value, bool):
            setattr(target, key, int(value))
        elif isinstance(current, dict) and isinstance(value, Mapping):
            merged = dict(current)
            merged.update(value)
            setattr(target, key, merged)
        elif current is None or isinstance(value, type(current)):
            setattr(target, key, value)
        elif value is not None:
            # Best-effort coercion (e.g. Optional[str] from a numeric YAML scalar).
            try:
                setattr(target, key, type(current)(value))
            except Exception:
                setattr(target, key, value)


def load_config(root: Optional[Path] = None, overrides: Optional[Mapping[str, Any]] = None) -> AppConfig:
    """Build the effective configuration."""
    cfg = AppConfig()
    if root is not None:
        cfg.paths = Paths(root=Path(root))
    cfg.paths.ensure()

    merged: Dict[str, Any] = {}
    for candidate in (cfg.paths.config_dir / "config.yaml", cfg.paths.config_dir / "config.json"):
        merged = _deep_merge(merged, load_yaml_or_json(candidate))
    for candidate in (cfg.paths.config_dir / "local.yaml", cfg.paths.config_dir / "local.json"):
        merged = _deep_merge(merged, load_yaml_or_json(candidate))
    if overrides:
        merged = _deep_merge(merged, dict(overrides))
    merged = _deep_merge(merged, _env_overrides())

    known = {
        "llm": cfg.llm,
        "memory": cfg.memory,
        "speech": cfg.speech,
        "server": cfg.server,
        "extraction": cfg.extraction,
    }
    for section, target in known.items():
        data = merged.get(section)
        if isinstance(data, Mapping):
            _apply_section(target, data)

    # Scalars on the root object.
    if isinstance(merged.get("paths"), Mapping):
        raw_root = merged["paths"].get("root")
        if raw_root:
            cfg.paths = Paths(root=Path(str(raw_root))).ensure()
    top_scalars = {k: v for k, v in merged.items() if not isinstance(v, Mapping)}
    _apply_section(cfg, top_scalars)

    leftover = {
        k: v
        for k, v in merged.items()
        if k not in known and k not in {"paths"} and k not in top_scalars
    }
    if leftover:
        cfg.extra.update(leftover)
    return cfg


__all__ = [
    "AppConfig",
    "ConfigError",
    "ExtractionConfig",
    "LLMConfig",
    "MemoryConfig",
    "Paths",
    "PROJECT_ROOT",
    "ServerConfig",
    "SpeechConfig",
    "load_config",
    "load_yaml_or_json",
]
