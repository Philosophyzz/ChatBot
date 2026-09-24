"""Application composition root.

One class wires every plugin together and owns startup/shutdown ordering. Keeping
this in a single place (rather than scattered across route handlers) is what makes
the system swappable: to change the LLM runtime, the vector store or a TTS engine,
you edit configuration; to add a capability, you register a plugin and name it
here.

Degradation policy: **memory and speech failures must never prevent chatting.**
Every optional component is constructed defensively, and the resulting health
report tells the UI exactly what is running in reduced mode and why.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.chat import ChatEngine
from core.config import AppConfig, load_config
from core.logging import get_logger, setup_logging
from core.registry import (
    KIND_LLM,
    registry,
)
from core.session import SessionManager
from llm.supervisor import ModelSupervisor
from memory.db import Database
from memory.engine import MemoryEngine
from memory.store import MemoryStore

log = get_logger(__name__)


@dataclass
class ComponentStatus:
    """Why a component ended up in its current state — surfaced to the UI."""

    name: str
    ok: bool
    detail: str = ""
    error: str = ""

    def to_public(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "error": self.error}


class ChatBotApp:
    """Holds every live component for the server process."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config()
        setup_logging(self.config.log_level, self.config.paths.logs_dir)
        self.paths = self.config.paths
        self.registry = registry
        self.llm: Any = None
        self.memory: Optional[MemoryEngine] = None
        self.personas: Any = None
        self.sessions: Optional[SessionManager] = None
        self.engine: Optional[ChatEngine] = None
        self.stt: Any = None
        self.tts: Any = None
        #: Lets the web UI / pet switch the base model without editing files and restarting.
        self.models: Optional[ModelSupervisor] = None
        self.started_at: float = 0.0
        self.status: List[ComponentStatus] = []
        self._chat_semaphore: Optional[Any] = None  # asyncio.Semaphore, created on start
        self._retired_llms: List[Any] = []

    # ----------------------------------------------------------------------------------
    # Startup
    # ----------------------------------------------------------------------------------
    def _llm_port(self) -> int:
        """Port of the chat model server, read from ``llm.base_url`` (default 8080)."""
        try:
            from urllib.parse import urlparse

            return int(urlparse(str(self.config.llm.base_url)).port or 8080)
        except Exception:  # noqa: BLE001
            return 8080

    def rebind_chat_model(self, alias: str) -> None:
        """Point the live engine at a newly started server alias.

        The chat engine holds the *same* client object as ``self.llm`` (see ``start``), so
        updating the attribute here is what makes a switch take effect without a restart —
        otherwise every request after a switch would name a model the server no longer serves.
        """
        self.config.llm.model = alias
        target = self.llm
        if target is not None and hasattr(target, "model"):
            target.model = alias
        chat = getattr(self, "engine", None)
        inner = getattr(chat, "llm", None)
        if inner is not None and inner is not target and hasattr(inner, "model"):
            inner.model = alias
        log.info("chat model rebound", extra={"alias": alias})

    async def switch_model(self, tier_id: str) -> Dict[str, Any]:
        """Switch the base model (blocking work moved off the event loop) and rebind."""
        import asyncio

        from core.model_connection import managed_local
        if not managed_local(self.config.llm):
            from llm.supervisor import ModelSwitchError
            raise ModelSwitchError("档位切换只用于本项目的本地 8080 服务；当前连接请在模型调用设置中切换")

        if self.models is None:
            raise RuntimeError("模型管理器没有初始化")
        result = await asyncio.to_thread(self.models.switch, tier_id, on_ready=None)
        self.rebind_chat_model(tier_id)
        return result

    async def start(self, *, mock: Optional[bool] = None) -> "ChatBotApp":
        import asyncio

        started = time.perf_counter()
        self.paths.ensure()
        self.registry.load_plugins("plugs")
        self.status = []

        # ---- Base model supervisor ------------------------------------------------
        # Created before the LLM client so a failed switch can still report which tiers
        # exist and which model files are actually on disk.
        self.models = self._create_model_manager()

        # ---- LLM -----------------------------------------------------------------
        use_mock = self.config.llm.mock if mock is None else mock
        self.llm = self._build_llm(use_mock=use_mock)

        # ---- Personas ------------------------------------------------------------
        from persona.manager import PersonaManager

        self.personas = PersonaManager(self.config)
        loaded = self.personas.load()
        self.status.append(ComponentStatus("personas", True, f"已加载 {len(loaded)} 个人设"))

        # ---- Memory --------------------------------------------------------------
        try:
            self.memory = await MemoryEngine.create(self.config, llm=self.llm)
            self.memory.start_background()
            stats = await self.memory.store.stats()
            self.status.append(
                ComponentStatus(
                    "memory",
                    True,
                    f"向量后端={self.memory.vectors.name if self.memory.vectors else '?'}，"
                    f"嵌入={self.memory.embedder.name if self.memory.embedder else '?'}，"
                    f"重排={self.memory.reranker.name if self.memory.reranker else '?'}，"
                    f"已有记忆 {stats.get('memories', 0)} 条",
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("memory subsystem failed to initialise")
            self.memory = None
            self.status.append(ComponentStatus("memory", False, error=str(exc)[:300]))

        # ---- Speech --------------------------------------------------------------
        self.stt, stt_status = self._build_optional("stt", self._build_stt)
        self.tts, tts_status = self._build_optional("tts", self._build_tts)
        self.status.extend([stt_status, tts_status])
        if self.stt is not None and hasattr(self.stt, "start_watchdog"):
            try:
                self.stt.start_watchdog()
            except Exception:  # noqa: BLE001
                pass
        for backend in self._iter_tts_backends():
            if hasattr(backend, "start_watchdog"):
                try:
                    backend.start_watchdog()
                except Exception:  # noqa: BLE001
                    pass

        # ---- Sessions & engine ---------------------------------------------------
        store = self.memory.store if self.memory is not None else None
        if store is None:
            # Chat still works without memory: give the session manager a store of
            # its own so conversations are still persisted and listable.
            try:
                database = Database(self.config.paths.db_path)
                database.init()
                store = MemoryStore(database)
            except Exception:  # noqa: BLE001
                log.exception("sessions store unavailable; running fully in-memory")
                store = None
        self.sessions = SessionManager(store)
        self.engine = ChatEngine(
            config=self.config,
            llm=self.llm,
            personas=self.personas,
            memory=self.memory,
            sessions=self.sessions,
            stt=self.stt,
            tts=self.tts,
        )
        # One model, one GPU: serialise generations instead of thrashing VRAM.
        self._chat_semaphore = asyncio.Semaphore(1)

        self.started_at = time.time()
        log.info(
            "application started",
            extra={
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "llm": getattr(self.llm, "name", "?"),
                "mock": use_mock,
            },
        )
        return self

    async def stop(self) -> None:
        if self.memory is not None:
            try:
                await self.memory.close()
            except Exception:  # noqa: BLE001
                log.exception("error closing memory subsystem")
        for component in (self.stt, self.tts, self.llm, *self._retired_llms):
            if component is not None and hasattr(component, "aclose"):
                try:
                    await component.aclose()
                except Exception:  # noqa: BLE001
                    pass
        log.info("application stopped")

    # ----------------------------------------------------------------------------------
    # Component builders
    # ----------------------------------------------------------------------------------
    def _build_llm(self, *, use_mock: bool) -> Any:
        cfg = self.config.llm
        if use_mock or cfg.backend == "mock":
            try:
                self.llm_backend_name = "mock"
                instance = self.registry.create(KIND_LLM, "mock")
                self.status.append(ComponentStatus("llm", True, "内置模拟模型（无需 GPU，用于自检/演示）"))
                return instance
            except Exception as exc:  # noqa: BLE001
                self.status.append(ComponentStatus("llm", False, error=f"模拟模型不可用：{exc}"))
                raise

        backend = cfg.backend if self.registry.has(KIND_LLM, cfg.backend) else "llamacpp"
        # NOTE: do NOT put "name" in kwargs. ``Registry.create(kind, name, ...)`` already
        # receives the plugin name positionally, so adding it again raised
        # "got multiple values for argument 'name'" and prevented the server from
        # starting at all. ``OpenAICompatBackend`` also accepts ``name`` for its own
        # ``.name`` attribute, but passing it is redundant: the registry argument is
        # what selects the plugin.
        kwargs: Dict[str, Any] = {
            "base_url": cfg.base_url,
            "model": cfg.model,
            "api_key": cfg.api_key,
            "timeout_s": cfg.request_timeout_s,
            "max_retries": cfg.max_retries,
        }
        extra_llm = (self.config.extra or {}).get("llm") or {}
        if isinstance(extra_llm, dict):
            # These are backend constructor parameters (connect timeout, extra request
            # body), not registry arguments, so they are safe to forward.
            for key in ("connect_timeout_s", "extra_body"):
                if key in extra_llm:
                    kwargs[key] = extra_llm[key]
        instance = self.registry.create(KIND_LLM, backend, **kwargs)
        instance.name = backend
        self.status.append(
            ComponentStatus("llm", True, f"{backend} @ {cfg.base_url}（模型 {cfg.model}）")
        )
        return instance

    def set_model_connection(self, values: Dict[str, Any]) -> Dict[str, Any]:
        from core.model_connection import validate_connection
        from llm.openai_compat import OpenAICompatBackend

        values = dict(values)
        if str(values.get("base_url", "")).rstrip("/") == self.config.llm.base_url.rstrip("/"):
            values.setdefault("backend", self.config.llm.backend)
        connection = validate_connection(values)
        if self.engine and self.engine._active_turns:
            from core.errors import BadRequest
            raise BadRequest("请等当前回复结束后再切换模型调用模式")
        if self.models and getattr(self.models, "_progress", {}).get("state") == "switching":
            from core.errors import BadRequest
            raise BadRequest("请等本地模型加载完成后再切换")
        new = OpenAICompatBackend(base_url=connection["base_url"], model=connection["model"],
                                  api_key=connection["api_key"], name=connection["backend"],
                                  timeout_s=self.config.llm.request_timeout_s)
        # In-flight memory jobs may still be using the old client. Close it on shutdown.
        if self.llm:
            self._retired_llms.append(self.llm)
        self.llm = new
        for key, value in connection.items():
            setattr(self.config.llm, key, value)
        self.models = self._create_model_manager()
        if self.engine:
            self.engine.llm = new
        if self.memory:
            self.memory.llm = self.memory.extractor.llm = self.memory.consolidator.llm = new
        return connection

    def _create_model_manager(self):
        if self.config.llm.backend == "vllm":
            from llm.vllm_supervisor import VllmSupervisor
            return VllmSupervisor(self.paths.root, self.config, port=self._llm_port())
        return ModelSupervisor(self.paths.root, self.config, port=self._llm_port())

    def _build_stt(self) -> Any:
        from speech import build_stt

        return build_stt(self.config)

    def _build_tts(self) -> Any:
        from speech import build_tts

        return build_tts(self.config)

    def _build_optional(self, name: str, factory: Any) -> tuple:
        try:
            instance = factory()
            return instance, ComponentStatus(name, True, _describe_optional(name, instance))
        except Exception as exc:  # noqa: BLE001
            log.warning("%s unavailable", name, extra={"error": str(exc)[:300]})
            return None, ComponentStatus(name, False, error=str(exc)[:300])

    def _iter_tts_backends(self) -> List[Any]:
        backends: List[Any] = []
        router_backends = getattr(self.tts, "backends", None)
        if isinstance(router_backends, dict):
            backends.extend(router_backends.values())
        elif self.tts is not None:
            backends.append(self.tts)
        return backends

    def chat_lock(self) -> Any:
        """The semaphore that serialises generations on the single GPU.

        Exposed as a method (rather than letting routers reach for
        ``app._chat_semaphore``) so the transport layer never depends on a private
        attribute of the composition root.
        """
        return self._chat_semaphore

    # ----------------------------------------------------------------------------------
    # Health
    # ----------------------------------------------------------------------------------
    async def health(self, *, deep: bool = False) -> Dict[str, Any]:
        engine_health: Dict[str, Any] = {}
        if self.engine is not None:
            engine_health = await self.engine.health()
        payload: Dict[str, Any] = {
            "ok": True,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            "version": _VERSION,
            "root": str(self.paths.root),
            "llm": engine_health.get("llm") or {},
            "memory": engine_health.get("memory") or {"ok": False, "error": "记忆未启用"},
            "stt": engine_health.get("stt") or {"ok": False, "error": "语音识别未启用"},
            "tts": engine_health.get("tts") or {"ok": False, "error": "语音合成未启用"},
            "vram_guard": _vram_guard_snapshot(),
            "components": [component.to_public() for component in self.status],
            "registry": self.registry.snapshot(),
        }
        if deep:
            payload["ffmpeg"] = _check_ffmpeg()
            payload["config"] = self.config.to_public()
        return payload


_VERSION = "0.1.0"


def _describe_optional(name: str, instance: Any) -> str:
    if name == "stt":
        model = getattr(instance, "model_name", "")
        return f"{getattr(instance, 'name', '?')}（模型 {model}，按需加载）"
    if name == "tts":
        preference = getattr(instance, "preference", [])
        return f"{getattr(instance, 'name', '?')}（优先级 {'>'.join(preference)}）"
    return str(getattr(instance, "name", name))


def _check_ffmpeg() -> Dict[str, Any]:
    from speech.audio import ffmpeg_path, has_ffmpeg

    return {"ok": has_ffmpeg(), "path": ffmpeg_path(), "hint": "winget install Gyan.FFmpeg"}


def _vram_guard_snapshot() -> Dict[str, Any]:
    """Which GPU models the guard knows about and how much VRAM is free right now.

    Surfaced in /api/health because the failure it prevents is invisible from the outside:
    when whisper + IndexTTS2 + the chat model oversubscribe the card, the process is killed by
    native code (Windows: ucrtbase.dll, 0xC0000409) and all the user sees is the browser's
    "Failed to fetch". This makes the state checkable *before* that happens.
    """
    try:
        from core.vram import coordinator

        return coordinator.describe()
    except Exception as exc:  # noqa: BLE001 - health must never fail because of a diagnostic
        return {"enabled": False, "error": str(exc)}


def create_app(config: Optional[AppConfig] = None) -> ChatBotApp:
    """Convenience for scripts: build the app object (not yet started)."""
    return ChatBotApp(config or load_config())


__all__ = ["ChatBotApp", "ComponentStatus", "create_app"]
