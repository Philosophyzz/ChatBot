"""Speech subsystem: audio I/O, transcription and synthesis factories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from speech.audio import (
    TARGET_SAMPLE_RATE,
    decode_to_pcm,
    has_ffmpeg,
    normalize_clip,
    pcm_to_wav,
    sniff_format,
)
from speech.stt import FasterWhisperSTT, MockSTT
from speech.tts import (
    EdgeTTSBackend,
    GPTSoVITSTTSBackend,
    IndexTTSBackend,
    OpenAITTSBackend,
    SapiTTSBackend,
    ToneTTSBackend,
    TTSRouter,
)

log = get_logger(__name__)


def build_stt(config: Any, *, mock: bool = False) -> Any:
    """Construct the configured STT backend."""
    if mock:
        return MockSTT()
    speech_cfg = config.speech
    if speech_cfg.stt_backend == "mock":
        return MockSTT()
    models_root = config.paths.models_dir
    model = resolve_stt_model(config, speech_cfg.stt_model)
    # 本地目录解析成功时，同时把「仓库 id」作为兜底传下去：半成品文件加载会失败，
    # 那时自动联网重取，而不是让桌宠的语音输入一直不可用。
    fallback = speech_cfg.stt_model if model != speech_cfg.stt_model else None
    log.info("stt model resolved", extra={"configured": speech_cfg.stt_model, "using": model})
    stt = FasterWhisperSTT(
        model=model,
        device=speech_cfg.stt_device,
        compute_type=speech_cfg.stt_compute_type,
        language=speech_cfg.stt_language,
        # Cache Whisper weights under the project's models directory so nothing
        # lands on C: (a fresh download is ~1.6 GB).
        download_root=str(models_root / "whisper"),
        endpoint=getattr(speech_cfg, "stt_endpoint", "https://hf-mirror.com") or None,
        fallback_model=fallback,
        cuda_dll_dirs=list(getattr(speech_cfg, "stt_cuda_dll_dirs", None) or []),
        idle_unload_s=speech_cfg.stt_idle_unload_s,
        vad_filter=speech_cfg.vad_filter,
    )
    _register_vram_users(config, {"stt": stt})
    return stt


def resolve_stt_model(config: Any, configured: str) -> str:
    """Accept a size name, a HuggingFace repo id, or a **local directory**.

    Downloading the 1.6GB model by hand
    (``hf download ... --local-dir models\\whisper\\large-v3-turbo``) produces a plain
    directory, which faster-whisper would ignore: given a repo id it looks in its own
    cache layout instead. Resolving an existing local folder here means a manual or
    offline download just works, with no config edit.
    """
    try:
        direct = Path(configured)
        if direct.is_dir() and (direct / "model.bin").exists():
            return str(direct)
    except OSError:
        pass
    if configured and "/" not in configured and "\\" not in configured:
        local = Path(str(config.paths.models_dir)) / "whisper" / configured
        if (local / "model.bin").exists():
            return str(local)
    return configured


def _first_existing(*candidates: Any) -> Optional[str]:
    """First path that exists, as a string. Locates the optional TTS environment."""
    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = Path(str(candidate))
        except Exception:  # noqa: BLE001
            continue
        if path.exists():
            return str(path)
    return None


def _resolve_from_root(config: Any, value: Any) -> Optional[str]:
    """Resolve a configured path against the project root, not the process CWD.

    ``config.yaml`` stores ``models/tts/IndexTTS2`` — a relative path. Whether that resolves
    depends on where the service was started from: the launcher sets the working directory, so
    it works there, but a service started from anywhere else silently reports
    "模型目录里找不到 config.yaml" and the TTS router quietly falls back to ``edge`` — the user
    just hears a different voice and has no idea why. Everything the app owns is on D:, so
    anchoring to the project root is always the right reading of a relative setting.
    """
    if not value:
        return None
    try:
        path = Path(str(value))
    except Exception:  # noqa: BLE001
        return str(value)
    if path.is_absolute():
        return str(path)
    root = getattr(getattr(config, "paths", None), "root", None)
    return str((Path(str(root)) / path) if root else path)


def _register_vram_users(config: Any, backends: Dict[str, Any]) -> None:
    """Teach the VRAM coordinator which speech models are big, and how to evict them.

    Called from ``build_stt``/``build_tts``. The numbers are deliberately conservative
    (measured/estimated at load time on this machine): whisper int8 ≈ 1.8 GB, IndexTTS2 ≈ 4 GB,
    chat 9B ≈ 5.3 GB — a 16 GB card with a browser open cannot hold whisper *and* IndexTTS2
    next to the chat model, and overcommitting kills the process natively.
    """
    from core.vram import configure, coordinator  # noqa: PLC0415

    speech_cfg = getattr(config, "speech", None)
    extra = getattr(config, "extra", {}) or {}
    tts_section = extra.get("tts") if isinstance(extra.get("tts"), dict) else {}
    configure(
        enabled=bool(getattr(speech_cfg, "vram_guard", True)),
        margin_gb=float(getattr(speech_cfg, "vram_margin_gb", 0.8) or 0.8),
    )
    stt = backends.get("stt")
    if stt is not None and hasattr(stt, "unload"):
        coordinator.register(
            "stt",
            needs_gb=float(getattr(stt, "vram_gb", 1.8)),
            unload=stt.unload,
            note="faster-whisper (CTranslate2, int8)",
        )
    for name, backend in (backends.get("tts") or {}).items():
        if backend is None or not hasattr(backend, "unload"):
            continue
        needs = float(getattr(backend, "vram_gb", 0.0) or 0.0)
        if needs <= 0:
            continue  # edge/sapi/openai do not hold a GPU model
        coordinator.register(name, needs_gb=needs, unload=backend.unload, note=getattr(backend, "label", ""))
    log.info("vram coordinator configured", extra=coordinator.describe())


def build_tts(config: Any, *, mock: bool = False) -> Any:
    """Construct the TTS router with every configured backend mounted.

    All backends are constructed eagerly (they are thin wrappers that load models
    lazily themselves) so the router can report accurate health for each and fall
    back instantly instead of discovering a missing dependency mid-sentence.
    """
    if mock:
        return TTSRouter({"tone": ToneTTSBackend()}, preference=["tone"])

    speech_cfg = config.speech
    extra = getattr(config, "extra", {}) or {}
    tts_section = extra.get("tts") if isinstance(extra.get("tts"), dict) else {}
    indextts_section = (tts_section or {}).get("indextts") if isinstance((tts_section or {}).get("indextts"), dict) else {}
    edge_section = (tts_section or {}).get("edge") if isinstance((tts_section or {}).get("edge"), dict) else {}

    model_dir = _resolve_from_root(
        config,
        indextts_section.get("model_dir")
        or getattr(speech_cfg, "indextts_model_dir", None)
        or str(config.paths.tts_dir / "IndexTTS2"),
    )
    # The dedicated environment created by ``scripts/install-tts.ps1``. ``indextts``
    # pins torch 2.8 / numpy 2.2.6 / keras / opencv…; installing those here would
    # downgrade numpy under the running app, so the model runs behind a JSON worker in
    # its own environment (see tools/tts_worker.py) and this one stays clean.
    #
    # Candidates are tried in order: an explicit setting always wins, then the conda
    # environment (packages on D:, consistent with the app environment), then a plain
    # venv. Checking the filesystem here is deliberate — asking conda would mean
    # spawning a process during app startup.
    python_exe = indextts_section.get("python_exe") or _first_existing(
        config.paths.root / "venvs" / "tts" / "Scripts" / "python.exe",
        Path(str(config.paths.venvs_dir)) / "tts" / "python.exe",
        Path("D:/conda-envs/chatbot-tts/python.exe"),
        Path.home() / ".conda" / "envs" / "chatbot-tts" / "python.exe",
        Path("D:/miniconda3/envs/chatbot-tts/python.exe"),
        Path("C:/ProgramData/miniconda3/envs/chatbot-tts/python.exe"),
    )
    worker_script = indextts_section.get("worker") or str(
        config.paths.root / "tools" / "tts_worker.py"
    )
    gpt_sovits_section = (
        (tts_section or {}).get("gpt_sovits")
        if isinstance((tts_section or {}).get("gpt_sovits"), dict)
        else {}
    )
    backends: Dict[str, Any] = {
        "indextts": IndexTTSBackend(
            model_dir=str(model_dir),
            device=str(indextts_section.get("device") or "cuda:0"),
            idle_unload_s=int(indextts_section.get("idle_unload_s") or 600),
            default_reference=indextts_section.get("default_reference")
            or str(config.paths.voices_dir / "default_sweet.wav"),
            fp16=bool(indextts_section.get("fp16", True)),
            python_exe=_resolve_from_root(config, python_exe),
            worker_script=_resolve_from_root(config, worker_script),
        ),
        # Fine-tuned voices: talks to the GPT-SoVITS inference server started by
        # scripts\train-tts.ps1 -Serve (default 127.0.0.1:9090).
        "gpt_sovits": GPTSoVITSTTSBackend(
            base_url=str(gpt_sovits_section.get("base_url") or "http://127.0.0.1:9090"),
            text_lang=str(gpt_sovits_section.get("text_lang") or "zh"),
            prompt_lang=str(gpt_sovits_section.get("prompt_lang") or "zh"),
            media_type=str(gpt_sovits_section.get("media_type") or "wav"),
            split_method=str(gpt_sovits_section.get("split_method") or "cut5"),
            speed=float(gpt_sovits_section.get("speed") or 1.0),
            extra=gpt_sovits_section.get("extra") if isinstance(gpt_sovits_section.get("extra"), dict) else None,
        ),
        "edge": EdgeTTSBackend(
            default_voice=str(edge_section.get("default_voice") or "zh-CN-XiaoxiaoNeural")
        ),
        "sapi": SapiTTSBackend(),
        "openai": OpenAITTSBackend(
            base_url=str((tts_section or {}).get("openai_base_url") or "http://127.0.0.1:8090/v1"),
            model=str((tts_section or {}).get("openai_model") or "tts-1"),
        ),
    }
    preference: List[str] = [name for name in speech_cfg.tts_preference if name in backends]
    if not preference:
        preference = ["indextts", "edge", "openai", "sapi"]

    # 注意形状：_register_vram_users 期望 {"stt": ..., "tts": {name: backend}}。
    # 之前这里直接把 backends 传了进去，于是 backends.get("tts") 是 None —— IndexTTS2
    # 根本没被登记，守卫生效不了（日志里表现为 "nothing left to unload"）。
    _register_vram_users(config, {"tts": backends})

    # Explicit quality ranking from ``tts.tts_backends.<name>.priority``. Without it the
    # router would treat "offline" as the primary sort key and promote the dated Windows
    # SAPI voice above the much better edge voice.
    priorities: Dict[str, int] = {}
    raw_priorities = (tts_section or {}).get("tts_backends")
    if isinstance(raw_priorities, dict):
        for name, spec in raw_priorities.items():
            if isinstance(spec, dict) and isinstance(spec.get("priority"), int):
                priorities[str(name)] = int(spec["priority"])

    return TTSRouter(
        backends,
        preference=preference,
        prefer_offline=bool(speech_cfg.prefer_offline),
        priorities=priorities,
    )


__all__ = [
    "EdgeTTSBackend",
    "FasterWhisperSTT",
    "IndexTTSBackend",
    "MockSTT",
    "OpenAITTSBackend",
    "SapiTTSBackend",
    "TARGET_SAMPLE_RATE",
    "TTSRouter",
    "ToneTTSBackend",
    "build_stt",
    "build_tts",
    "decode_to_pcm",
    "has_ffmpeg",
    "normalize_clip",
    "pcm_to_wav",
    "sniff_format",
]
