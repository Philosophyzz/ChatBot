"""Speech-to-text backends.

``faster_whisper`` is the default: CTranslate2 inference, int8 quantization, and a
built-in VAD that trims silence before decoding. On a 16 GB card shared with a 27B
model, ``large-v3-turbo`` at ``int8_float16`` costs roughly 1.5 GB of VRAM and
transcribes a five-second utterance in well under a second.

Two engineering decisions that matter more than the model choice:

* **Lazy load + idle unload.** Whisper is only needed while the user is speaking.
  The model loads on the first request and is released after
  ``speech.stt_idle_unload_s`` of silence, returning its VRAM to the LLM. This is
  what makes "27B model + speech" co-exist on a 16 GB GPU at all.
* **Blocking work goes to a thread.** CTranslate2 is synchronous and CPU/GPU
  bound; running it inline would freeze the whole event loop (and therefore the
  chat stream) for the duration of a transcription.
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.errors import AudioError, BackendUnavailable
from core.logging import get_logger
from core.registry import KIND_STT, register
from core.types import AudioClip, Transcript
from speech.audio import (
    TARGET_SAMPLE_RATE,
    decode_to_pcm,
    pcm_duration_s,
    pcm_to_wav,
    rms,
)

log = get_logger(__name__)

#: Whisper model aliases users can put in config, mapped to the real model ids.
#:
#: These must be full repository ids, not bare size names: given "large-v3-turbo",
#: faster-whisper resolves it to ``Systran/faster-whisper-large-v3-turbo``, which answers
#: HTTP 401 on hf-mirror — the download fails and voice input is simply unavailable.
#: ``deepdml/faster-whisper-large-v3-turbo-ct2`` is the same CTranslate2 conversion and
#: downloads fine through the mirror.
_MODEL_ALIASES = {
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3": "large-v3",
    "medium": "medium",
    "small": "small",
    "base": "base",
    "tiny": "tiny",
}

#: Rough VRAM cost per model at int8, used for the diagnostics endpoint.
#: Keyed loosely (substring match) because the configured value can be a repository id.
_VRAM_HINT_MB = {
    "large-v3-turbo": 1600,
    "turbo": 1600,
    "large-v3": 3100,
    "medium": 1500,
    "small": 500,
    "base": 160,
    "tiny": 80,
}


def _prepare_cuda_dlls(explicit: Optional[List[str]] = None) -> List[str]:
    """Make CUDA 12's cuBLAS/cuDNN discoverable for CTranslate2 on Windows.

    CTranslate2 4.x links against cuBLAS 12 / cuDNN 9, but this machine's driver is
    CUDA 13 — the DLLs are simply not on ``PATH``, so transcription dies with
    ``Library cublas64_12.dll is not found`` even though the GPU clearly works (llama.cpp
    uses it happily). The DLLs are already on disk inside any PyTorch ``cu12x`` install,
    so instead of downloading another ~1GB of NVIDIA wheels we point the loader at them.

    Returns the directories that were actually added, for logging.
    """
    import os
    import sys

    candidates: List[Path] = []
    for entry in explicit or []:
        if entry:
            candidates.append(Path(entry))

    # Sibling environments: D:\conda-envs\<anything>\Lib\site-packages\torch\lib
    try:
        envs_root = Path(sys.executable).parents[1]  # <env>/python.exe -> <envs root>
        for sibling in sorted(envs_root.glob("*/Lib/site-packages/torch/lib")):
            candidates.append(sibling)
    except Exception:  # noqa: BLE001
        pass

    # NVIDIA pip packages, if anything installs them later.
    for package in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
        try:
            import site

            for base in site.getsitepackages():
                candidates.append(Path(base) / package)
        except Exception:  # noqa: BLE001
            pass

    added: List[str] = []
    for directory in candidates:
        try:
            if not directory.is_dir() or not any(directory.glob("cublas64_*.dll")):
                continue
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(directory))
            # Some libraries resolve through PATH only, so do both.
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(directory))
        except Exception:  # noqa: BLE001
            continue
    return added


@register(KIND_STT, "faster_whisper")
class FasterWhisperSTT:
    """Local, offline transcription via CTranslate2."""

    name = "faster_whisper"

    def __init__(
        self,
        model: str = "large-v3-turbo",
        *,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "zh",
        download_root: Optional[str] = None,
        endpoint: Optional[str] = None,
        fallback_model: Optional[str] = None,
        cuda_dll_dirs: Optional[List[str]] = None,
        idle_unload_s: int = 900,
        vad_filter: bool = True,
        beam_size: int = 5,
        cpu_fallback: bool = True,
    ) -> None:
        self.model_name = _MODEL_ALIASES.get(str(model).lower(), str(model))
        self.device = device
        #: VRAM this model holds once loaded — read by core.vram's coordinator, which evicts
        #: IndexTTS2 before whisper loads (they do not both fit next to the chat model).
        self.vram_gb = 1.8
        self.compute_type = compute_type
        self.language = language or None
        self.download_root = download_root
        #: HuggingFace endpoint used for the first-use download (~1.6GB). Defaults to the
        #: mirror; set ``speech.stt_endpoint: ""`` to use huggingface.co directly.
        self.endpoint = endpoint
        #: Used when ``model`` is a local directory that fails to load — a hand-downloaded
        #: folder is often still incomplete, and falling back beats "model is corrupt".
        self.fallback_model = fallback_model
        #: 额外搜索 CUDA 运行库的目录（见 _prepare_cuda_dlls）
        self.cuda_dll_dirs = list(cuda_dll_dirs or [])
        self.idle_unload_s = max(0, int(idle_unload_s))
        self.vad_filter = bool(vad_filter)
        self.beam_size = int(beam_size)
        self.cpu_fallback = bool(cpu_fallback)
        self._model: Any = None
        self._lock = threading.Lock()
        self._loaded_device = ""
        self._last_used = 0.0
        self._watchdog: Optional[asyncio.Task] = None

    # -- lifecycle ---------------------------------------------------------------------
    def _load_sync(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            # First-use download of a ~1.6GB CTranslate2 model. Two things make that
            # download work in this environment and fail without them:
            #   * HuggingFace is very slow/unreachable directly here, so use the mirror;
            #   * huggingface_hub 1.x prefers the Xet storage backend, which the mirror
            #     does not proxy — it answers 401 from cas-server.xethub.hf.co. Disabling
            #     Xet makes it fall back to plain HTTPS.
            if self.endpoint and not os.environ.get("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = self.endpoint
            if self.endpoint and not os.environ.get("HF_HUB_DISABLE_XET"):
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise BackendUnavailable(
                    "未安装 faster-whisper。请运行：pip install faster-whisper",
                    detail={"error": str(exc)},
                ) from exc

            added = _prepare_cuda_dlls(self.cuda_dll_dirs)
            if added:
                log.info("cuda dll directories added", extra={"dirs": added[:4]})

            # Ask the coordinator to make room: on a 16 GB card whisper and IndexTTS2 cannot
            # both live next to the chat model, and the failure mode is a native abort (the
            # whole service disappears), not a catchable Python error.
            if self.device != "cpu":
                try:
                    from core.vram import coordinator  # noqa: PLC0415

                    freed = coordinator.before_load("stt")
                    if freed:
                        log.info("unloaded %s to make room for whisper", ", ".join(freed))
                except Exception as exc:  # noqa: BLE001 - never block STT on the guard
                    log.warning("vram guard failed", extra={"error": str(exc)})

            attempts: List[tuple] = [(self.model_name, self.device, self.compute_type)]
            if self.cpu_fallback and self.device != "cpu":
                # A 16 GB card may already be full of the chat model; degrading to
                # CPU keeps voice input working (slower) instead of failing.
                attempts.append((self.model_name, "cpu", "int8"))
            # 本地目录可能是没下完的半成品（用户手动下载时很常见）。加载失败就退回
            # 仓库 id 联网取，而不是让语音输入一直报「模型损坏」。
            if self.fallback_model and self.fallback_model != self.model_name:
                attempts.append((self.fallback_model, self.device, self.compute_type))
            last_error: Optional[BaseException] = None
            for model_ref, device, compute_type in attempts:
                try:
                    log.info(
                        "loading whisper model",
                        extra={"model": model_ref, "device": device, "compute_type": compute_type},
                    )
                    self._model = WhisperModel(
                        model_ref,
                        device=device,
                        compute_type=compute_type,
                        download_root=self.download_root,
                        # A local directory (manual download) must never trigger a network
                        # round-trip: on this network it would simply hang and fail.
                        local_files_only=Path(model_ref).is_dir(),
                    )
                    self._loaded_device = device
                    return self._model
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    log.warning(
                        "whisper load failed",
                        extra={"device": device, "compute_type": compute_type, "error": str(exc)[:200]},
                    )
            raise BackendUnavailable(
                f"Whisper 模型加载失败（{self.model_name}）：{last_error}",
                detail={"model": self.model_name, "download_root": self.download_root},
            )

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._loaded_device = ""
        # Encourage the runtime to hand VRAM back before the LLM needs it.
        try:
            import gc

            gc.collect()
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - torch may not be installed at all
            pass
        log.info("whisper model unloaded (VRAM released)")

    def start_watchdog(self) -> None:
        if self.idle_unload_s <= 0 or self._watchdog is not None:
            return
        self._watchdog = asyncio.create_task(self._watchdog_loop(), name="stt-idle-watchdog")

    async def _watchdog_loop(self) -> None:
        interval = max(30, self.idle_unload_s // 4)
        while True:
            await asyncio.sleep(interval)
            if self._model is not None and time.time() - self._last_used > self.idle_unload_s:
                await asyncio.to_thread(self.unload)

    async def aclose(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        await asyncio.to_thread(self.unload)

    # -- transcription -----------------------------------------------------------------
    async def transcribe(
        self, audio: AudioClip, language: Optional[str] = None, prompt: Optional[str] = None
    ) -> Transcript:
        # decode_to_pcm returns (pcm, sample_rate, channels) — unpack all three.
        pcm, _rate, _channels = decode_to_pcm(audio.data)
        if len(pcm) < 3200:  # < 0.1 s at 16 kHz
            raise AudioError("录音太短了，按住说话多说几个字试试")
        level = rms(pcm)
        if level < 60:
            raise AudioError("没有检测到有效人声，请确认麦克风权限和输入设备")

        # faster-whisper accepts a path, a binary file object or an np.ndarray —
        # never raw headerless bytes. Wrapping in a WAV container is the cheapest
        # conversion that works on every version, and avoids a numpy dependency here.
        wav_bytes = pcm_to_wav(pcm, sample_rate=_rate or TARGET_SAMPLE_RATE, channels=1)

        def _run() -> Transcript:
            model = self._load_sync()
            self._last_used = time.time()
            segments, info = model.transcribe(
                io.BytesIO(wav_bytes),
                language=language or self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                vad_parameters={"min_silence_duration_ms": 400} if self.vad_filter else None,
                condition_on_previous_text=False,  # avoids runaway repetition on short clips
                initial_prompt=prompt or None,
                without_timestamps=False,
            )
            collected: List[Dict[str, Any]] = []
            pieces: List[str] = []
            for segment in segments:
                pieces.append(segment.text)
                collected.append(
                    {
                        "start": round(float(segment.start), 2),
                        "end": round(float(segment.end), 2),
                        "text": segment.text.strip(),
                        "no_speech_prob": round(float(getattr(segment, "no_speech_prob", 0.0)), 3),
                    }
                )
            text = "".join(pieces).strip()
            confidence = 1.0
            if collected:
                probs = [1.0 - float(seg.get("no_speech_prob") or 0.0) for seg in collected]
                confidence = sum(probs) / len(probs)
            return Transcript(
                text=text,
                language=getattr(info, "language", None) or language or self.language,
                confidence=round(confidence, 3),
                segments=tuple(collected),
                duration_s=pcm_duration_s(pcm),
            )

        return await asyncio.to_thread(_run)

    async def health(self) -> Dict[str, Any]:
        loaded = self._model is not None
        return {
            "ok": True,
            "name": self.name,
            "model": self.model_name,
            "loaded": loaded,
            "device": self._loaded_device or self.device,
            "compute_type": self.compute_type,
            "vram_hint_mb": _VRAM_HINT_MB.get(self.model_name, 0) if loaded else 0,
            "idle_unload_s": self.idle_unload_s,
        }


@register(KIND_STT, "mock")
class MockSTT:
    """Transcription stub for tests and offline demos."""

    name = "mock"

    def __init__(self, text: str = "你好，你还记得我喜欢喝什么吗？", **_ignored: Any) -> None:
        self.text = text

    async def transcribe(
        self, audio: AudioClip, language: Optional[str] = None, prompt: Optional[str] = None
    ) -> Transcript:
        return Transcript(text=self.text, language=language or "zh", confidence=0.99, duration_s=audio.duration_s)

    async def health(self) -> Dict[str, Any]:
        return {"ok": True, "name": self.name, "detail": "模拟语音识别"}

    async def aclose(self) -> None:  # pragma: no cover
        return None


__all__ = ["FasterWhisperSTT", "MockSTT"]
