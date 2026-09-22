"""Text-to-speech backends and the local-first router.

Requirements this layer has to satisfy simultaneously:

* **"语音输出要甜美"** — needs a voice that can actually sound sweet, not a robotic
  console reader. The shipped default is a local neural TTS model with zero-shot
  voice cloning and emotion control; a high-quality online voice is the fallback.
* **"支持设置各种人设"** — every persona carries its own voice id, reference audio,
  speed and emotion, so switching character changes how it *sounds*, not only what
  it says.
* **Coexisting with a 27B LLM on 16 GB of VRAM** — a local TTS model is expensive.
  It therefore loads lazily, is released after an idle period, and the router will
  happily use an online/offline-cheap backend instead when the GPU is busy.

Backends shipped
----------------
``indextts``        IndexTTS2 — zero-shot cloning + emotion control (the "sweet" one)
``edge``            edge-tts — Microsoft neural voices, free, needs network
``openai``          any OpenAI-compatible ``/audio/speech`` endpoint (self-hosted too)
``sapi``            Windows built-in SAPI — always-present last resort, no network
``tone``            generates a short beep; used only in tests

Streaming: ``synthesize`` yields :class:`SpeechChunk` objects. The engine feeds it
one sentence at a time so playback can start after the first sentence instead of
waiting for the whole answer — the difference between a snappy assistant and a
slow one, since generation and synthesis then overlap.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import struct
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from core.errors import AudioError, BackendUnavailable
from core.logging import get_logger
from core.registry import KIND_TTS, register
from core.types import SpeechChunk, VoiceSpec

log = get_logger(__name__)

#: Emotion label -> edge-tts style/pitch/rate adjustments.
#: edge-tts has no native emotion control, so persona expressiveness is expressed
#: as prosody. Keeping the mapping in one table means a new persona emotion is a
#: one-line change rather than a code path.
_EDGE_EMOTION_STYLE: Dict[str, Tuple[float, float]] = {
    # emotion: (rate_delta, pitch_delta_semitones)
    "sweet": (0.04, 2.5),
    "cheerful": (0.10, 2.0),
    "happy": (0.10, 2.0),
    "excited": (0.16, 3.0),
    "calm": (-0.06, -0.5),
    "neutral": (0.0, 0.0),
    "serious": (-0.08, -1.5),
    "sad": (-0.12, -2.0),
    "gentle": (-0.03, 1.0),
    "shy": (0.0, 1.5),
}


def _pitch_percent(semitones: float) -> str:
    """Deprecated: kept only for callers that still expect a percentage string.

    edge-tts used to accept ``+10%``/``-5Hz``; current versions validate the unit and
    reject a percentage for pitch ("Invalid pitch '+12%'"). Use
    :func:`_pitch_hz` instead.
    """
    percent = (2 ** (semitones / 12.0) - 1.0) * 100.0
    return f"{percent:+.0f}%"


#: Reference fundamental used to convert semitones into an absolute Hz delta.
#: edge-tts's pitch parameter takes Hz (``+30Hz``), not a ratio, so a ratio has to be
#: anchored somewhere. ~200 Hz is a typical female fundamental, and the value only
#: needs to be perceptually right, not physically exact.
_PITCH_BASE_HZ = 200.0


def _pitch_hz(semitones: float, base_hz: float = _PITCH_BASE_HZ) -> str:
    """Convert a semitone offset into edge-tts's ``+NNHz`` pitch format.

    Clamped to ±100 Hz: beyond that the result stops sounding like speech and edge-tts
    starts rejecting extreme values.
    """
    delta = base_hz * (2 ** (semitones / 12.0) - 1.0)
    delta = max(-100.0, min(100.0, delta))
    return f"{delta:+.0f}Hz"


# --------------------------------------------------------------------------------------
# edge-tts (online, high quality, free)
# --------------------------------------------------------------------------------------


@register(KIND_TTS, "edge")
class EdgeTTSBackend:
    """Microsoft Edge neural voices through the ``edge-tts`` package."""

    name = "edge"
    supports_cloning = False
    offline = False

    def __init__(
        self,
        *,
        default_voice: str = "zh-CN-XiaoxiaoNeural",
        timeout_s: float = 30.0,
        max_attempts: int = 3,
        retry_delay_s: float = 0.6,
    ) -> None:
        self.default_voice = default_voice
        self.timeout_s = float(timeout_s)
        #: ``NoAudioReceived`` from the edge service is transient; retrying inside the
        #: backend keeps the router from downgrading the user to a worse voice.
        self.max_attempts = max(1, int(max_attempts))
        self.retry_delay_s = max(0.0, float(retry_delay_s))

    @staticmethod
    def _import() -> Any:
        try:
            import edge_tts  # type: ignore

            return edge_tts
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                "未安装 edge-tts。请运行：pip install edge-tts",
                detail={"error": str(exc)},
            ) from exc

    def _prosody(self, voice: VoiceSpec) -> Dict[str, str]:
        emotion = (voice.emotion or "neutral").lower()
        rate_delta, pitch_delta = _EDGE_EMOTION_STYLE.get(emotion, (0.0, 0.0))
        alpha = max(0.0, min(1.0, float(voice.emotion_alpha)))
        rate = 1.0 + (rate_delta * alpha) + (float(voice.speed) - 1.0)
        rate = max(0.5, min(2.0, rate))
        # edge-tts expects a *rate percentage* but a *pitch in Hz*; mixing the two up
        # fails at request time with "Invalid pitch '+12%'".
        pitch = pitch_delta * alpha + float(voice.pitch_shift)
        return {"rate": f"{(rate - 1.0) * 100:+.0f}%", "pitch": _pitch_hz(pitch)}

    async def synthesize(
        self, text: str, voice: VoiceSpec, *, stream: bool = False
    ) -> AsyncIterator[SpeechChunk]:
        edge_tts = self._import()
        text = (text or "").strip()
        if not text:
            return
        voice_id = voice.voice_id if voice.voice_id and voice.voice_id != "default" else self.default_voice
        prosody = self._prosody(voice)

        # edge-tts intermittently reports "No audio was received" — the service accepts
        # the websocket, then closes it without synthesizing. It is transient (a handful
        # of attempts succeed immediately after), so one failure must not downgrade the
        # user to the robotic SAPI voice for the rest of the session. Retry with a short
        # backoff before giving up; the router only falls back if every attempt fails.
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                index = 0
                buffer = bytearray()
                communicate = edge_tts.Communicate(
                    text,
                    voice_id,
                    rate=prosody["rate"],
                    pitch=prosody["pitch"],
                    volume="+0%",
                )
                async for chunk in communicate.stream():
                    if chunk.get("type") != "audio" or not chunk.get("data"):
                        continue
                    data = bytes(chunk["data"])
                    if stream:
                        index += 1
                        yield SpeechChunk(data=data, mime="audio/mpeg", index=index, text=text)
                    else:
                        buffer.extend(data)
                if stream and index:
                    return
                if buffer:
                    yield SpeechChunk(
                        data=bytes(buffer), mime="audio/mpeg", index=1, is_final=True, text=text
                    )
                    return
                raise AudioError("edge-tts 未返回音频数据（服务端临时拒绝）")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                log.warning(
                    "edge-tts attempt failed, retrying",
                    extra={
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "voice": voice_id,
                        "rate": prosody["rate"],
                        "pitch": prosody["pitch"],
                        "chars": len(text),
                        "text_head": text[:40],
                    },
                )
                await asyncio.sleep(self.retry_delay_s * attempt)

        raise AudioError(
            f"edge-tts 合成失败（已重试 {self.max_attempts} 次，可能无网络或被限流）：{last_error}",
            detail={"voice": voice_id, "attempts": self.max_attempts},
        )

    async def health(self) -> Dict[str, Any]:
        try:
            self._import()
        except BackendUnavailable as exc:
            return {"ok": False, "name": self.name, "error": exc.message}
        return {"ok": True, "name": self.name, "offline": False, "default_voice": self.default_voice}

    async def list_voices(self) -> List[Dict[str, str]]:
        edge_tts = self._import()
        voices = await edge_tts.list_voices()
        out = [
            {
                "name": v.get("ShortName", ""),
                "locale": v.get("Locale", ""),
                "gender": v.get("Gender", ""),
                "friendly": v.get("FriendlyName", ""),
            }
            for v in voices
            if str(v.get("Locale", "")).startswith("zh")
        ]
        return out

    async def aclose(self) -> None:  # pragma: no cover
        return None


# --------------------------------------------------------------------------------------
# IndexTTS2 (local, cloning + emotion)
# --------------------------------------------------------------------------------------


@register(KIND_TTS, "indextts")
class IndexTTSBackend:
    """Local IndexTTS2: zero-shot voice cloning with explicit emotion control.

    Why this model: it accepts a short reference clip *plus* a separate emotion
    descriptor, which is exactly what "甜美音色 + 可切换人设" needs — the timbre
    comes from the reference audio, the mood comes from the emotion field, and the
    same persona can be soft-spoken or excited without re-recording anything.

    The model is loaded lazily and released when idle, because at ~2 GB it competes
    directly with the LLM for a 16 GB card.
    """

    name = "indextts"
    supports_cloning = True
    offline = True
    #: VRAM estimate for core.vram's coordinator: IndexTTS2 loads via accelerate, which by
    #: default claims 90% of whatever is free — so this is the number that decides whether
    #: whisper gets evicted first (they do not both fit next to the chat model).
    vram_gb = 4.0

    def __init__(
        self,
        *,
        model_dir: Optional[str] = None,
        device: str = "cuda:0",
        idle_unload_s: int = 600,
        default_reference: Optional[str] = None,
        fp16: bool = True,
        python_exe: Optional[str] = None,
        worker_script: Optional[str] = None,
        worker_timeout_s: float = 300.0,
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.idle_unload_s = max(0, int(idle_unload_s))
        self.default_reference = default_reference
        self.fp16 = bool(fp16)
        # Separate-environment bridge. ``indextts`` needs torch 2.8 + numpy 2.2.6 +
        # transformers + keras…; installing that into the app's environment can break a
        # working install, so by default the model runs in ``venvs/tts`` behind a JSON
        # worker (tools/tts_worker.py). In-process import stays as a fallback for anyone
        # who did install it locally.
        self.python_exe = python_exe
        self.worker_script = worker_script
        self.worker_timeout_s = float(worker_timeout_s)
        self._worker: Any = None
        self._worker_lock = threading.Lock()
        self._tts: Any = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._watchdog: Optional[asyncio.Task] = None
        self._available: Optional[bool] = None

    # -- external worker ---------------------------------------------------------------
    @property
    def uses_worker(self) -> bool:
        return bool(self.python_exe and Path(str(self.python_exe)).exists() and self.worker_script)

    def _worker_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One JSON line out, one JSON line back, on a long-lived worker process.

        Spawning the interpreter per sentence would add ~1s of import time to every
        chunk, so the process is kept alive and reused; it is what makes the per-chunk
        cost the model's own latency rather than Python's startup cost.
        """
        import subprocess

        with self._worker_lock:
            process = self._worker
            if process is None or process.poll() is not None:
                if process is not None:
                    log.warning("tts worker exited, restarting", extra={"code": process.returncode})
                _log_path = Path.cwd() / "logs"
                _log_path.mkdir(parents=True, exist_ok=True)
                stderr_target = open(_log_path / "tts-worker.log", "a", encoding="utf-8")
                process = subprocess.Popen(  # noqa: S603 - fixed, configured interpreter
                    [str(self.python_exe), str(self.worker_script)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_target,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    cwd=str(Path(str(self.worker_script)).resolve().parents[1]),
                )
                self._worker = process
            assert process.stdin is not None and process.stdout is not None
            payload = {**payload, "id": int(time.time() * 1000) % 1_000_000}
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
            deadline = time.time() + self.worker_timeout_s
            while True:
                if process.poll() is not None:
                    self._worker = None
                    raise AudioError(
                        f"IndexTTS2 worker 进程退出（code {process.returncode}），"
                        "详见 logs/tts-worker.log"
                    )
                line = process.stdout.readline()
                if line:
                    break
                if time.time() > deadline:
                    raise AudioError(f"IndexTTS2 worker 超时（{self.worker_timeout_s:.0f}s 无响应）")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AudioError(f"IndexTTS2 worker 返回了非 JSON 内容：{line[:200]}") from exc
        return response

    def worker_ping(self) -> Dict[str, Any]:
        return self._worker_request({"cmd": "ping"})

    def stop_worker(self) -> None:
        with self._worker_lock:
            process, self._worker = self._worker, None
        if process is None:
            return
        try:
            if process.stdin and process.poll() is None:
                process.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                process.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            process.wait(timeout=5)
        except Exception:  # noqa: BLE001
            process.kill()

    # -- lifecycle ---------------------------------------------------------------------
    def _resolve_cfg_path(self) -> Optional[str]:
        """Locate the ``config.yaml`` IndexTTS2 needs.

        The upstream signature is ``IndexTTS2(cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints", ...)`` and it calls ``OmegaConf.load(cfg_path)``
        unconditionally — passing ``None`` (as this backend used to) dies immediately
        with ``TypeError: Unexpected file type``, which looks nothing like a
        configuration problem.
        """
        if not self.model_dir:
            return None
        root = Path(str(self.model_dir))
        for candidate in (root / "config.yaml", root / "checkpoints" / "config.yaml"):
            if candidate.is_file():
                return str(candidate)
        return None

    def _aux_paths(self) -> Optional[Dict[str, str]]:
        """Point the loader at the pre-downloaded auxiliary models, if all are present.

        Returning ``None`` lets the library call ``ensure_models_available()`` and fetch
        several GB by itself. Downloads are the user's call here, so a complete set is
        passed explicitly and an incomplete one is reported (see ``health``) instead of
        silently pulling gigabytes.
        """
        if not self.model_dir:
            return None
        cache = Path(str(self.model_dir)) / "hf_cache"
        w2v_dir = cache / "w2v-bert-2.0"
        semantic = cache / "semantic_codec_model.safetensors"
        campplus = cache / "campplus_cn_common.bin"
        bigvgan = cache / "bigvgan"
        if (
            w2v_dir.is_dir()
            and any(w2v_dir.iterdir())
            and semantic.is_file()
            and campplus.is_file()
            and (bigvgan / "config.json").is_file()
            and (bigvgan / "bigvgan_generator.pt").is_file()
        ):
            return {
                "w2v_bert": str(w2v_dir),
                "semantic_codec": str(semantic),
                "campplus": str(campplus),
                "bigvgan": str(bigvgan),
            }
        return None

    def missing_aux_models(self) -> List[str]:
        """Which auxiliary models are absent (used by ``health`` for an honest report)."""
        if not self.model_dir:
            return ["模型目录未配置"]
        cache = Path(str(self.model_dir)) / "hf_cache"
        missing: List[str] = []
        if not (cache / "w2v-bert-2.0").is_dir():
            missing.append("hf_cache/w2v-bert-2.0（语义特征）")
        if not (cache / "semantic_codec_model.safetensors").is_file():
            missing.append("hf_cache/semantic_codec_model.safetensors（语义编解码）")
        if not (cache / "campplus_cn_common.bin").is_file():
            missing.append("hf_cache/campplus_cn_common.bin（说话人向量）")
        if not (cache / "bigvgan" / "bigvgan_generator.pt").is_file():
            missing.append("hf_cache/bigvgan/（声码器）")
        if self._resolve_cfg_path() is None:
            missing.append("config.yaml（模型配置）")
        return missing

    def _load_sync(self) -> Any:
        if self._tts is not None:
            return self._tts
        with self._lock:
            if self._tts is not None:
                return self._tts
            try:
                from indextts.infer_v2 import IndexTTS2  # type: ignore
            except Exception as exc:  # noqa: BLE001
                self._available = False
                raise BackendUnavailable(
                    "未安装 IndexTTS2（indextts 包）。装进当前环境即可："
                    "python -m pip install -e vendor\\index-tts -i https://pypi.tuna.tsinghua.edu.cn/simple",
                    detail={"error": str(exc)},
                ) from exc
            if not self.model_dir:
                self._available = False
                raise BackendUnavailable("未配置 IndexTTS2 模型目录（config.speech.indextts.model_dir）")
            cfg_path = self._resolve_cfg_path()
            if cfg_path is None:
                self._available = False
                raise BackendUnavailable(
                    f"模型目录里找不到 config.yaml：{self.model_dir}",
                    detail={"expected": ["<model_dir>/config.yaml", "<model_dir>/checkpoints/config.yaml"]},
                )
            missing = self.missing_aux_models()
            if missing:
                self._available = False
                raise BackendUnavailable(
                    "缺少 IndexTTS2 辅助模型：" + "、".join(missing),
                    detail={"hint": "python -c \"from indextts.utils.model_download import ensure_models_available as e; print(e(r'<model_dir>'))\""},
                )
            log.info(
                "loading IndexTTS2",
                extra={"model_dir": self.model_dir, "device": self.device, "cfg": cfg_path},
            )
            # Make room first: IndexTTS2 loads through accelerate, which by default claims
            # *90% of whatever VRAM is free* — leaving whisper (or the next whisper call)
            # nothing, and the resulting CUDA failure aborts the process natively instead of
            # raising. On this card whisper + IndexTTS2 + the chat model do not all fit.
            if str(self.device).startswith("cuda"):
                try:
                    from core.vram import coordinator  # noqa: PLC0415

                    freed = coordinator.before_load("indextts")
                    if freed:
                        log.info("unloaded %s to make room for IndexTTS2", ", ".join(freed))
                except Exception as exc:  # noqa: BLE001 - never block TTS on the guard
                    log.warning("vram guard failed", extra={"error": str(exc)})
            try:
                self._tts = IndexTTS2(
                    model_dir=str(self.model_dir),
                    cfg_path=cfg_path,
                    use_fp16=self.fp16,
                    device=self.device,
                    aux_paths=self._aux_paths(),
                )
            except TypeError:
                # Older revisions take ``is_fp16`` and no ``aux_paths``; accept both
                # rather than pinning to one upstream revision.
                self._tts = IndexTTS2(
                    model_dir=str(self.model_dir), cfg_path=cfg_path, is_fp16=self.fp16, device=self.device
                )
            self._available = True
            return self._tts

    def unload(self) -> None:
        with self._lock:
            self._tts = None
        if self._worker is not None:
            # Ask the worker to release VRAM; if the bridge is configured we never hold
            # the model in this process, so this is the only place it lives.
            try:
                self._worker_request({"cmd": "unload"})
            except Exception as exc:  # noqa: BLE001
                log.debug("tts worker unload failed", extra={"error": str(exc)})
        try:
            import gc

            gc.collect()
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        log.info("IndexTTS2 unloaded (VRAM released)")

    def start_watchdog(self) -> None:
        if self.idle_unload_s <= 0 or self._watchdog is not None:
            return
        self._watchdog = asyncio.create_task(self._watchdog_loop(), name="tts-idle-watchdog")

    async def _watchdog_loop(self) -> None:
        interval = max(30, self.idle_unload_s // 3)
        while True:
            await asyncio.sleep(interval)
            if self._tts is not None and time.time() - self._last_used > self.idle_unload_s:
                await asyncio.to_thread(self.unload)

    async def aclose(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        await asyncio.to_thread(self.unload)

    # -- synthesis ---------------------------------------------------------------------
    def _synthesize_sync(self, text: str, voice: VoiceSpec, output_path: str) -> None:
        tts = self._load_sync()
        self._last_used = time.time()
        reference = voice.reference_audio or self.default_reference
        if not reference:
            raise AudioError(
                "该人设没有配置参考音频（voice.reference_audio），无法用 IndexTTS2 克隆音色。"
                "请在 config/personas.yaml 里指定，或改用 edge 音色。"
            )
        kwargs: Dict[str, Any] = {"spk_audio_prompt": reference, "output_path": output_path, "verbose": False}
        style = dict(voice.style or {})
        # Emotion handling differs between IndexTTS2 revisions: an explicit vector
        # wins, then a text descriptor, then nothing (the reference sets the mood).
        emotion_vector = style.get("emotion_vector")
        emo_alpha = float(style.get("emotion_alpha", voice.emotion_alpha))
        if emotion_vector is not None:
            kwargs["emo_vector"] = emotion_vector
            kwargs["emo_alpha"] = emo_alpha
        elif voice.emotion and voice.emotion != "neutral":
            kwargs["use_emo_text"] = True
            kwargs["emo_text"] = str(style.get("emotion_text") or _emotion_text_for(voice.emotion))
            kwargs["emo_alpha"] = emo_alpha
        # NOTE: ``text`` is passed positionally by the caller below. Setting it in
        # kwargs as well would raise "got multiple values for keyword argument" and
        # silently skip the fallback branch, dropping every emotion/cloning arg.
        try:
            tts.infer(text=text, **kwargs)
        except TypeError as exc:
            # Genuine signature mismatch (older IndexTTS2 revisions): retry with the
            # minimal documented call, but keep the audio path working.
            log.warning(
                "IndexTTS2 signature mismatch, retrying with minimal args",
                extra={"error": str(exc)[:200]},
            )
            tts.infer(spk_audio_prompt=reference, text=text, output_path=output_path)

    async def synthesize(
        self, text: str, voice: VoiceSpec, *, stream: bool = False
    ) -> AsyncIterator[SpeechChunk]:
        text = (text or "").strip()
        if not text:
            return
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory(prefix="chatbot-tts-") as tmpdir:
            output = str(_Path(tmpdir) / "out.wav")
            if self.uses_worker:
                await asyncio.to_thread(self._synthesize_via_worker, text, voice, output)
            else:
                await asyncio.to_thread(self._synthesize_sync, text, voice, output)
            path = _Path(output)
            if not path.exists() or path.stat().st_size == 0:
                raise AudioError("IndexTTS2 未生成音频文件")
            data = await asyncio.to_thread(path.read_bytes)
        yield SpeechChunk(data=data, mime="audio/wav", index=1, is_final=True, text=text)

    def _reference_for(self, voice: VoiceSpec) -> str:
        reference = voice.reference_audio or self.default_reference
        if not reference:
            raise AudioError(
                "该人设没有配置参考音频（voice.reference_audio），无法用 IndexTTS2 克隆音色。"
                "请在 config/personas.yaml 里指定，或改用 edge 音色。"
            )
        return str(reference)

    def _synthesize_via_worker(self, text: str, voice: VoiceSpec, output_path: str) -> None:
        reference = self._reference_for(voice)
        self._last_used = time.time()
        style = dict(voice.style or {})
        response = self._worker_request(
            {
                "cmd": "synthesize",
                "text": text,
                "output": output_path,
                "reference": str(Path(reference).resolve()),
                "model_dir": str(Path(str(self.model_dir)).resolve()) if self.model_dir else None,
                "device": self.device,
                "fp16": self.fp16,
                "emotion": voice.emotion,
                "emotion_alpha": float(style.get("emotion_alpha", voice.emotion_alpha)),
                "emotion_text": style.get("emotion_text"),
                "emotion_vector": style.get("emotion_vector"),
            }
        )
        if not response.get("ok"):
            detail = str(response.get("error") or "未知错误")
            raise AudioError(
                f"IndexTTS2 合成失败：{detail}",
                detail={"hint": "详见 logs/tts-worker.log"},
            )
        log.debug(
            "indextts worker done",
            extra={"ms": response.get("ms"), "bytes": response.get("bytes"), "chars": len(text)},
        )

    async def health(self) -> Dict[str, Any]:
        if self.uses_worker:
            info: Dict[str, Any] = {
                "ok": None,
                "name": self.name,
                "offline": True,
                "supports_cloning": True,
                "model_dir": self.model_dir,
                "device": self.device,
                "python_exe": self.python_exe,
                "bridge": "worker",
            }
            try:
                pong = await asyncio.to_thread(self.worker_ping)
                # 光有 torch 不算可用：环境里没有 indextts 时，合成一定失败，
                # 这里必须如实报 false，而不是等到用户说话时才炸。
                info["indextts_installed"] = bool(pong.get("indextts"))
                info["ok"] = bool(pong.get("ok")) and bool(pong.get("cuda")) and bool(pong.get("indextts"))
                if not pong.get("indextts"):
                    info["error"] = (
                        "TTS 环境里还没装 index-tts 包：" + str(self.python_exe) +
                        " -m pip install -e vendor\\index-tts"
                    )
                info["loaded"] = bool(pong.get("loaded"))
                info["torch"] = pong.get("torch")
                info["device_name"] = pong.get("device_name")
                info["vram_free_mb"] = pong.get("vram_free_mb")
                if not pong.get("cuda"):
                    info["error"] = "TTS 环境里的 torch 看不到 CUDA"
            except Exception as exc:  # noqa: BLE001
                info["ok"] = False
                info["error"] = f"TTS worker 不可用：{type(exc).__name__}: {exc}"
            if info["ok"] and self.model_dir and not Path(str(self.model_dir)).exists():
                info["ok"] = False
                info["error"] = f"模型目录不存在：{self.model_dir}（先运行 scripts\\install-tts.ps1）"
            if info["ok"] and not (self.default_reference and Path(str(self.default_reference)).exists()):
                # Cloning needs a reference clip; without one every call would fail, so
                # report it here instead of at the first spoken sentence.
                info["warning"] = f"缺少参考音频：{self.default_reference}（人设里的 voice.reference_audio 也可以）"
            return info
        info = {
            "ok": self._available if self._available is not None else None,
            "name": self.name,
            "loaded": self._tts is not None,
            "offline": True,
            "supports_cloning": True,
            "model_dir": self.model_dir,
            "device": self.device,
            "bridge": "in-process",
        }
        if self._available is None:
            try:
                import importlib.util

                info["ok"] = importlib.util.find_spec("indextts") is not None
                if not info["ok"]:
                    info["error"] = (
                        "未安装 IndexTTS2（indextts 包）。装进当前环境即可："
                        "python -m pip install -e vendor\\index-tts -i https://pypi.tuna.tsinghua.edu.cn/simple"
                    )
                    info["hint"] = (
                        "注意它会钉住 numpy==2.2.6；若同时要用 GPT-SoVITS，装完把 numpy 恢复到 2.2.6"
                    )
            except Exception:  # noqa: BLE001
                info["ok"] = False
        return info


def _emotion_text_for(emotion: str) -> str:
    mapping = {
        "sweet": "用甜美、温柔、撒娇的语气说",
        "cheerful": "用轻快、开心、有活力的语气说",
        "calm": "用平静、舒缓、放松的语气说",
        "serious": "用认真、严肃、沉稳的语气说",
        "sad": "用低落、难过的语气说",
        "gentle": "用轻柔、体贴的语气说",
        "excited": "用兴奋、激动的语气说",
        "shy": "用害羞、腼腆的语气说",
    }
    return mapping.get((emotion or "").lower(), f"用{emotion}的语气说")


# --------------------------------------------------------------------------------------
# OpenAI-compatible / SAPI / test backends
# --------------------------------------------------------------------------------------


@register(KIND_TTS, "openai")
class OpenAITTSBackend:
    """Any ``/v1/audio/speech`` endpoint — cloud API or a self-hosted server."""

    name = "openai"
    supports_cloning = False
    offline = False

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8090/v1",
        api_key: str = "sk-local",
        model: str = "tts-1",
        default_voice: str = "alloy",
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.default_voice = default_voice
        self.timeout_s = float(timeout_s)

    async def synthesize(
        self, text: str, voice: VoiceSpec, *, stream: bool = False
    ) -> AsyncIterator[SpeechChunk]:
        try:
            import httpx
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable("缺少 httpx 依赖", detail={"error": str(exc)}) from exc
        text = (text or "").strip()
        if not text:
            return
        payload = {
            "model": self.model,
            "input": text,
            "voice": voice.voice_id if voice.voice_id and voice.voice_id != "default" else self.default_voice,
            "response_format": "mp3",
            "speed": max(0.25, min(4.0, float(voice.speed))),
        }
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_s,
            # Same reason as the LLM backends: a system proxy (Clash etc.) would
            # swallow requests to a local TTS server and answer 502.
            trust_env=False,
        ) as client:
            response = await client.post(
                "/audio/speech",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            if response.status_code >= 400:
                raise AudioError(
                    f"TTS 服务返回 {response.status_code}",
                    detail={"body": response.text[:300], "base_url": self.base_url},
                )
            yield SpeechChunk(data=response.content, mime="audio/mpeg", index=1, is_final=True, text=text)

    async def health(self) -> Dict[str, Any]:
        return {"ok": True, "name": self.name, "base_url": self.base_url, "offline": False}

    async def aclose(self) -> None:  # pragma: no cover
        return None


@register(KIND_TTS, "sapi")
class SapiTTSBackend:
    """Windows built-in speech synthesis.

    Deliberately last-resort: the voices are dated, but it needs no network, no
    model download and no GPU, so a user always hears *something* even on a fresh
    Windows box with nothing installed yet.
    """

    name = "sapi"
    supports_cloning = False
    offline = True

    def __init__(self, *, default_voice: str = "", rate: int = 1) -> None:
        self.default_voice = default_voice
        self.rate = int(rate)

    def _synthesize_sync(self, text: str, voice: VoiceSpec, output_path: str, text_path: str) -> None:
        """Run the SAPI synthesizer synchronously (called from a worker thread).

        Deliberately *not* ``asyncio.create_subprocess_exec``: on Windows the
        asyncio subprocess API needs the Proactor loop, and under uvicorn the running
        loop can be a Selector loop, where it raises ``NotImplementedError``. A plain
        ``subprocess.run`` inside ``asyncio.to_thread`` works identically under every
        loop and cannot deadlock on stdio. The script is passed via ``input=`` (not a
        command-line argument) so no shell quoting or %-expansion can corrupt it.
        """
        import subprocess

        # The text travels through a UTF-8 *file*, not stdin or argv: with
        # ``-NonInteractive`` a redirected stdin yields an empty
        # ``[Console]::In.ReadToEnd()``, and the synthesizer then writes a header-only
        # WAV (46 bytes, zero frames) while still exiting 0.
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(text)

        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {max(-10, min(10, self.rate + int(round((float(voice.speed) - 1.0) * 5))))}; "
            + (f"$s.SelectVoice('{self.default_voice}'); " if self.default_voice else "")
            + f"$s.SetOutputToWaveFile('{output_path}'); "
            + f"$t = Get-Content -LiteralPath '{text_path}' -Raw -Encoding UTF8; "
            + "$s.Speak($t); $s.Dispose()"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "-"],
            input=script.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise AudioError(
                "Windows SAPI 合成失败",
                detail={
                    "code": completed.returncode,
                    "stderr": (completed.stderr or b"").decode("utf-8", "replace")[:300],
                },
            )

    async def synthesize(
        self, text: str, voice: VoiceSpec, *, stream: bool = False
    ) -> AsyncIterator[SpeechChunk]:
        text = (text or "").strip()
        if not text:
            return
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory(prefix="chatbot-sapi-") as tmpdir:
            output = _Path(tmpdir) / "out.wav"
            text_file = _Path(tmpdir) / "text.txt"
            await asyncio.to_thread(self._synthesize_sync, text, voice, str(output), str(text_file))
            if not output.exists():
                raise AudioError("Windows SAPI 未生成音频文件")
            data = await asyncio.to_thread(output.read_bytes)

        # Guard against the silent-empty-output failure mode: a WAV with no frames is
        # useless, and reporting it as success is worse than failing loudly, because the
        # router would then never try the next backend.
        if len(data) <= 44 or not data.startswith(b"RIFF"):
            raise AudioError(
                "Windows SAPI 返回了空音频（仅 WAV 头）",
                detail={"bytes": len(data), "voice": self.default_voice or "(default)"},
            )
        yield SpeechChunk(data=data, mime="audio/wav", index=1, is_final=True, text=text)

    async def health(self) -> Dict[str, Any]:
        import sys

        supported = sys.platform == "win32"
        return {
            "ok": supported,
            "name": self.name,
            "offline": True,
            "error": None if supported else "仅 Windows 可用",
        }

    async def aclose(self) -> None:  # pragma: no cover
        return None


@register(KIND_TTS, "tone")
class ToneTTSBackend:
    """Emits a short sine tone. Used by tests to assert the audio pipeline end to end."""

    name = "tone"
    supports_cloning = False
    offline = True

    def __init__(self, *, sample_rate: int = 22050, base_freq: float = 660.0) -> None:
        self.sample_rate = sample_rate
        self.base_freq = base_freq

    async def synthesize(
        self, text: str, voice: VoiceSpec, *, stream: bool = False
    ) -> AsyncIterator[SpeechChunk]:
        text = text or ""
        duration = min(2.0, max(0.15, len(text) * 0.045))
        import array
        import wave

        frames = int(self.sample_rate * duration)
        samples = array.array("h", [0]) * frames
        freq = self.base_freq * max(0.5, min(2.0, float(voice.speed)))
        for index in range(frames):
            value = math.sin(2 * math.pi * freq * index / self.sample_rate)
            # Fade in/out so the tone is not a click.
            envelope = min(1.0, index / 400.0, (frames - index) / 400.0)
            samples[index] = int(value * envelope * 8000)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:  # type: ignore[arg-type]
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(samples.tobytes())
        yield SpeechChunk(data=buffer.getvalue(), mime="audio/wav", index=1, is_final=True, text=text[:40])

    async def health(self) -> Dict[str, Any]:
        return {"ok": True, "name": self.name, "offline": True, "detail": "测试音源"}

    async def aclose(self) -> None:  # pragma: no cover
        return None


# --------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------


@register(KIND_TTS, "router")
class TTSRouter:
    """Chooses a TTS backend per request, local-first with health awareness.

    Selection logic, in order:

    1. If the persona pins a backend (``voice.backend``), use it. An explicit
       choice is never second-guessed.
    2. Otherwise walk ``speech.tts_preference`` and pick the first backend whose
       health probe passes. Health is cached briefly so a per-sentence synthesis
       does not re-probe the model server.
    3. Never let an online backend win over a healthy offline one when
       ``prefer_offline`` is set — a local voice keeps working with no network and
       leaks no text to a third party.

    A failed backend is remembered briefly, so a broken local install does not add
    a timeout to every sentence while the user is mid-conversation.
    """

    name = "router"
    supports_cloning = True
    offline = True  # capable of offline operation; individual backends vary

    def __init__(
        self,
        backends: Dict[str, Any],
        *,
        preference: Optional[List[str]] = None,
        prefer_offline: bool = True,
        priorities: Optional[Dict[str, int]] = None,
        health_cache_s: float = 45.0,
        failure_cooldown_s: float = 120.0,
        cache_max_bytes: int = 48 * 1024 * 1024,
    ) -> None:
        self.backends = backends
        self.preference = list(preference or [b for b in backends])
        self.prefer_offline = bool(prefer_offline)
        #: Explicit quality ranking (lower wins). This is what keeps order honest:
        #: without it, ``prefer_offline`` would promote Windows SAPI above edge-tts,
        #: trading a lot of voice quality for a little privacy. Offline preference is
        #: only applied as a tie-break among backends of comparable priority.
        self.priorities = dict(priorities or {})
        self.health_cache_s = float(health_cache_s)
        self.failure_cooldown_s = float(failure_cooldown_s)
        self._health: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._failed_until: Dict[str, float] = {}
        self._last_used: Optional[str] = None
        # Result cache for complete (non-streaming) clips, keyed by backend + voice +
        # text. Edge-tts costs 1.5–5s per sentence over the network; a persona greeting
        # or a repeated short answer ("好呀") should not pay that twice. Streaming calls
        # bypass the cache: progressive output is the whole point of that path.
        self.cache_max_bytes = max(0, int(cache_max_bytes))
        self._cache: "OrderedDict[str, Tuple[str, bytes]]" = OrderedDict()
        self._cache_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0

    # -- cache -------------------------------------------------------------------------
    @staticmethod
    def _cache_key(backend_name: str, voice: VoiceSpec, text: str) -> str:
        material = "|".join(
            [
                backend_name,
                str(voice.voice_id or ""),
                str(voice.emotion or ""),
                f"{float(voice.speed or 1.0):.3f}",
                f"{float(voice.emotion_alpha or 0.0):.3f}",
                f"{float(voice.pitch_shift or 0.0):.3f}",
                text,
            ]
        )
        return hashlib.sha1(material.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[Tuple[str, bytes]]:
        entry = self._cache.get(key)
        if entry is None:
            self.cache_misses += 1
            return None
        self._cache.move_to_end(key)
        self.cache_hits += 1
        return entry

    def _cache_put(self, key: str, mime: str, data: bytes) -> None:
        if self.cache_max_bytes <= 0 or not data or len(data) > self.cache_max_bytes:
            return
        previous = self._cache.pop(key, None)
        if previous is not None:
            self._cache_bytes -= len(previous[1])
        self._cache[key] = (mime, data)
        self._cache_bytes += len(data)
        while self._cache_bytes > self.cache_max_bytes and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= len(evicted[1])

    def cache_stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._cache),
            "bytes": self._cache_bytes,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
        }

    # -- selection ---------------------------------------------------------------------
    def _rank(self, name: str) -> Tuple[int, int, int]:
        """Sort key for a backend: (quality priority, online-ness, preference index)."""
        explicit = self.priorities.get(name)
        # An unlisted backend inherits its position in tts_preference, offset so that
        # explicit priorities still dominate.
        preference_index = self.preference.index(name) if name in self.preference else len(self.preference)
        quality = explicit if isinstance(explicit, int) else 50 + preference_index
        offline = 0 if getattr(self.backends.get(name), "offline", False) else 1
        # prefer_offline is a *tie-break*: it only reorders backends whose quality
        # priority is identical, which is exactly the privacy-vs-cost decision it was
        # meant to express (two similar-quality engines, prefer the local one).
        online_rank = offline if self.prefer_offline else 0
        return (quality, online_rank, preference_index)
    async def _health_of(self, name: str) -> Dict[str, Any]:
        cached = self._health.get(name)
        now = time.time()
        if cached and now - cached[0] < self.health_cache_s:
            return cached[1]
        backend = self.backends.get(name)
        if backend is None:
            info = {"ok": False, "error": "未注册该后端"}
        else:
            try:
                info = await backend.health()
            except Exception as exc:  # noqa: BLE001
                info = {"ok": False, "error": str(exc)[:200]}
        self._health[name] = (now, info)
        return info

    async def select(self, voice: VoiceSpec) -> Any:
        if voice.backend:
            backend = self.backends.get(voice.backend)
            if backend is None:
                raise BackendUnavailable(f"人设指定的语音后端未注册：{voice.backend}")
            return backend

        now = time.time()
        # Order by declared quality first, then by privacy/offline preference, then by
        # the configured preference list. See ``_rank`` for why quality must outrank
        # "offline".
        ordered = sorted(self.preference, key=self._rank)

        candidates: List[Tuple[str, Any]] = []
        for name in ordered:
            backend = self.backends.get(name)
            if backend is None:
                continue
            if self._failed_until.get(name, 0.0) > now:
                continue
            info = await self._health_of(name)
            if info.get("ok"):
                candidates.append((name, backend))
        if not candidates:
            # Everything is unhealthy or cooling down: retry anything known-good
            # rather than failing the user's request outright.
            for name in ordered:
                backend = self.backends.get(name)
                if backend is not None:
                    candidates.append((name, backend))
        if not candidates:
            raise BackendUnavailable("没有可用的语音合成后端，请检查 speech.tts_preference 配置")
        return candidates[0][1]

    # -- synthesis ---------------------------------------------------------------------
    async def synthesize(
        self, text: str, voice: VoiceSpec, *, stream: bool = False
    ) -> AsyncIterator[SpeechChunk]:
        backend = await self.select(voice)

        cache_key = ""
        buffered = bytearray()
        buffered_mime = ""
        if not stream:
            cache_key = self._cache_key(backend.name, voice, text or "")
            cached = self._cache_get(cache_key)
            if cached is not None:
                cached_mime, cached_data = cached
                log.debug("tts cache hit", extra={"backend": backend.name, "bytes": len(cached_data)})
                self._last_used = backend.name
                yield SpeechChunk(
                    data=cached_data, mime=cached_mime, index=1, is_final=True, text=text
                )
                return

        # NOTE: ``_last_used`` is only advanced when this backend actually delivers
        # audio. Setting it up-front made the ``X-TTS-Backend`` response header claim
        # "edge" on responses that were really produced by the SAPI fallback, which
        # sent debugging in entirely the wrong direction.
        log.debug(
            "tts selected",
            extra={
                "backend": backend.name,
                "pinned": voice.backend,
                "offline": getattr(backend, "offline", None),
                "voice_id": voice.voice_id,
                "emotion": voice.emotion,
                "chars": len(text or ""),
            },
        )
        produced = 0
        try:
            async for chunk in backend.synthesize(text, voice, stream=stream):
                produced += len(chunk.data)
                if produced > 44:
                    self._last_used = backend.name
                if not stream:
                    buffered.extend(chunk.data)
                    buffered_mime = chunk.mime or buffered_mime
                yield chunk
        except Exception as exc:  # noqa: BLE001
            self._failed_until[backend.name] = time.time() + self.failure_cooldown_s
            log.warning(
                "tts backend failed, falling back",
                extra={
                    "backend": backend.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "produced_bytes": produced,
                },
            )
            for name in self.preference:
                if name == backend.name:
                    continue
                fallback = self.backends.get(name)
                if fallback is None:
                    continue
                try:
                    fallback_bytes = 0
                    async for chunk in fallback.synthesize(text, voice, stream=stream):
                        fallback_bytes += len(chunk.data)
                        self._last_used = fallback.name
                        yield chunk
                    log.info(
                        "tts fell back to another backend",
                        extra={"from": backend.name, "to": fallback.name, "bytes": fallback_bytes},
                    )
                    return
                except Exception as inner:  # noqa: BLE001
                    log.debug(
                        "tts fallback failed",
                        extra={"backend": name, "error": f"{type(inner).__name__}: {inner}"},
                    )
            raise AudioError(
                f"所有语音后端都失败了（最后错误：{type(exc).__name__}: {exc}）",
                detail={"tried": list(self.preference), "first_error": str(exc)[:200]},
            ) from exc

        # A backend that yielded nothing usable is a failure, not a success: returning
        # an empty WAV makes the UI play silence and hides which engine is broken.
        if produced <= 44:
            self._failed_until[backend.name] = time.time() + self.failure_cooldown_s
            log.warning(
                "tts backend produced empty audio",
                extra={"backend": backend.name, "bytes": produced},
            )
            for name in self.preference:
                if name == backend.name:
                    continue
                fallback = self.backends.get(name)
                if fallback is None:
                    continue
                try:
                    fallback_bytes = 0
                    async for chunk in fallback.synthesize(text, voice, stream=stream):
                        fallback_bytes += len(chunk.data)
                        self._last_used = fallback.name
                        yield chunk
                    if fallback_bytes > 44:
                        log.info(
                            "tts fell back after empty audio",
                            extra={"from": backend.name, "to": fallback.name, "bytes": fallback_bytes},
                        )
                        return
                except Exception as inner:  # noqa: BLE001
                    log.debug(
                        "tts fallback failed",
                        extra={"backend": name, "error": f"{type(inner).__name__}: {inner}"},
                    )
            raise AudioError(
                f"语音后端 {backend.name} 返回了空音频，且没有可用的后备后端",
                detail={"bytes": produced, "tried": list(self.preference)},
            )

        if cache_key and produced > 44:
            self._cache_put(cache_key, buffered_mime or "audio/mpeg", bytes(buffered))

    async def health(self) -> Dict[str, Any]:
        details = {}
        for name in self.backends:
            details[name] = await self._health_of(name)
        healthy = [name for name, info in details.items() if info.get("ok")]
        return {
            "ok": bool(healthy),
            "name": self.name,
            "preference": self.preference,
            "prefer_offline": self.prefer_offline,
            "healthy": healthy,
            "last_used": self._last_used,
            "backends": details,
        }

    async def aclose(self) -> None:
        for backend in self.backends.values():
            if hasattr(backend, "aclose"):
                try:
                    await backend.aclose()
                except Exception:  # noqa: BLE001
                    pass


# -- GPT-SoVITS (fine-tuned voices) ------------------------------------------------
# The implementation lives in src/plugs/gpt_sovits_tts.py (which imports the contracts
# from this module). Re-exported here so `build_tts` can import it from one place.
from plugs.gpt_sovits_tts import GPTSoVITSTTSBackend  # noqa: E402

__all__ = [
    "EdgeTTSBackend",
    "IndexTTSBackend",
    "OpenAITTSBackend",
    "SapiTTSBackend",
    "TTSRouter",
    "GPTSoVITSTTSBackend",
    "ToneTTSBackend",
]
