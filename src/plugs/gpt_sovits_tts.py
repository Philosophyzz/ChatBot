"""GPT-SoVITS TTS backend — speaks with a voice you fine-tuned yourself.

Why this exists next to the IndexTTS2 backend: they solve different problems.

* **IndexTTS2** = zero-shot cloning. No training, but the timbre is only as good as the
  5–15s reference clip, and it lives in this process's GPU budget.
* **GPT-SoVITS** = fine-tuned. You feed it 10–60 minutes of clean speech and it learns
  that speaker; similarity and stability are noticeably better, and inference is cheap
  (RTF ~0.03 on a 4060Ti per its README) because it runs as a small separate server.

The training pipeline lives in ``scripts/train-tts.ps1``; this file only talks to the
inference server that training produces (``api_v2.py``, default port 9090).

Protocol (GPT-SoVITS ``api_v2``)::

    POST /tts  {"text", "text_lang", "ref_audio_path", "prompt_text", "prompt_lang", ...}
    -> audio/wav bytes

The per-persona voice is configured exactly like the other backends::

    # config/personas.yaml
    my_voice:
      voice:
        backend: gpt_sovits
        reference_audio: models/voices/my_voice_ref.wav
        reference_text: "参考音频里说的那句话"
        emotion: neutral
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import urlparse

from core.errors import AudioError, BackendUnavailable
from core.logging import get_logger
from core.registry import KIND_TTS, register
from core.types import SpeechChunk, VoiceSpec

log = get_logger(__name__)


def _port_open(host: str, port: int, timeout_s: float = 1.5) -> bool:
    """Cheap liveness probe — api_v2 has no /health endpoint, so a TCP connect is it."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


@register(KIND_TTS, "gpt_sovits")
class GPTSoVITSTTSBackend:
    """HTTP client for a GPT-SoVITS inference server (fine-tuned voices)."""

    name = "gpt_sovits"
    supports_cloning = True
    offline = True  # the server runs locally; no third-party service is involved

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:9090",
        text_lang: str = "zh",
        prompt_lang: str = "zh",
        media_type: str = "wav",
        split_method: str = "cut5",
        timeout_s: float = 120.0,
        speed: float = 1.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.text_lang = text_lang
        self.prompt_lang = prompt_lang
        self.media_type = media_type
        self.split_method = split_method
        self.timeout_s = float(timeout_s)
        self.speed = float(speed)
        self.extra = dict(extra or {})
        self._last_used = 0.0
        parsed = urlparse(self.base_url)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)

    # -- health ------------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        alive = await asyncio.to_thread(_port_open, self._host, self._port)
        info: Dict[str, Any] = {
            "ok": bool(alive),
            "name": self.name,
            "offline": True,
            "supports_cloning": True,
            "base_url": self.base_url,
            "hint": "启动推理服务：scripts\\train-tts.ps1 -Serve",
        }
        if not alive:
            info["error"] = f"GPT-SoVITS 推理服务未运行（{self.base_url}）"
        return info

    # -- synthesis ---------------------------------------------------------------------
    async def synthesize(
        self, text: str, voice: VoiceSpec, *, stream: bool = False
    ) -> AsyncIterator[SpeechChunk]:
        text = (text or "").strip()
        if not text:
            return
        reference = voice.reference_audio
        if not reference:
            raise AudioError(
                "该人设没有配置参考音频（voice.reference_audio）。GPT-SoVITS 推理必须给一段"
                "参考音频与其文本（微调后的模型同样需要），请在 config/personas.yaml 里补上。"
            )
        prompt_text = voice.reference_text or ""
        if not prompt_text:
            log.warning(
                "gpt-sovits prompt_text is empty",
                extra={"reference": reference, "hint": "留空会导致韵律变差，建议填写参考音频对应的文字"},
            )

        payload: Dict[str, Any] = {
            "text": text,
            "text_lang": self.text_lang,
            "ref_audio_path": str(reference),
            "prompt_text": prompt_text,
            "prompt_lang": self.prompt_lang,
            "text_split_method": self.split_method,
            "media_type": self.media_type,
            "streaming_mode": False,
            "speed_factor": float(voice.speed or self.speed or 1.0),
        }
        payload.update(self.extra)

        started = time.perf_counter()
        try:
            import httpx
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"缺少 httpx：{exc}") from exc

        # trust_env=False: this machine has a system proxy, and routing 127.0.0.1 through
        # it turns perfectly healthy local calls into HTTP 502.
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout_s, trust_env=False
        ) as client:
            try:
                response = await client.post("/tts", json=payload)
            except Exception as exc:  # noqa: BLE001
                raise AudioError(
                    f"连不上 GPT-SoVITS 推理服务（{self.base_url}）：{type(exc).__name__}: {exc}"
                ) from exc
            if response.status_code >= 400:
                detail = response.text[:300]
                raise AudioError(f"GPT-SoVITS 合成失败（HTTP {response.status_code}）：{detail}")
            audio = response.content

        if not audio or len(audio) <= 44:
            raise AudioError("GPT-SoVITS 返回了空音频")
        self._last_used = time.time()
        log.debug(
            "gpt-sovits synthesized",
            extra={
                "chars": len(text),
                "bytes": len(audio),
                "ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        mime = "audio/wav" if self.media_type == "wav" else f"audio/{self.media_type}"
        yield SpeechChunk(data=audio, mime=mime, index=1, is_final=True, text=text)
