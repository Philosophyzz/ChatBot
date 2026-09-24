"""Backend client for the desktop pet.

The pet is a *thin client*: it owns the window, the microphone and the speaker, and
delegates everything else (LLM, long-term memory, personas, STT, TTS routing) to the
same local API the web UI uses. That is why the pet needs no model knowledge at all —
switching persona or model in the web UI changes the pet too.

Two details that matter and are easy to get wrong:

* ``trust_env=False`` on every HTTP client. This machine runs a system proxy
  (Clash on 127.0.0.1:7890) and without this flag calls to ``127.0.0.1`` get routed
  through it, coming back as HTTP 502 even though the service is perfectly healthy.
* The TTS stream is a sequence of ``[4-byte length][1-byte tag][payload]`` frames, not
  a media file. Frames arrive sentence by sentence, which is what lets the pet start
  speaking ~2s in instead of waiting for the whole answer.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8077"


@dataclass
class ChatEvent:
    kind: str                      # "text" | "done" | "error" | "memory"
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


class BackendError(RuntimeError):
    """Raised with a message meant to be shown to the user verbatim."""


class BackendClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, *, timeout_s: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        # See module docstring: without trust_env=False the system proxy breaks localhost.
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s, trust_env=False)
        #: Metadata of the most recent TTS stream. The endpoint sends a summary frame
        #: *after* the last audio frame (which engine spoke, how many frames, elapsed ms),
        #: so it cannot be attached to any audio chunk — it lands here instead.
        self.last_tts_meta: Dict[str, Any] = {}
        self._sessions: Dict[str, str] = {}

    def close(self) -> None:
        self._client.close()

    # -- diagnostics -------------------------------------------------------------------
    def health(self, *, deep: bool = False) -> Dict[str, Any]:
        try:
            response = self._client.get("/api/health", params={"deep": str(deep).lower()}, timeout=30)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise BackendError(f"连不上本地服务（{self.base_url}）：{exc}") from exc

    def online(self) -> bool:
        try:
            return bool(self.health().get("ok"))
        except BackendError:
            return False

    # -- personas ----------------------------------------------------------------------
    def personas(self) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        response = self._client.get("/api/personas", timeout=30)
        response.raise_for_status()
        body = response.json()
        return body.get("personas") or [], body.get("default")

    def update_persona(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        response = self._client.put(f"/api/personas/{persona['id']}", json=persona, timeout=30)
        if response.is_error:
            raise BackendError(response.text[:400])
        return response.json()

    # -- base model --------------------------------------------------------------------
    def models(self) -> Dict[str, Any]:
        """Available base-model tiers, which are downloaded, and what is running."""
        response = self._client.get("/api/models", timeout=30)
        response.raise_for_status()
        return response.json()

    def set_model_connection(self, values: Dict[str, Any]) -> Dict[str, Any]:
        response = self._client.put("/api/models/connection", json=values, timeout=30)
        if response.is_error:
            raise BackendError(response.text[:400])
        return response.json()

    def switch_model(self, tier_id: str) -> Dict[str, Any]:
        """Switch the base model. Takes 30–120 s (the server reloads the weights)."""
        response = self._client.post("/api/models/switch", json={"tier": tier_id}, timeout=600)
        if response.status_code >= 400:
            detail: Any
            try:
                detail = response.json().get("detail")
            except Exception:  # noqa: BLE001
                detail = response.text[:200]
            raise BackendError(str(detail or f"切换失败（HTTP {response.status_code}）"))
        return response.json()

    # -- chat --------------------------------------------------------------------------
    def chat_stream(self, text: str, *, persona_id: Optional[str] = None) -> Iterator[ChatEvent]:
        """Stream one turn. Yields text deltas, then a final ``done`` event."""
        payload: Dict[str, Any] = {"message": text, "stream": True, "voice_mode": True}
        session_key = persona_id or "default"
        if session_key in self._sessions:
            payload["session_id"] = self._sessions[session_key]
        if persona_id:
            payload["persona_id"] = persona_id
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    detail = response.read().decode("utf-8", "replace")[:300]
                    raise BackendError(f"对话失败（HTTP {response.status_code}）：{detail}")
                event_name = "message"
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    kind = data.get("kind") or event_name
                    if kind == "session" and data.get("session", {}).get("id"):
                        self._sessions[session_key] = data["session"]["id"]
                    if kind == "text":
                        yield ChatEvent("text", text=str(data.get("text") or ""), data=data)
                    elif kind == "done":
                        yield ChatEvent("done", text=str(data.get("text") or ""), data=data)
                    elif kind == "error":
                        error = data.get("error") or {}
                        raise BackendError(str(error.get("message") or "对话出错"))
                    elif kind in {"memory", "memory_skipped", "session", "thinking", "emotion"}:
                        yield ChatEvent(kind, text=str(data.get("text") or ""), data=data)
        except httpx.HTTPError as exc:
            raise BackendError(f"对话请求失败：{exc}") from exc

    # -- speech in ---------------------------------------------------------------------
    def transcribe(self, wav_bytes: bytes, *, language: str = "zh") -> str:
        files = {"audio": ("speech.wav", wav_bytes, "audio/wav")}
        data = {"language": language}
        try:
            response = self._client.post("/api/voice/stt", files=files, data=data, timeout=300)
        except httpx.HTTPError as exc:
            raise BackendError(f"语音识别请求失败：{exc}") from exc
        if response.status_code >= 400:
            raise BackendError(f"语音识别失败（HTTP {response.status_code}）：{response.text[:200]}")
        body = response.json()
        # The API nests the result: {"transcript": {"text": …}, "elapsed_ms": …}. Reading a
        # top-level "text" silently produced None, and the pet then reported "没听到声音"
        # — a parsing bug that is indistinguishable from a dead microphone.
        transcript = body.get("transcript") if isinstance(body.get("transcript"), dict) else {}
        result = body.get("result") if isinstance(body.get("result"), dict) else {}
        for candidate in (transcript.get("text"), body.get("text"), result.get("text")):
            if candidate:
                return str(candidate).strip()
        return ""

    # -- speech out --------------------------------------------------------------------
    def speak_stream(
        self, text: str, *, persona_id: Optional[str] = None
    ) -> Iterator[Tuple[bytes, Dict[str, Any]]]:
        """Yield ``(audio_bytes, meta)`` per synthesized chunk, in order.

        Frames: ``[4-byte big-endian length][1-byte tag][payload]``; tag ``b"M"`` is a
        JSON metadata frame, ``b"A"`` is audio. Metadata frames are merged and attached
        to the following audio frame (the last one carries the summary).
        """
        payload: Dict[str, Any] = {"text": text}
        if persona_id:
            payload["persona_id"] = persona_id
        meta: Dict[str, Any] = {}
        try:
            with self._client.stream("POST", "/api/voice/tts/stream", json=payload) as response:
                if response.status_code >= 400:
                    detail = response.read().decode("utf-8", "replace")[:300]
                    raise BackendError(f"语音合成失败（HTTP {response.status_code}）：{detail}")
                buffer = b""
                for chunk in response.iter_bytes():
                    buffer += chunk
                    while len(buffer) >= 5:
                        length = struct.unpack(">I", buffer[:4])[0]
                        if len(buffer) < 5 + length:
                            break
                        tag = buffer[4:5]
                        body = buffer[5 : 5 + length]
                        buffer = buffer[5 + length :]
                        if tag == b"A" and body:
                            yield body, dict(meta)
                        elif tag == b"M":
                            try:
                                update = json.loads(body.decode("utf-8"))
                            except json.JSONDecodeError:
                                continue
                            if update.get("error"):
                                raise BackendError(str(update["error"]))
                            meta.update(update)
        except httpx.HTTPError as exc:
            raise BackendError(f"语音合成请求失败：{exc}") from exc
        # Reached only on a complete stream: keep the trailing summary frame.
        self.last_tts_meta = dict(meta)

    def speak(self, text: str, *, persona_id: Optional[str] = None) -> bytes:
        """Whole clip at once — used by the fallback path and by tests."""
        return b"".join(chunk for chunk, _ in self.speak_stream(text, persona_id=persona_id))

    def tts_backend(self) -> str:
        """Which engine is speaking (edge-tts / IndexTTS2 / SAPI…), for the status line."""
        try:
            info = self.health(deep=True).get("tts") or {}
        except BackendError:
            return "?"
        return str(info.get("last_used") or (info.get("healthy") or ["?"])[0])
