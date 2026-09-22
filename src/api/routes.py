"""HTTP and WebSocket routes.

Surface map:

* ``/api/health``, ``/api/config``, ``/api/plugins`` — diagnostics and the live
  registry (the plugin list is generated from what is actually registered, so the
  UI can never advertise a capability the process does not have).
* ``/api/personas*`` — persona CRUD, including voice settings.
* ``/api/sessions*`` — conversation lifecycle and history.
* ``/api/chat`` and ``/api/ws/chat`` — streaming chat over SSE and WebSocket.
* ``/api/voice/stt``, ``/api/voice/tts`` and ``/api/voice/tts/stream`` — speech in
  and out.
* ``/api/memory*`` — inspect, correct, search and consolidate long-term memory.
  The user being able to *see and fix* what the assistant remembers is what keeps
  a memory system trustworthy over months.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import traceback
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse

from api.models import (
    ChatRequest,
    MemoryCreateRequest,
    MemoryUpdateRequest,
    PersonaUpsertRequest,
    ProfileUpsertRequest,
    SessionCreateRequest,
    SessionRenameRequest,
    SettingsPatch,
    TTSRequest,
)
from core.errors import BadRequest, ChatBotError, NotFound
from core.logging import get_logger
from core.types import MemoryKind, Message, Role, VoiceSpec
from core.utils import speakable_text, split_sentences

log = get_logger(__name__)

router = APIRouter(prefix="/api")


def _app(request: Request) -> Any:
    app = getattr(request.app.state, "chatbot", None)
    if app is None:  # pragma: no cover - misconfiguration, not a user error
        raise HTTPException(status_code=503, detail="应用未初始化")
    return app


def _error_response(exc: ChatBotError) -> JSONResponse:
    return JSONResponse(status_code=exc.http_status, content=exc.to_public())


def _sse(payload: Dict[str, Any], *, event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ======================================================================================
# Diagnostics
# ======================================================================================


@router.get("/health")
async def health(request: Request, deep: bool = False) -> Dict[str, Any]:
    return await _app(request).health(deep=deep)


@router.get("/models")
async def list_models(request: Request) -> Dict[str, Any]:
    """Base-model tiers: which are downloaded, and what is serving right now.

    The web UI and the pet tray both render this, so it carries everything the user needs to
    decide — size, context, expected speed, the honest trade-off note, and, for a tier that is
    not downloaded yet, the exact command that fetches it.
    """
    import asyncio

    supervisor = getattr(_app(request), "models", None)
    if supervisor is None:
        raise HTTPException(status_code=503, detail="模型管理器不可用")
    return await asyncio.to_thread(supervisor.status)


@router.post("/models/switch")
async def switch_model(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Switch the base chat model: stop the old server, start the chosen one.

    Loading a 10 GB model takes 30–120 s, so this call blocks until the new server answers
    ``/health``; the UI shows progress by polling ``GET /api/models``. Only the chat server on
    :8080 is restarted — the embedding/reranker servers keep running, so long-term memory does
    not degrade into hash embeddings while the user experiments.
    """
    from llm.supervisor import ModelSwitchError

    app = _app(request)
    if getattr(app, "models", None) is None:
        raise HTTPException(status_code=503, detail="模型管理器不可用")
    tier = str((payload or {}).get("tier") or (payload or {}).get("id") or "").strip()
    if not tier:
        raise HTTPException(status_code=400, detail="缺少参数 tier")
    try:
        return await app.switch_model(tier)
    except ModelSwitchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - message already carries the log tail
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.post("/shutdown")
async def shutdown(request: Request) -> Dict[str, Any]:
    """Ask the process to exit gracefully. Localhost only.

    Why this exists: Windows has no usable SIGTERM — ``Stop-Process -Force`` kills the
    interpreter outright, so the SQLite connections are never closed, the WAL is never
    checkpointed (a multi-megabyte ``memory.sqlite3-wal`` accumulates) and the database
    file can stay locked after "stopping" the service. ``scripts/stop.ps1`` calls this
    endpoint first and only force-kills as a fallback.

    The host check matters because the server can be bound to the LAN for phone access;
    a remote caller must not be able to take the assistant down.
    """
    host = (request.client.host if request.client else "") or ""
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="仅允许本机调用")

    server = getattr(request.app.state, "uvicorn_server", None)
    if server is None:
        return {"ok": False, "detail": "服务不是以可停止模式启动的"}

    # Let the response reach the client before uvicorn starts tearing down.
    asyncio.get_running_loop().call_later(0.25, setattr, server, "should_exit", True)
    log.info("graceful shutdown requested")
    return {"ok": True}


@router.get("/config")
async def get_config(request: Request) -> Dict[str, Any]:
    return _app(request).config.to_public()


@router.patch("/config")
async def patch_config(request: Request, patch: SettingsPatch) -> Dict[str, Any]:
    """Apply a runtime settings change and persist it to ``config/local.yaml``.

    Only a safe subset is mutable here: sampling, retrieval weights and debug
    flags change behaviour without a restart. Model/server endpoints are
    deliberately *not* hot-swappable — they would require reloading a 15 GB model,
    which belongs in the start script, not in a request handler.
    """
    app = _app(request)
    applied: Dict[str, Any] = {}
    if patch.debug is not None:
        app.config.debug = bool(patch.debug)
        applied["debug"] = app.config.debug
    if patch.default_persona is not None:
        try:
            app.personas.get(patch.default_persona, strict=True)
        except NotFound as exc:
            return _error_response(exc)
        app.config.default_persona = patch.default_persona
        applied["default_persona"] = patch.default_persona
    for section_name, values in (("llm", patch.llm), ("memory", patch.memory), ("speech", patch.speech)):
        if not values:
            continue
        section = getattr(app.config, section_name)
        for key, value in values.items():
            if hasattr(section, key):
                setattr(section, key, value)
                applied[f"{section_name}.{key}"] = value
    if applied:
        _persist_overrides(app, applied)
    return {"applied": applied, "config": app.config.to_public()}


def _persist_overrides(app: Any, applied: Dict[str, Any]) -> None:
    """Write changed settings to config/local.yaml so they survive a restart."""
    path = app.config.paths.config_dir / "local.yaml"
    nested: Dict[str, Dict[str, Any]] = {}
    for key, value in applied.items():
        if "." in key:
            section, _, field = key.partition(".")
            nested.setdefault(section, {})[field] = value
        else:
            nested.setdefault("_root", {})[key] = value
    lines: List[str] = ["# 由设置界面写入的运行时覆盖；优先级高于 config.yaml\n", "# 该文件由程序维护，手改前请先停止服务。\n"]
    for section, fields in nested.items():
        if section == "_root":
            for key, value in fields.items():
                lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
        else:
            lines.append(f"{section}:\n")
            for key, value in fields.items():
                lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False)}\n")
    try:
        path.write_text("".join(lines), encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        log.warning("could not persist settings overrides", extra={"error": str(exc)})


@router.get("/plugins")
async def list_plugins(request: Request) -> Dict[str, Any]:
    app = _app(request)
    return {
        "registry": app.registry.snapshot(),
        "constants": {
            "llm": ["llamacpp", "ollama", "vllm", "openai", "mock"],
            "tts": ["router", "indextts", "edge", "openai", "sapi", "tone"],
            "stt": ["faster_whisper", "mock"],
            "vector_store": ["sqlite", "memory"],
            "reranker": ["llamacpp", "heuristic", "none"],
            "embedder": ["llamacpp", "hash"],
        },
    }


# ======================================================================================
# Personas
# ======================================================================================


@router.get("/personas")
async def list_personas(request: Request) -> Dict[str, Any]:
    app = _app(request)
    default = app.config.default_persona
    return {
        "personas": [persona.to_public() for persona in app.personas.list()],
        # Sent explicitly so the UI can badge/highlight the default instead of guessing
        # from list order, which changes as personas are added.
        "default": default,
    }


@router.get("/personas/{persona_id}")
async def get_persona(request: Request, persona_id: str) -> Dict[str, Any]:
    try:
        return _app(request).personas.get(persona_id, strict=True).to_public()
    except NotFound as exc:
        return _error_response(exc)


@router.put("/personas/{persona_id}")
async def upsert_persona(request: Request, persona_id: str, payload: PersonaUpsertRequest) -> Dict[str, Any]:
    app = _app(request)
    try:
        persona = app.personas.upsert(persona_id, payload.model_dump())
    except BadRequest as exc:
        return _error_response(exc)
    return {"persona": persona.to_public()}


@router.delete("/personas/{persona_id}")
async def delete_persona(request: Request, persona_id: str) -> Dict[str, Any]:
    app = _app(request)
    try:
        removed = app.personas.delete(persona_id)
    except (BadRequest, NotFound) as exc:
        return _error_response(exc)
    return {"deleted": removed, "persona_id": persona_id}


# ======================================================================================
# Sessions
# ======================================================================================


@router.get("/sessions")
async def list_sessions(request: Request, scope: str = "", limit: int = 50) -> Dict[str, Any]:
    app = _app(request)
    return {"sessions": await app.sessions.list(scope=scope, limit=limit)}


@router.post("/sessions")
async def create_session(request: Request, payload: SessionCreateRequest) -> Dict[str, Any]:
    app = _app(request)
    try:
        persona = app.personas.get(payload.persona_id, strict=True)
    except NotFound as exc:
        return _error_response(exc)
    session = await app.sessions.create(
        persona_id=persona.id,
        scope=payload.scope,
        title=payload.title,
        voice_mode=payload.voice_mode,
    )
    return {"session": session.public_summary(), "greeting": persona.greeting}


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str, limit: int = 100) -> Dict[str, Any]:
    app = _app(request)
    try:
        session = await app.sessions.get(session_id)
    except NotFound as exc:
        return _error_response(exc)
    return {"session": session.to_public(limit=limit)}


@router.patch("/sessions/{session_id}")
async def rename_session(request: Request, session_id: str, payload: SessionRenameRequest) -> Dict[str, Any]:
    app = _app(request)
    try:
        await app.sessions.rename(session_id, payload.title)
    except NotFound as exc:
        return _error_response(exc)
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> Dict[str, Any]:
    app = _app(request)
    try:
        await app.sessions.delete(session_id)
    except NotFound as exc:
        return _error_response(exc)
    return {"ok": True}


# ======================================================================================
# Chat
# ======================================================================================


async def _turn_events(app: Any, payload: ChatRequest, request: Optional[Request] = None) -> AsyncIterator[Dict[str, Any]]:
    """Run one turn under the GPU serialisation semaphore.

    One GPU, one model: concurrent generations would thrash VRAM and make both
    answers slower, so turns queue here instead of racing.
    """
    lock = app.chat_lock()
    if lock is None:  # pragma: no cover - only before startup
        async for event in app.engine.stream_turn(
            session_id=payload.session_id,
            user_text=payload.message,
            persona_id=payload.persona_id,
            voice_mode=payload.voice_mode,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            extra_instructions=payload.extra_instructions,
        ):
            yield event
        return

    async with lock:
        async for event in app.engine.stream_turn(
            session_id=payload.session_id,
            user_text=payload.message,
            persona_id=payload.persona_id,
            voice_mode=payload.voice_mode,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            extra_instructions=payload.extra_instructions,
        ):
            yield event
            if request is not None and await request.is_disconnected():
                log.info("client disconnected, aborting turn")
                return


@router.post("/chat")
async def chat(request: Request, payload: ChatRequest) -> Response:
    app = _app(request)
    if not payload.stream:
        try:
            result = await app.engine.complete_turn(
                session_id=payload.session_id,
                user_text=payload.message,
                persona_id=payload.persona_id,
                voice_mode=payload.voice_mode,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
                extra_instructions=payload.extra_instructions,
            )
        except ChatBotError as exc:
            return _error_response(exc)
        body: Dict[str, Any] = {
            "text": result.text,
            "thinking": result.thinking,
            "stats": result.stats.to_public(),
            "message_id": getattr(result.message, "id", None),
        }
        if payload.speak and result.text:
            body["audio"] = await _inline_audio(app, result.text, payload.persona_id)
        return JSONResponse(content=body)

    async def generator() -> AsyncIterator[str]:
        try:
            async for event in _turn_events(app, payload, request):
                yield _sse(event, event=event.get("kind", "message"))
                if payload.speak and event.get("kind") == "done" and event.get("text"):
                    audio = await _inline_audio(app, event["text"], payload.persona_id)
                    if audio:
                        yield _sse({"kind": "audio", **audio}, event="audio")
        except ChatBotError as exc:
            yield _sse({"kind": "error", **exc.to_public()}, event="error")
        except asyncio.CancelledError:  # client went away
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("chat stream failed")
            yield _sse({"kind": "error", "error": {"code": "internal", "message": str(exc)}}, event="error")

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


async def _inline_audio(app: Any, text: str, persona_id: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        chunks = [chunk async for chunk in app.engine.synthesize_speech(text, persona_id=persona_id, stream=False)]
    except ChatBotError as exc:
        log.warning("inline speech synthesis failed", extra={"error": exc.message})
        return None
    if not chunks:
        return None
    audio = chunks[0]
    return {"data_b64": base64.b64encode(audio.data).decode("ascii"), "mime": audio.mime}


@router.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket) -> None:
    """Bidirectional chat channel — the primary transport for the web UI.

    Protocol: the client sends JSON commands (``chat``, ``stop``, ``ping``); the
    server replies with the same event objects the SSE endpoint emits, so the two
    transports share one client-side renderer.
    """
    app = getattr(websocket.app.state, "chatbot", None)
    if app is None:  # pragma: no cover
        await websocket.close(code=1011, reason="app not ready")
        return
    await websocket.accept()
    current: Optional[asyncio.Task] = None
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                command = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"kind": "error", "error": {"code": "bad_request", "message": "无效的 JSON"}})
                continue
            action = str(command.get("action") or "chat")
            if action == "ping":
                await websocket.send_json({"kind": "pong", "ts": time.time()})
                continue
            if action == "stop":
                if current is not None and not current.done():
                    current.cancel()
                await websocket.send_json({"kind": "stopped"})
                continue
            if action != "chat":
                await websocket.send_json(
                    {"kind": "error", "error": {"code": "bad_request", "message": f"未知指令：{action}"}}
                )
                continue

            # A new chat command supersedes an in-flight turn.
            if current is not None and not current.done():
                current.cancel()

            try:
                payload = ChatRequest(**command)
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json(
                    {"kind": "error", "error": {"code": "bad_request", "message": str(exc)[:300]}}
                )
                continue

            async def run(turn_payload: ChatRequest = payload) -> None:
                try:
                    async for event in _turn_events(app, turn_payload):
                        await websocket.send_json(event)
                        if turn_payload.speak and event.get("kind") == "done" and event.get("text"):
                            audio = await _inline_audio(app, event["text"], turn_payload.persona_id)
                            if audio:
                                await websocket.send_json({"kind": "audio", **audio})
                except asyncio.CancelledError:
                    await _safe_send(websocket, {"kind": "stopped"})
                    raise
                except ChatBotError as exc:
                    await _safe_send(websocket, {"kind": "error", **exc.to_public()})
                except Exception as exc:  # noqa: BLE001
                    log.exception("ws turn failed")
                    await _safe_send(websocket, {"kind": "error", "error": {"code": "internal", "message": str(exc)}})

            current = asyncio.create_task(run())
    except WebSocketDisconnect:
        log.info("websocket disconnected")
    finally:
        if current is not None and not current.done():
            current.cancel()


async def _safe_send(websocket: WebSocket, payload: Dict[str, Any]) -> None:
    try:
        await websocket.send_json(payload)
    except Exception:  # noqa: BLE001 - the peer may already be gone
        pass


# ======================================================================================
# Voice
# ======================================================================================


@router.post("/voice/stt")
async def speech_to_text(
    request: Request,
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
) -> Dict[str, Any]:
    app = _app(request)
    if app.engine is None or app.engine.stt is None:
        return _error_response(
            ChatBotError("语音识别未启用（speech.stt_backend 配置或依赖缺失）", detail={"hint": "pip install faster-whisper"})
        )
    data = await audio.read()
    if not data:
        return _error_response(BadRequest("音频为空"))
    if len(data) > app.config.server.max_audio_bytes:
        return _error_response(
            BadRequest(
                f"音频过大（{len(data) // 1024} KB > {app.config.server.max_audio_bytes // 1024} KB）"
            )
        )
    from speech.audio import normalize_clip

    started = time.perf_counter()
    try:
        clip = await asyncio.to_thread(
            normalize_clip, data, vad_trim=app.config.speech.vad_filter
        )
        transcript = await app.engine.stt.transcribe(
            clip, language=language or app.config.speech.stt_language or None, prompt=prompt
        )
    except ChatBotError as exc:
        return _error_response(exc)
    except Exception as exc:  # noqa: BLE001
        log.exception("stt failed")
        return _error_response(ChatBotError(f"语音识别失败：{exc}"))
    return {
        "transcript": transcript.to_public(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@router.post("/voice/tts")
async def text_to_speech(request: Request, payload: TTSRequest) -> Response:
    app = _app(request)
    if app.engine is None or app.engine.tts is None:
        return _error_response(ChatBotError("语音合成未启用"))
    voice, _persona_id = _resolve_voice(app, payload)

    text = speakable_text(payload.text)
    try:
        chunks = [chunk async for chunk in app.engine.tts.synthesize(text, voice, stream=payload.stream)]
    except ChatBotError as exc:
        return _error_response(exc)
    if not chunks:
        return _error_response(BadRequest("没有可合成的文本"))

    merged = b"".join(chunk.data for chunk in chunks)
    if not merged or len(merged) <= 44:
        # A header-only WAV (or nothing at all) is silence. Returning it as 200 makes
        # the UI play nothing with no explanation, so fail loudly instead.
        return _error_response(
            ChatBotError(
                "语音合成为空（后端返回了无音频数据的容器）",
                detail={"bytes": len(merged), "backend": str(getattr(app.engine.tts, "_last_used", ""))},
            )
        )

    # Always return a single, fully-buffered response with a real Content-Length.
    # This used to stream chunked, which broke ordinary HTTP clients: Windows
    # PowerShell's Invoke-WebRequest cannot read a chunked body and reports a bogus
    # 46-byte result, which made a perfectly good audio response look like a failure
    # and sent debugging in the wrong direction for a long time.
    return Response(
        content=merged,
        media_type=chunks[0].mime,
        headers={
            "Cache-Control": "no-store",
            "X-TTS-Backend": str(getattr(app.engine.tts, "_last_used", "")),
            "X-TTS-Chunks": str(len(chunks)),
        },
    )


@router.post("/voice/tts/stream")
async def text_to_speech_stream(request: Request, payload: TTSRequest) -> Response:
    """Sentence-by-sentence synthesis, delivered as it is produced.

    Why this exists: with an online backend (edge-tts) one request for a whole reply
    costs 1.5–5s *per sentence* — a three-sentence answer made the user wait ~12s in
    silence before a single word was spoken. Speech is naturally sequential, so the fix
    is to synthesize sentence 1, send it immediately, and keep going while the browser
    is already playing.

    Framing (deliberately dumb, so the client needs no parser library)::

        [4-byte big-endian length][1-byte tag][payload]

    ``tag = b"A"`` audio bytes, ``tag = b"M"`` a JSON metadata frame (first frame:
    requested voice/persona; last frame: which backend actually spoke, frame count and
    total milliseconds). Unknown tags are ignored by the client, which keeps this
    extensible without breaking older pages.
    """
    app = _app(request)
    if app.engine is None or app.engine.tts is None:
        return _error_response(ChatBotError("语音合成未启用"))

    voice, persona_id = _resolve_voice(app, payload)
    text = speakable_text(payload.text)
    # Small chunks, not whole sentences: the budget is the user's time-to-first-word.
    # ``split_sentences`` merges up to ``max_chars``, so passing the TTS budget (default
    # 40) keeps the first chunk to a couple of seconds of speech.
    chunk_chars = max(16, int(getattr(app.config.speech, "tts_chunk_chars", 40) or 40))
    concurrency = max(1, int(getattr(app.config.speech, "tts_chunk_concurrency", 2) or 1))
    blocks = [b for b in split_sentences(text, max_chars=chunk_chars) if b.strip()]
    if not blocks:
        return _error_response(BadRequest("没有可合成的文本"))

    async def frames() -> AsyncIterator[bytes]:
        started = time.perf_counter()
        yield _frame(
            b"M",
            {
                "persona_id": persona_id,
                "voice_id": voice.voice_id,
                "chunks": len(blocks),
                "concurrency": concurrency,
                "preference": list(getattr(app.engine.tts, "preference", [])),
            },
        )

        # Start every chunk immediately but let at most ``concurrency`` run at once, then
        # emit strictly in order. Chunk 1 therefore leaves as soon as it is ready while
        # chunk 2 is already being synthesized — the browser starts speaking without
        # waiting for the tail, and the total time drops as well.
        gate = asyncio.Semaphore(concurrency)

        async def synth(block: str) -> bytes:
            async with gate:
                data = bytearray()
                async for chunk in app.engine.tts.synthesize(block, voice, stream=False):
                    data.extend(chunk.data)
                return bytes(data)

        tasks = [asyncio.create_task(synth(block)) for block in blocks]
        index = 0
        try:
            for position, task in enumerate(tasks):
                try:
                    data = await task
                except ChatBotError as exc:
                    # The response headers are long gone, so a failure has to be reported
                    # inside the stream; silently stopping would look like a normal end.
                    yield _frame(b"M", {"error": exc.message, "detail": exc.detail, "at": position})
                    break
                if not data:
                    continue
                index += 1
                yield _frame(b"A", data)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
        yield _frame(
            b"M",
            {
                "backend": str(getattr(app.engine.tts, "_last_used", "")),
                "frames": index,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "cache": getattr(app.engine.tts, "cache_stats", lambda: {})(),
            },
        )

    return StreamingResponse(
        frames(),
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def _frame(tag: bytes, payload: bytes | Dict[str, Any]) -> bytes:
    """One length-prefixed frame: 4-byte big-endian length, 1-byte tag, payload."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if isinstance(payload, dict) else payload
    return len(body).to_bytes(4, "big") + tag + body


def _resolve_voice(app: Any, payload: TTSRequest) -> tuple[VoiceSpec, Optional[str]]:
    """Persona voice with optional per-request overrides. Shared by both TTS routes."""
    persona = app.personas.get(payload.persona_id) if payload.persona_id else app.personas.get(None)
    voice = persona.voice
    if payload.voice:
        merged = voice.to_public()
        merged.update(payload.voice)
        return (
            VoiceSpec(
                backend=payload.backend or merged.get("backend"),
                voice_id=str(merged.get("voice_id") or "default"),
                reference_audio=merged.get("reference_audio"),
                reference_text=merged.get("reference_text"),
                speed=float(merged.get("speed") or 1.0),
                emotion=str(merged.get("emotion") or "neutral"),
                emotion_alpha=float(merged.get("emotion_alpha") or 0.8),
                pitch_shift=float(merged.get("pitch_shift") or 0.0),
                style=dict(merged.get("style") or {}),
            ),
            persona.id,
        )
    if payload.backend:
        return (
            VoiceSpec(
                backend=payload.backend,
                voice_id=voice.voice_id,
                reference_audio=voice.reference_audio,
                reference_text=voice.reference_text,
                speed=voice.speed,
                emotion=voice.emotion,
                emotion_alpha=voice.emotion_alpha,
                pitch_shift=voice.pitch_shift,
                style=dict(voice.style),
            ),
            persona.id,
        )
    return voice, persona.id


@router.get("/voice/tts/diagnose")
async def diagnose_tts(request: Request, text: str = "你好呀，我是小甜。") -> Dict[str, Any]:
    """Exercise every TTS backend in-process and report the exact error per backend.

    Speech backends fail for environmental reasons — no network, a missing native
    library, a voice that does not exist — and the router deliberately swallows those
    in order to fall back. That makes "the voice sounds robotic" the only visible
    symptom while the real cause stays buried in a log line. This endpoint surfaces the
    full traceback so the cause is one request away.

    ``repeats`` synthesizes the same text several times in a row, which is how
    intermittent provider throttling ("NoAudioReceived") becomes visible instead of
    looking like a broken installation.
    """
    app = _app(request)
    if app.tts is None:
        return _error_response(ChatBotError("语音合成未启用"))

    persona = app.personas.get(None)
    repeats = 1
    try:
        repeats = max(1, min(5, int(request.query_params.get("repeats", 1))))
    except (TypeError, ValueError):
        repeats = 1

    results: List[Dict[str, Any]] = []
    backends = getattr(app.tts, "backends", None)
    if not isinstance(backends, dict):
        backends = {getattr(app.tts, "name", "tts"): app.tts}

    for name, backend in backends.items():
        entry: Dict[str, Any] = {"backend": name, "attempts": []}
        try:
            entry["health"] = await backend.health()
        except Exception as exc:  # noqa: BLE001
            entry["health"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if not entry["health"].get("ok"):
            entry["skipped"] = "健康检查未通过"
            results.append(entry)
            continue
        for index in range(repeats):
            attempt: Dict[str, Any] = {"run": index + 1}
            started = time.perf_counter()
            try:
                chunks = [chunk async for chunk in backend.synthesize(text, persona.voice, stream=False)]
                data = b"".join(chunk.data for chunk in chunks)
                attempt["bytes"] = len(data)
                attempt["mime"] = chunks[0].mime if chunks else None
                attempt["ok"] = len(data) > 44
                if len(data) <= 44:
                    attempt["error"] = f"空音频（仅 {len(data)} 字节）"
            except Exception as exc:  # noqa: BLE001
                attempt["ok"] = False
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                attempt["traceback"] = traceback.format_exc()[-1200:]
            attempt["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            entry["attempts"].append(attempt)
        entry["ok"] = any(item.get("ok") for item in entry["attempts"])
        results.append(entry)

    return {
        "preference": getattr(app.tts, "preference", []),
        "priorities": getattr(app.tts, "priorities", {}),
        "prefer_offline": getattr(app.tts, "prefer_offline", None),
        "last_used": getattr(app.tts, "_last_used", None),
        "persona": persona.id,
        "voice": persona.voice.to_public(),
        "results": results,
    }


@router.get("/voice/tts/voices")
async def list_tts_voices(request: Request, locale: str = "zh") -> Dict[str, Any]:
    """Enumerate available voices for backends that can list them."""
    app = _app(request)
    out: Dict[str, Any] = {}
    backends = getattr(app.tts, "backends", {}) if app.tts is not None else {}
    edge = backends.get("edge") if isinstance(backends, dict) else None
    if edge is not None and hasattr(edge, "list_voices"):
        try:
            voices = await edge.list_voices()
            out["edge"] = [voice for voice in voices if voice.get("locale", "").startswith(locale)] or voices
        except Exception as exc:  # noqa: BLE001
            out["edge_error"] = str(exc)[:200]
    out["personas"] = [
        {"persona_id": persona.id, "name": persona.name, "voice": persona.voice.to_public()}
        for persona in app.personas.list()
    ]
    return out


# ======================================================================================
# Memory
# ======================================================================================


def _require_memory(app: Any) -> Any:
    if app.memory is None:
        raise HTTPException(status_code=503, detail="记忆子系统未启用")
    return app.memory


@router.get("/memory/stats")
async def memory_stats(request: Request) -> Dict[str, Any]:
    app = _app(request)
    memory = _require_memory(app)
    return await memory.health()


@router.get("/memory/items")
async def list_memories(
    request: Request,
    scope: str = "",
    kind: str = "",
    include_retired: bool = False,
    min_importance: float = 0.0,
    limit: int = 100,
    offset: int = 0,
    order: str = "recent",
) -> Dict[str, Any]:
    memory = _require_memory(_app(request))
    kinds = [kind] if kind else []
    items = await memory.store.list_memories(
        scope=scope,
        kinds=kinds,
        include_retired=include_retired,
        min_importance=min_importance,
        limit=min(500, max(1, limit)),
        offset=max(0, offset),
        order=order,
    )
    return {"items": [item.to_public() for item in items], "count": len(items)}


@router.get("/memory/items/{memory_id}")
async def get_memory(request: Request, memory_id: str) -> Dict[str, Any]:
    """Fetch a single memory (used by the memory panel and by external tooling)."""
    memory = _require_memory(_app(request))
    item = await memory.store.get_memory(memory_id)
    if item is None:
        return _error_response(NotFound(f"记忆不存在：{memory_id}"))
    return {"item": item.to_public()}


@router.post("/memory/items")
async def create_memory(request: Request, payload: MemoryCreateRequest) -> Dict[str, Any]:
    memory = _require_memory(_app(request))
    kind = payload.kind if payload.kind in {k.value for k in MemoryKind} else MemoryKind.FACT.value
    memory_id = await memory.store.add_memory(
        content=payload.content,
        kind=kind,
        subject=payload.subject,
        predicate=payload.predicate,
        object=payload.object,
        importance=payload.importance,
        confidence=payload.confidence,
        scope=payload.scope,
        tags=payload.tags,
        meta={"origin": "manual"},
    )
    if memory_id is None:
        return {"created": False, "reason": "duplicate"}
    if memory.embedder is not None and memory.vectors is not None:
        try:
            vector = await memory.embedder.embed_one(payload.content)
            if vector:
                await memory.vectors.upsert([memory_id], [vector], [{"kind": kind}])
        except Exception as exc:  # noqa: BLE001
            log.debug("manual memory embedding failed", extra={"error": str(exc)})
    return {"created": True, "id": memory_id}


@router.patch("/memory/items/{memory_id}")
async def update_memory(request: Request, memory_id: str, payload: MemoryUpdateRequest) -> Dict[str, Any]:
    memory = _require_memory(_app(request))
    updated = await memory.update_memory(
        memory_id,
        content=payload.content,
        importance=payload.importance,
        confidence=payload.confidence,
        tags=payload.tags,
    )
    if not updated:
        return _error_response(NotFound(f"记忆不存在：{memory_id}"))
    if payload.content and memory.embedder is not None and memory.vectors is not None:
        try:
            vector = await memory.embedder.embed_one(payload.content)
            if vector:
                await memory.vectors.upsert([memory_id], [vector], [{"kind": "fact"}])
        except Exception:  # noqa: BLE001
            pass
    return {"updated": True}


@router.delete("/memory/items/{memory_id}")
async def delete_memory(request: Request, memory_id: str, hard: bool = False) -> Dict[str, Any]:
    memory = _require_memory(_app(request))
    exists = await memory.store.get_memory(memory_id)
    if exists is None:
        return _error_response(NotFound(f"记忆不存在：{memory_id}"))
    await memory.forget(memory_id, hard=hard)
    return {"deleted": True, "hard": hard}


@router.get("/memory/search")
async def search_memory(
    request: Request,
    q: str,
    scope: str = "",
    top_k: int = 10,
    token_budget: int = 1200,
    include_history: bool = True,
    limit: int = 20,
) -> Dict[str, Any]:
    """Unified search: structured memory + (optionally) raw chat history."""
    memory = _require_memory(_app(request))
    if not q.strip():
        return _error_response(BadRequest("查询内容不能为空"))
    from core.types import RetrievalQuery

    result = await memory.retriever.retrieve(
        RetrievalQuery(text=q, session_id="search", scope=scope, top_k=top_k, token_budget=token_budget)
    )
    payload: Dict[str, Any] = {
        "memories": [hit.to_public() for hit in result.hits],
        "rendered": result.rendered,
        "tokens_used": result.tokens_used,
        "debug": result.debug,
    }
    if include_history:
        payload["history"] = await memory.search_history(q, scope=scope, limit=limit)
    return payload


@router.get("/memory/history")
async def search_history(
    request: Request, q: str, scope: str = "", conversation_id: str = "", limit: int = 20
) -> Dict[str, Any]:
    memory = _require_memory(_app(request))
    if not q.strip():
        return _error_response(BadRequest("查询内容不能为空"))
    messages = await memory.search_history(q, scope=scope, conversation_id=conversation_id, limit=limit)
    return {"messages": messages, "count": len(messages)}


@router.get("/memory/graph")
async def memory_graph(request: Request, scope: str = "", limit: int = 120, query: str = "") -> Dict[str, Any]:
    """Entities plus edges, for the memory visualisation panel."""
    memory = _require_memory(_app(request))
    entities = await memory.store.list_entities(scope=scope, limit=limit, query=query)
    relations = await memory.store.relations_for_entities([entity.id for entity in entities], scope=scope)
    return {
        "entities": [entity.to_public() for entity in entities],
        "relations": relations,
    }


@router.get("/memory/profile")
async def get_profile(request: Request, scope: str = "") -> Dict[str, Any]:
    memory = _require_memory(_app(request))
    return {"profile": await memory.store.get_profile(scope=scope)}


@router.put("/memory/profile")
async def upsert_profile(request: Request, payload: ProfileUpsertRequest) -> Dict[str, Any]:
    memory = _require_memory(_app(request))
    await memory.store.upsert_profile(
        payload.key, payload.value, confidence=payload.confidence, scope=payload.scope
    )
    return {"ok": True}


@router.delete("/memory/profile")
async def delete_profile(request: Request, key: str, scope: str = "") -> Dict[str, Any]:
    """Delete one profile attribute (the UI's "forget this about me" action)."""
    memory = _require_memory(_app(request))
    if not key.strip():
        return _error_response(BadRequest("key 不能为空"))
    await memory.store.delete_profile(key, scope=scope)
    return {"ok": True, "key": key}


@router.post("/memory/consolidate")
async def consolidate(request: Request) -> Dict[str, Any]:
    """Run a maintenance pass now (usually scheduled, exposed for the UI button)."""
    memory = _require_memory(_app(request))
    report = await memory.consolidate_now()
    return {"report": report.to_public()}


@router.post("/memory/extract/{session_id}")
async def force_extract(request: Request, session_id: str) -> Dict[str, Any]:
    """Re-run extraction for the last turn of a session (debugging aid)."""
    app = _app(request)
    memory = _require_memory(app)
    # SessionManager.get *raises* NotFound; it never returns None, so a bare
    # truthiness check would let the exception escape as an unhandled 500.
    try:
        session = await app.sessions.get(session_id)
    except NotFound as exc:
        return _error_response(exc)
    user_message: Optional[Message] = None
    assistant_message: Optional[Message] = None
    for message in reversed(session.messages):
        if assistant_message is None and message.role == Role.ASSISTANT:
            assistant_message = message
        elif message.role == Role.USER:
            user_message = message
            break
    if user_message is None:
        return _error_response(BadRequest("该会话还没有可抽取的用户消息"))
    outcome = await memory.extractor.process_turn(
        conversation_id=session.id,
        user_message=user_message,
        assistant_message=assistant_message,
        persona_id=session.persona_id,
        scope=session.scope,
        force=True,
    )
    return {"outcome": outcome.to_public()}


__all__ = ["router"]
