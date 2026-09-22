"""The chat engine: one turn, start to finish.

Responsibilities, in the order they happen:

1. Resolve session, persona and memory scope.
2. Retrieve memory **in parallel with nothing else blocking** and emit a
   ``memory`` event, so the UI can show the "thinking about what I remember"
   affordance while the model warms up.
3. Assemble the prompt: persona prompt + time + memory block + a history window
   trimmed to the configured context budget. Trimming is *middle-out* — the most
   recent turns and the system prompt survive, the distant middle is replaced by a
   note, because losing the beginning of a long conversation hurts less than losing
   the current thread.
4. Stream the completion, emitting ``reasoning``/``text`` deltas as they arrive and
   forwarding llama.cpp's token-rate telemetry.
5. Persist both messages and hand the turn to the memory write path
   asynchronously, so memory work never delays the reply.

Everything the engine emits is a plain dict event, which is what makes the whole
thing channel-agnostic: the WebSocket handler relays them, a test collects them,
and a future desktop shell would consume the same stream.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from core.errors import BackendUnavailable, ChatBotError
from core.logging import get_logger
from core.session import Session, SessionManager
from core.types import LLMRequest, LLMStats, Message, Role
from core.utils import estimate_messages_tokens, estimate_tokens
from memory.engine import MemoryEngine
from persona.manager import PersonaManager, build_system_prompt

log = get_logger(__name__)


@dataclass
class TurnResult:
    """Outcome of one completed turn (used by tests and non-streaming callers)."""

    text: str = ""
    thinking: str = ""
    message: Optional[Message] = None
    stats: LLMStats = field(default_factory=LLMStats)
    memory: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


class ChatEngine:
    """Turns a user utterance into a streamed, memory-aware reply."""

    def __init__(
        self,
        *,
        config: Any,
        llm: Any,
        personas: PersonaManager,
        memory: Optional[MemoryEngine],
        sessions: SessionManager,
        stt: Any = None,
        tts: Any = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.personas = personas
        self.memory = memory
        self.sessions = sessions
        self.stt = stt
        self.tts = tts
        self._active_turns = 0

    # ----------------------------------------------------------------------------------
    # Prompt assembly
    # ----------------------------------------------------------------------------------
    async def build_prompt(
        self,
        session: Session,
        user_text: str,
        *,
        persona_id: Optional[str] = None,
        voice_mode: bool = False,
        extra_instructions: str = "",
    ) -> Dict[str, Any]:
        """Assemble messages + metadata for one completion request."""
        persona = self.personas.get(persona_id or session.persona_id)
        started = time.perf_counter()

        memory_context = None
        if self.memory is not None:
            memory_context = await self.memory.build_context(
                user_text,
                session_id=session.id,
                persona_id=persona.id,
                scope=persona.memory_scope or session.scope,
            )
        memory_ms = round((time.perf_counter() - started) * 1000, 1)

        system_prompt = build_system_prompt(
            persona,
            memory_block=memory_context.rendered if memory_context else "",
            extra_instructions=extra_instructions,
            voice_mode=voice_mode,
        )

        budget = max(512, int(self.config.llm.context_tokens) - int(self.config.llm.reserve_tokens))
        system_tokens = estimate_tokens(system_prompt)
        history = self._trim_history(
            session.messages, budget=max(256, budget - system_tokens)
        )
        messages = [Message(role=Role.SYSTEM, content=system_prompt)]
        messages.extend(history)
        messages.append(Message(role=Role.USER, content=user_text))

        return {
            "persona": persona,
            "messages": messages,
            "memory": memory_context,
            "prompt_tokens_est": estimate_messages_tokens(messages),
            "history_turns": len(history),
            "memory_ms": memory_ms,
        }

    @staticmethod
    def _trim_history(messages: List[Message], *, budget: int) -> List[Message]:
        """Keep the newest turns within ``budget``, noting what was elided."""
        if not messages:
            return []
        selected: List[Message] = []
        used = 0
        for message in reversed(messages):
            cost = estimate_tokens(message.content) + 4
            if used + cost > budget and selected:
                break
            selected.append(message)
            used += cost
        selected.reverse()
        dropped = len(messages) - len(selected)
        if dropped > 0:
            note = Message(
                role=Role.SYSTEM,
                content=f"（此前还有 {dropped} 条较早的对话未展示；如需回忆请参考上面的长期记忆，或向用户确认。）",
                meta={"synthetic": True, "dropped": dropped},
            )
            selected.insert(0, note)
        return selected

    # ----------------------------------------------------------------------------------
    # Streaming turn
    # ----------------------------------------------------------------------------------
    async def stream_turn(
        self,
        *,
        session_id: Optional[str],
        user_text: str,
        persona_id: Optional[str] = None,
        voice_mode: bool = False,
        scope: str = "global",
        message_meta: Optional[Dict[str, Any]] = None,
        extra_instructions: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield event dicts for one turn.

        Event kinds (stable API for every channel):
        ``session``, ``memory``, ``memory_skipped``, ``start``, ``reasoning``,
        ``text``, ``usage``, ``done``, ``error``.
        """
        started = time.perf_counter()
        self._active_turns += 1
        session: Optional[Session] = None
        try:
            session = await self.sessions.get_or_create(
                session_id,
                persona_id=persona_id or self.config.default_persona,
                scope=scope,
                voice_mode=voice_mode,
            )
            session.voice_mode = voice_mode
            yield {"kind": "session", "session": session.public_summary()}

            try:
                prompt = await self.build_prompt(
                    session,
                    user_text,
                    persona_id=persona_id,
                    voice_mode=voice_mode,
                    extra_instructions=extra_instructions,
                )
            except ChatBotError as exc:
                yield {"kind": "error", **exc.to_public()}
                return

            persona = prompt["persona"]
            memory_context = prompt["memory"]
            if memory_context is not None:
                yield {
                    "kind": "memory",
                    "memory": memory_context.to_public(include_hits=self.config.debug),
                    "rendered": memory_context.rendered if self.config.debug else None,
                }
            else:
                yield {"kind": "memory_skipped", "reason": "memory disabled"}

            user_message = Message(
                role=Role.USER,
                content=user_text,
                meta={**(message_meta or {}), "session_id": session.id},
            )
            session.append(user_message)
            if self.memory is not None:
                try:
                    await self.memory.store.ensure_conversation(
                        session.id,
                        title=session.title,
                        persona_id=persona.id,
                        scope=persona.memory_scope or scope,
                    )
                    await self.memory.store.add_message(
                        session.id,
                        user_message,
                        persona_id=persona.id,
                        scope=persona.memory_scope or scope,
                    )
                except Exception:  # noqa: BLE001 - persistence must not break the reply
                    log.exception("failed to persist user message")

            request = LLMRequest(
                messages=prompt["messages"],
                temperature=float(temperature if temperature is not None else persona.temperature),
                top_p=persona.top_p,
                max_tokens=int(max_tokens or persona.max_tokens),
            )
            yield {
                "kind": "start",
                "session_id": session.id,
                "persona": persona.to_public(),
                "prompt_tokens_est": prompt["prompt_tokens_est"],
                "history_turns": prompt["history_turns"],
                "memory_ms": prompt["memory_ms"],
            }

            text_parts: List[str] = []
            thinking_parts: List[str] = []
            stats = LLMStats(model=getattr(self.llm, "model", ""))
            try:
                async for event in self.llm.stream(request):
                    if event.kind == "text":
                        text_parts.append(event.text)
                        yield {"kind": "text", "text": event.text}
                    elif event.kind == "reasoning":
                        thinking_parts.append(event.text)
                        yield {"kind": "reasoning", "text": event.text}
                    elif event.kind == "tool":
                        yield {"kind": "tool", "data": event.data}
                    elif event.kind == "usage":
                        stats = _stats_from(event.data)
                        yield {"kind": "usage", "stats": stats.to_public()}
                    elif event.kind == "error":
                        yield {"kind": "error", "error": {"code": "stream_error", "message": event.data.get("message", "生成中断")}}
                    elif event.kind == "done":
                        break
            except ChatBotError as exc:
                yield {"kind": "error", **exc.to_public()}
                return
            except Exception as exc:  # noqa: BLE001
                log.exception("llm stream failed")
                yield {
                    "kind": "error",
                    "error": {"code": "llm_failed", "message": f"模型调用失败：{exc}"},
                }
                return

            answer = "".join(text_parts).strip()
            thinking = "".join(thinking_parts).strip()
            assistant_message = Message(
                role=Role.ASSISTANT,
                content=answer,
                meta={
                    "model": stats.model or getattr(self.llm, "model", ""),
                    "persona_id": persona.id,
                    "ttft_ms": stats.ttft_ms,
                    "tokens_per_second": stats.tokens_per_second,
                    "thinking": bool(thinking),
                },
            )
            session.append(assistant_message)
            session.last_memory = memory_context.to_public() if memory_context else {}

            if self.memory is not None:
                try:
                    await self.memory.store.add_message(
                        session.id,
                        assistant_message,
                        tokens=stats.completion_tokens,
                        persona_id=persona.id,
                        scope=persona.memory_scope or scope,
                    )
                    await self.memory.store.update_conversation(
                        session.id,
                        summary=answer[:400] if session.turn_count <= 1 else None,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("failed to persist assistant message")
                try:
                    await self.memory.remember_turn(
                        session_id=session.id,
                        user_message=user_message,
                        assistant_message=assistant_message,
                        persona_id=persona.id,
                        scope=persona.memory_scope or scope,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("failed to queue memory extraction")

            elapsed = (time.perf_counter() - started) * 1000
            yield {
                "kind": "done",
                "session_id": session.id,
                "message_id": assistant_message.id,
                "text": answer,
                "thinking": thinking if self.config.debug else None,
                "stats": stats.to_public(),
                "elapsed_ms": round(elapsed, 1),
                "memory": memory_context.to_public() if memory_context else None,
            }
        finally:
            self._active_turns -= 1

    # ----------------------------------------------------------------------------------
    # Non-streaming convenience
    # ----------------------------------------------------------------------------------
    async def complete_turn(self, **kwargs: Any) -> TurnResult:
        """Collect a streamed turn into a :class:`TurnResult` (tests, CLI, scripts)."""
        result = TurnResult()
        async for event in self.stream_turn(**kwargs):
            result.events.append(event)
            kind = event.get("kind")
            if kind == "text":
                result.text += event.get("text", "")
            elif kind == "reasoning":
                result.thinking += event.get("text", "")
            elif kind == "usage":
                result.stats = _stats_from(event.get("stats") or {})
            elif kind == "memory":
                result.memory = event.get("memory") or {}
            elif kind == "done":
                result.text = event.get("text") or result.text
                result.stats = _stats_from(event.get("stats") or {})
                result.message = Message(
                    role=Role.ASSISTANT,
                    content=result.text,
                    id=event.get("message_id") or "",
                    meta={"stats": event.get("stats") or {}},
                )
            elif kind == "error":
                error = event.get("error") or {}
                raise BackendUnavailable(error.get("message", "生成失败"), detail=error)
        return result

    # ----------------------------------------------------------------------------------
    # Speech helpers
    # ----------------------------------------------------------------------------------
    async def synthesize_speech(
        self, text: str, *, persona_id: Optional[str] = None, stream: bool = True
    ) -> AsyncIterator[Any]:
        """Synthesize an answer with the persona's voice, sentence by sentence.

        Chunking by sentence is what makes playback start early: the first sentence
        is spoken while the rest is still being generated or synthesized.
        """
        if self.tts is None:
            raise BackendUnavailable("语音合成未启用")
        from core.utils import speakable_text, split_sentences

        persona = self.personas.get(persona_id or self.config.default_persona)
        # Same chunk budget as the streaming TTS route: smaller chunks mean the first
        # words are spoken while the rest is still being synthesized.
        chunk_chars = max(16, int(getattr(self.config.speech, "tts_chunk_chars", 40) or 40))
        blocks = split_sentences(speakable_text(text), max_chars=chunk_chars) or [speakable_text(text)]
        for index, block in enumerate(blocks):
            if not block.strip():
                continue
            async for chunk in self.tts.synthesize(block, persona.voice, stream=stream):
                yield chunk
            await asyncio.sleep(0)  # yield control between sentences

    async def transcribe(self, clip: Any, *, prompt: Optional[str] = None) -> Any:
        if self.stt is None:
            raise BackendUnavailable("语音识别未启用")
        return await self.stt.transcribe(
            clip, language=self.config.speech.stt_language or None, prompt=prompt
        )

    # ----------------------------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "llm": await _safe_health(self.llm),
            "active_turns": self._active_turns,
            "personas": len(self.personas.load()),
            "context_tokens": self.config.llm.context_tokens,
        }
        if self.memory is not None:
            payload["memory"] = await self.memory.health()
        if self.stt is not None:
            payload["stt"] = await _safe_health(self.stt)
        if self.tts is not None:
            payload["tts"] = await _safe_health(self.tts)
        return payload


async def _safe_health(component: Any) -> Dict[str, Any]:
    if component is None or not hasattr(component, "health"):
        return {"ok": False, "error": "未配置"}
    try:
        return await component.health()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}


def _stats_from(data: Dict[str, Any]) -> LLMStats:
    fields = {
        key: value
        for key, value in (data or {}).items()
        if key in LLMStats.__dataclass_fields__ and key != "timings"
    }
    stats = LLMStats(**fields)
    if isinstance(data.get("timings"), dict):
        stats.timings = data["timings"]
    return stats


__all__ = ["ChatEngine", "TurnResult"]
