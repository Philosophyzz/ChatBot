"""Session state: persona binding, in-memory history and persistence.

A session is the unit the UI thinks in ("this conversation") and the unit memory
extraction is keyed to. Two design points:

* **History is a window, not a transcript.** The full record lives in SQLite; the
  session keeps a bounded list of recent messages purely to avoid a database read
  on every token. When a session is restored, the window is refilled from the
  store.
* **Scope isolation.** Each session carries a memory scope so a future multi-user
  or multi-character setup can keep separate memory namespaces without a schema
  change. Today the default scope is ``global`` (one user, one assistant).
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.errors import NotFound
from core.logging import get_logger
from core.types import Message, Role, new_id, now_ts

log = get_logger(__name__)

#: How many recent messages to keep in memory per session for prompt building.
DEFAULT_WINDOW = 60


@dataclass
class Session:
    """Live conversation state."""

    id: str
    persona_id: str
    scope: str = "global"
    title: str = ""
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)
    turn_count: int = 0
    #: Last memory-retrieval result, kept for the "why did it say that" panel.
    last_memory: Dict[str, Any] = field(default_factory=dict)
    #: Voice mode for this session: when on, replies are optimised for speech.
    voice_mode: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def append(self, message: Message, *, window: int = DEFAULT_WINDOW) -> None:
        self.messages.append(message)
        if len(self.messages) > window:
            # Drop from the front: the system prompt is rebuilt every turn and the
            # memory block carries anything older that still matters.
            self.messages = self.messages[-window:]
        self.updated_at = message.created_at
        if message.role == Role.USER:
            self.turn_count += 1
        if not self.title and message.role == Role.USER:
            self.title = _derive_title(message.content)

    def public_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title or "新对话",
            "persona_id": self.persona_id,
            "scope": self.scope,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_count": self.turn_count,
            "message_count": len(self.messages),
            "voice_mode": self.voice_mode,
            "last_memory": self.last_memory,
        }

    def to_public(self, *, limit: int = 100) -> Dict[str, Any]:
        payload = self.public_summary()
        payload["messages"] = [m.to_public() for m in self.messages[-limit:]]
        return payload


def _derive_title(text: str, *, max_chars: int = 24) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    return cleaned[:max_chars] + ("…" if len(cleaned) > max_chars else "")


class SessionManager:
    """Creates, caches and restores sessions."""

    def __init__(self, store: Any = None, *, max_live_sessions: int = 64) -> None:
        self.store = store
        self.max_live = max(4, int(max_live_sessions))
        # LRU: a long-running server must not grow without bound.
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------------------
    async def create(
        self,
        *,
        persona_id: str,
        scope: str = "global",
        session_id: Optional[str] = None,
        title: str = "",
        voice_mode: bool = False,
    ) -> Session:
        session = Session(
            id=session_id or new_id("conv_"),
            persona_id=persona_id,
            scope=scope,
            title=title,
            voice_mode=voice_mode,
        )
        async with self._lock:
            self._sessions[session.id] = session
            self._evict()
        if self.store is not None:
            try:
                await self.store.ensure_conversation(
                    session.id, title=title, persona_id=persona_id, scope=scope
                )
            except Exception:  # noqa: BLE001
                log.exception("failed to persist new conversation")
        return session

    async def get(self, session_id: str, *, restore: bool = True) -> Session:
        session = self._sessions.get(session_id)
        if session is not None:
            self._sessions.move_to_end(session_id)
            return session
        if self.store is None or not restore:
            raise NotFound(f"会话不存在：{session_id}")
        record = await self.store.get_conversation(session_id)
        if record is None:
            raise NotFound(f"会话不存在：{session_id}")
        session = Session(
            id=record["id"],
            persona_id=record.get("persona_id") or "",
            scope=record.get("scope") or "global",
            title=record.get("title") or "",
            created_at=float(record.get("created_at") or now_ts()),
            updated_at=float(record.get("updated_at") or now_ts()),
            turn_count=int(record.get("turn_count") or 0),
        )
        messages = await self.store.get_messages(session_id, limit=DEFAULT_WINDOW)
        session.messages = list(messages)
        async with self._lock:
            self._sessions[session.id] = session
            self._evict()
        log.info("restored session from disk", extra={"session": session_id, "messages": len(messages)})
        return session

    async def get_or_create(
        self, session_id: Optional[str], *, persona_id: str, scope: str = "global", voice_mode: bool = False
    ) -> Session:
        if not session_id:
            return await self.create(persona_id=persona_id, scope=scope, voice_mode=voice_mode)
        try:
            session = await self.get(session_id)
        except NotFound:
            return await self.create(
                persona_id=persona_id, scope=scope, session_id=session_id, voice_mode=voice_mode
            )
        if persona_id and session.persona_id != persona_id:
            session.persona_id = persona_id
            if self.store is not None:
                try:
                    await self.store.update_conversation(session.id, persona_id=persona_id)
                except Exception:  # noqa: BLE001
                    pass
        return session

    async def list(self, *, scope: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        if self.store is None:
            return [s.public_summary() for s in list(self._sessions.values())[-limit:]]
        records = await self.store.list_conversations(scope=scope, limit=limit)
        # Overlay live state so an in-progress conversation shows its real turn count.
        for record in records:
            live = self._sessions.get(record["id"])
            if live is not None:
                record["turn_count"] = live.turn_count
                record["title"] = live.title or record["title"]
                record["updated_at"] = live.updated_at
        return records

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
        if self.store is not None:
            await self.store.delete_conversation(session_id)

    async def rename(self, session_id: str, title: str) -> None:
        session = await self.get(session_id)
        session.title = title
        if self.store is not None:
            await self.store.update_conversation(session_id, title=title)

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _evict(self) -> None:
        while len(self._sessions) > self.max_live:
            self._sessions.popitem(last=False)


__all__ = ["DEFAULT_WINDOW", "Session", "SessionManager"]
