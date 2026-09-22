"""Wire models for the HTTP/WebSocket surface.

Kept separate from the core domain types on purpose: the wire format is a public
contract that deserves to evolve independently (and be validated) rather than
being whatever a dataclass happens to look like today.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - pydantic is a declared dependency
    raise ImportError("pydantic is required by the API layer (pip install pydantic)") from exc


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=16000, description="用户输入文本")
    session_id: Optional[str] = Field(None, description="会话 id；省略则新建")
    persona_id: Optional[str] = Field(None, description="人设 id；省略则用会话或默认人设")
    voice_mode: bool = Field(False, description="是否按语音朗读优化输出")
    stream: bool = Field(True, description="是否流式返回")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    speak: bool = Field(False, description="是否同时返回合成语音（SSE 内联 base64）")
    extra_instructions: str = Field("", max_length=2000)


class SessionCreateRequest(BaseModel):
    persona_id: Optional[str] = None
    scope: str = "global"
    title: str = ""
    voice_mode: bool = False


class SessionRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class PersonaUpsertRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    system_prompt: str = Field(..., min_length=1, max_length=8000)
    description: str = Field("", max_length=400)
    greeting: str = Field("", max_length=400)
    avatar: str = Field("🙂", max_length=8)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    memory_scope: str = "global"
    options: Dict[str, Any] = Field(default_factory=dict)
    voice: Dict[str, Any] = Field(default_factory=dict)


class MemoryUpdateRequest(BaseModel):
    content: Optional[str] = Field(None, max_length=4000)
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None


class MemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    kind: str = Field("fact")
    importance: float = Field(0.6, ge=0.0, le=1.0)
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    scope: str = "global"


class ProfileUpsertRequest(BaseModel):
    key: str = Field(..., min_length=1, max_length=48)
    value: str = Field(..., min_length=1, max_length=400)
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    scope: str = "global"


class ProfileDeleteRequest(BaseModel):
    key: str = Field(..., min_length=1, max_length=48)
    scope: str = "global"


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    persona_id: Optional[str] = None
    backend: Optional[str] = None
    voice: Optional[Dict[str, Any]] = None
    stream: bool = False


class SettingsPatch(BaseModel):
    llm: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None
    speech: Optional[Dict[str, Any]] = None
    debug: Optional[bool] = None
    default_persona: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool
    version: str
    uptime_s: float
    llm: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    stt: Dict[str, Any] = Field(default_factory=dict)
    tts: Dict[str, Any] = Field(default_factory=dict)
    components: List[Dict[str, Any]] = Field(default_factory=list)
    registry: Dict[str, List[str]] = Field(default_factory=dict)


__all__ = [
    "ChatRequest",
    "HealthResponse",
    "MemoryCreateRequest",
    "MemoryUpdateRequest",
    "PersonaUpsertRequest",
    "ProfileDeleteRequest",
    "ProfileUpsertRequest",
    "SessionCreateRequest",
    "SessionRenameRequest",
    "SettingsPatch",
    "TTSRequest",
]
