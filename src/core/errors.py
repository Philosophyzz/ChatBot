"""Error taxonomy.

Every failure the engine reports to a caller is one of these types, so the API
layer can map exceptions to status codes without string matching, and the UI can
show an actionable Chinese message instead of a stack trace.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ChatBotError(RuntimeError):
    """Base class. ``code`` is stable and safe to show to a user."""

    code = "chatbot_error"
    http_status = 500

    def __init__(self, message: str, *, detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: Dict[str, Any] = detail or {}

    def to_public(self) -> Dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "detail": self.detail}}


class ConfigError(ChatBotError):
    code = "config_error"
    http_status = 500


class BackendUnavailable(ChatBotError):
    """A model server or optional dependency is not reachable/installed."""

    code = "backend_unavailable"
    http_status = 503


class BackendTimeout(ChatBotError):
    code = "backend_timeout"
    http_status = 504


class UpstreamError(ChatBotError):
    """The backend answered, but with an error."""

    code = "upstream_error"
    http_status = 502


class BadRequest(ChatBotError):
    code = "bad_request"
    http_status = 400


class NotFound(ChatBotError):
    code = "not_found"
    http_status = 404


class Conflict(ChatBotError):
    code = "conflict"
    http_status = 409


class AudioError(ChatBotError):
    code = "audio_error"
    http_status = 422


class Unsupported(ChatBotError):
    code = "unsupported"
    http_status = 501


__all__ = [
    "AudioError",
    "BackendTimeout",
    "BackendUnavailable",
    "BadRequest",
    "ChatBotError",
    "ConfigError",
    "Conflict",
    "NotFound",
    "Unsupported",
    "UpstreamError",
]
