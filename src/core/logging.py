"""Logging setup.

Console output stays human readable, while ``logs/chatbot.jsonl`` receives one
JSON object per record. Structured logs matter here because the interesting
questions are quantitative: how many tokens a turn cost, how many memories were
retrieved, how long speech synthesis took.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_CONFIGURED = False
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
    "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """Render a record as a single JSON line, preserving ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: Dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name:<28} {record.getMessage()}"
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_") and k not in {"color_message"}
        }
        if extras:
            try:
                base += "  " + json.dumps(extras, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                base += f"  {extras}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    *,
    json_file: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Idempotently configure the root logger; returns a logger for the caller."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED and not force:
        return logging.getLogger("chatbot")

    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    if json_file:
        directory = Path(log_dir) if log_dir else Path(os.environ.get("CHATBOT_ROOT", ".")) / "logs"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                directory / "chatbot.jsonl", maxBytes=16 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            file_handler.setFormatter(JsonFormatter())
            root.addHandler(file_handler)
        except OSError:
            # A read-only or missing log directory must never stop the app.
            pass

    # Third-party noise control; the model servers are chatty.
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "watchfiles", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _CONFIGURED = True
    return logging.getLogger("chatbot")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["ConsoleFormatter", "JsonFormatter", "get_logger", "setup_logging"]
