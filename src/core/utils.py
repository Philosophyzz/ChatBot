"""Async helpers: bounded parallelism, retries, token estimation, timers.

These utilities exist so the rest of the codebase never hand-rolls a semaphore or
a retry loop. They are dependency-free and safe to unit test in isolation.
"""

from __future__ import annotations

import asyncio
import functools
import random
import re
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")

# --------------------------------------------------------------------------------------
# Token estimation
# --------------------------------------------------------------------------------------

#: Cheap heuristic: CJK characters are roughly one token each, latin text ~4 chars.
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_LATIN_WORD = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    """Estimate a token count without loading a tokenizer.

    Deliberately conservative (rounds up) because callers use it to decide whether a
    prompt still fits in the context window: being wrong *low* truncates a request at
    the model server, which is the failure mode that actually hurts.

    Calibration (measured, not guessed): for the sample used in
    ``tests/test_core.py`` — 16 han characters + the word "token" + 4 punctuation
    marks — a real Chinese tokenizer emits roughly one token per han character plus
    a few for the rest. The coefficients below round that up slightly: an overcount
    only trims history a little, an undercount makes the model server reject the
    request.
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    latin_matches = _LATIN_WORD.findall(text)
    latin_chars = sum(len(word) for word in latin_matches)
    remainder = max(0, len(text) - cjk - latin_chars)
    estimate = cjk * 1.05 + len(latin_matches) * 1.6 + remainder * 0.45
    return max(1, int(estimate + 0.999))


def estimate_messages_tokens(messages: Sequence[Any]) -> int:
    """Estimate the token cost of a chat payload, including per-message overhead."""
    total = 0
    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content", "")
        total += estimate_tokens(str(content or "")) + 4  # role + delimiters
    return total + 2


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-truncate text so its estimate fits ``max_tokens``."""
    if estimate_tokens(text) <= max_tokens:
        return text
    # Binary search on the character cut point keeps CJK/latin mixes honest.
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return text[:low]


# --------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------


@contextmanager
def timed() -> Any:
    """``with timed() as t: ...`` then ``t.ms``."""
    box = _TimingBox()
    start = time.perf_counter()
    try:
        yield box
    finally:
        box.ms = (time.perf_counter() - start) * 1000.0


class _TimingBox:
    __slots__ = ("ms",)

    def __init__(self) -> None:
        self.ms: Optional[float] = None


@asynccontextmanager
async def atimed() -> AsyncIterator[_TimingBox]:
    box = _TimingBox()
    start = time.perf_counter()
    try:
        yield box
    finally:
        box.ms = (time.perf_counter() - start) * 1000.0


# --------------------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------------------


def is_transient(exc: BaseException) -> bool:
    """Transport-level failures worth retrying. Never retries a partial stream."""
    name = type(exc).__name__
    if name in {"ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "RemoteProtocolError", "ConnectionResetError"}:
        return True
    text = str(exc).lower()
    return any(token in text for token in ("connection refused", "temporarily unavailable", "econnreset", "timed out"))


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.4,
    max_delay: float = 6.0,
    retry_on: Callable[[BaseException], bool] = is_transient,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> T:
    """Await ``func()`` with exponential backoff + jitter."""
    last: Optional[BaseException] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return await func()
        except BaseException as exc:  # noqa: BLE001 - re-raised below when exhausted
            last = exc
            if attempt >= attempts or not retry_on(exc):
                raise
            if on_retry:
                on_retry(attempt, exc)
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            await asyncio.sleep(delay * (0.7 + random.random() * 0.6))
    assert last is not None  # pragma: no cover
    raise last


def sync_retry(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.4,
    max_delay: float = 6.0,
) -> T:
    """Blocking variant used by setup/self-check scripts."""
    last: Optional[BaseException] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return func()
        except BaseException as exc:  # noqa: BLE001
            last = exc
            if attempt >= attempts or not is_transient(exc):
                raise
            time.sleep(min(max_delay, base_delay * (2 ** (attempt - 1))))
    assert last is not None  # pragma: no cover
    raise last


# --------------------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------------------


class LazySemaphore:
    """A semaphore created on first use, so it binds to the running loop."""

    def __init__(self, limit: int = 4) -> None:
        self.limit = max(1, limit)
        self._sem: Optional[asyncio.Semaphore] = None

    def _get(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.limit)
        return self._sem

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[None]:
        async with self._get():
            yield


async def gather_limited(
    coros: Iterable[Awaitable[T]], limit: int = 4, *, return_exceptions: bool = True
) -> List[Any]:
    """Run awaitables with bounded concurrency, preserving input order."""
    sem = LazySemaphore(limit)

    async def guarded(coro: Awaitable[T]) -> T:
        async with sem():
            return await coro

    return await asyncio.gather(*(guarded(c) for c in coros), return_exceptions=return_exceptions)


def fire_and_forget(coro: Awaitable[Any], *, name: str = "task") -> asyncio.Task:
    """Schedule background work and keep a strong reference so it is not GC'd."""
    task = asyncio.create_task(coro, name=name)
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    return task


_BACKGROUND: set = set()


# --------------------------------------------------------------------------------------
# Text utilities shared by memory + speech
# --------------------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n\.])")


def split_sentences(text: str, max_chars: int = 120) -> List[str]:
    """Split text into speakable sentences, merging fragments that are too short.

    Used to stream TTS sentence by sentence: the first sentence can start playing
    while the rest of the answer is still being generated.
    """
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    out: List[str] = []
    buffer = ""
    for part in parts:
        if not buffer:
            buffer = part
        elif len(buffer) + len(part) <= max_chars:
            buffer += part
        else:
            out.append(buffer)
            buffer = part
        while len(buffer) > max_chars:
            out.append(buffer[:max_chars])
            buffer = buffer[max_chars:]
    if buffer:
        out.append(buffer)
    return out


def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t\u3000]+", " ", text).strip()


_STRIP_MD = re.compile(r"```.*?```|`([^`]*)`|\*\*([^*]*)\*\*|\*([^*]*)\*|^#{1,6}\s*|^\s*[-*+]\s+|^\s*>\s*", re.M | re.S)


def speakable_text(text: str) -> str:
    """Strip markdown/code so the TTS does not read symbols aloud."""
    def _sub(match: re.Match) -> str:
        return match.group(1) or match.group(2) or match.group(3) or ""

    cleaned = _STRIP_MD.sub(_sub, text)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)  # links
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


__all__ = [
    "LazySemaphore",
    "atimed",
    "estimate_messages_tokens",
    "estimate_tokens",
    "fire_and_forget",
    "gather_limited",
    "is_transient",
    "normalize_whitespace",
    "retry_async",
    "speakable_text",
    "split_sentences",
    "sync_retry",
    "timed",
    "truncate_to_tokens",
]
