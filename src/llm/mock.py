"""Deterministic in-process LLM used for tests, demos and offline development.

It exists so the whole pipeline — streaming, memory extraction, retrieval, TTS
routing, the web UI — can be exercised end to end on a laptop with no GPU and no
15 GB download. It is also the reference implementation of the backend contract:
if a change breaks this file's assumptions, it breaks every real backend too.

Behaviour worth knowing:
* ``llm.backend: mock`` makes the server fully functional offline.
* Reasoning deltas are emitted so the UI's "thinking" block can be tested.
* When a request carries a ``json_schema``, it returns shape-correct JSON, which
  is what the memory extractor asks for in tests.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from core.registry import KIND_LLM, register
from core.types import Completion, LLMRequest, LLMStats, Message, Role, TokenEvent

#: Canned "facts" the mock extractor reports. They are intentionally about the
#: user so the memory pipeline has something meaningful to store and retrieve.
_MOCK_EXTRACTION = {
    "facts": [
        {
            "content": "用户的名字是小明",
            "subject": "用户",
            "predicate": "名字",
            "object": "小明",
            "kind": "profile",
            "importance": 0.9,
            "confidence": 0.95,
            "tags": ["身份"],
        },
        {
            "content": "用户喜欢喝美式咖啡，不加糖",
            "subject": "用户",
            "predicate": "喜欢",
            "object": "美式咖啡",
            "kind": "preference",
            "importance": 0.6,
            "confidence": 0.8,
            "tags": ["饮食"],
        },
        {
            "content": "用户正在开发一个本地聊天机器人项目",
            "subject": "用户",
            "predicate": "正在做",
            "object": "本地聊天机器人",
            "kind": "fact",
            "importance": 0.7,
            "confidence": 0.85,
            "tags": ["工作"],
        },
    ],
    "entities": [
        {"name": "用户", "type": "person", "aliases": ["我", "小明"]},
        {"name": "美式咖啡", "type": "concept", "aliases": []},
        {"name": "本地聊天机器人", "type": "project", "aliases": []},
    ],
    "relations": [
        {"subject": "用户", "predicate": "喜欢", "object": "美式咖啡", "confidence": 0.8},
    ],
    "profile": [
        {"key": "名字", "value": "小明", "confidence": 0.95},
        {"key": "饮品偏好", "value": "美式咖啡（不加糖）", "confidence": 0.8},
    ],
}


@register(KIND_LLM, "mock")
class MockLLMBackend:
    """A scripted assistant. Never touches the network or the GPU."""

    name = "mock"

    def __init__(self, *, delay_ms: float = 12.0, model: str = "mock-1", **_ignored: Any) -> None:
        self.delay_s = max(0.0, float(delay_ms) / 1000.0)
        self.model = model
        self.calls: List[LLMRequest] = []

    # -- contract ----------------------------------------------------------------------
    async def stream(self, request: LLMRequest) -> AsyncIterator[TokenEvent]:
        self.calls.append(request)
        last_user = _last_user_text(request)
        started = asyncio.get_event_loop().time()

        if request.json_schema is not None:
            payload = json.dumps(_MOCK_EXTRACTION, ensure_ascii=False)
            for chunk in _chunks(payload, 48):
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
                yield TokenEvent("text", chunk)
            stats = LLMStats(completion_tokens=len(payload) // 3, model=self.model)
            yield TokenEvent("usage", data=stats.to_public())
            yield TokenEvent("done", data={"finish_reason": "stop", "stats": stats.to_public()})
            return

        thinking = f"（模拟推理：用户说的是「{last_user[:40]}」，需要给出简短友好的回应。）"
        answer = self._answer_for(last_user)

        for chunk in _chunks(thinking, 24):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield TokenEvent("reasoning", chunk)
        for chunk in _chunks(answer, 12):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield TokenEvent("text", chunk)

        elapsed = (asyncio.get_event_loop().time() - started) * 1000
        stats = LLMStats(
            prompt_tokens=sum(len(m.content) // 3 for m in request.messages),
            completion_tokens=max(1, len(answer) // 3),
            total_tokens=0,
            ttft_ms=round(elapsed / 3, 1),
            total_ms=round(elapsed, 1),
            model=self.model,
            timings={"predicted_per_second": round(max(1.0, len(answer) / max(0.05, elapsed / 1000)), 2)},
        )
        stats.total_tokens = stats.prompt_tokens + stats.completion_tokens
        yield TokenEvent("usage", data=stats.to_public())
        yield TokenEvent("done", data={"finish_reason": "stop", "stats": stats.to_public()})

    async def complete(self, request: LLMRequest) -> Completion:
        text_parts: List[str] = []
        stats = LLMStats(model=self.model)
        async for event in self.stream(request):
            if event.kind == "text":
                text_parts.append(event.text)
            elif event.kind == "usage":
                stats = LLMStats(**{k: v for k, v in event.data.items() if k in LLMStats.__dataclass_fields__})
            elif event.kind == "done":
                break
        text = "".join(text_parts)
        return Completion(
            text=text,
            message=Message(role=Role.ASSISTANT, content=text, meta={"model": self.model, "mock": True}),
            stats=stats,
        )

    async def health(self) -> Dict[str, Any]:
        return {"ok": True, "name": self.name, "detail": "内置模拟模型（无需 GPU）", "models": [self.model]}

    async def aclose(self) -> None:  # pragma: no cover - symmetry with real backends
        return None

    # -- personality -------------------------------------------------------------------
    def _answer_for(self, user_text: str) -> str:
        """Small keyword rules so demos feel responsive instead of canned."""
        text = user_text or ""
        if re.search(r"名字|我是谁|记得", text):
            return "当然记得呀，你叫小明。你上次还说你只喝不加糖的美式咖啡呢～"
        if re.search(r"你好|在吗|hi|hello", text, re.I):
            return "我在呢～今天想聊点什么？我一直都听着呢。"
        if re.search(r"咖啡|喝", text):
            return "美式不加糖，记住了。今天喝了第几杯啦？"
        if re.search(r"项目|代码|机器人", text):
            return "你的本地聊天机器人项目进展怎么样？需要我帮你梳理架构吗？"
        if re.search(r"时间|几点", text):
            return "我这边随时在线，不过准确时间还得看你的系统时钟哦。"
        tail = text.strip()[:30]
        return f"我听到你说「{tail}」啦。这条我会记下来，下次聊天就能用上。（当前是模拟模型，配置好本地大模型后就是真实回答）"


def _chunks(text: str, size: int) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _last_user_text(request: LLMRequest) -> str:
    for message in reversed(list(request.messages)):
        if message.role == Role.USER:
            return message.content
    return ""


__all__ = ["MockLLMBackend"]
