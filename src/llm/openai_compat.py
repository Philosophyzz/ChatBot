"""OpenAI-compatible chat backend with hand-rolled SSE streaming.

One implementation serves every local runtime we care about, because they all
speak the same dialect:

* ``llama-server`` (llama.cpp)  — the shipped default, best GPU-offload control
* Ollama                       — ``/v1`` compatibility endpoint
* vLLM / SGLang / LM Studio    — same surface

Why not the official ``openai`` SDK? Three reasons that matter for this product:

1. **Reasoning models.** Qwen3.x emit ``reasoning_content`` deltas alongside
   ``content``. We surface them as a distinct event kind so the UI can render a
   collapsible "thinking" block without polluting the spoken answer.
2. **Prompt-cache telemetry.** llama.cpp reports ``timings`` in its final chunk
   (``prompt_ms``, ``predicted_per_second``, cache hits). That is exactly the data
   needed to tune GPU-layer offload on a 16 GB card, and the SDK drops it.
3. **Exact control of the request body.** We need to pass server-specific fields
   (``cache_prompt``, ``n_keep``, ``grammar``) without SDK validation noise.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from core.errors import BackendTimeout, BackendUnavailable, UpstreamError
from core.logging import get_logger
from core.registry import KIND_LLM, register
from core.types import Completion, LLMRequest, LLMStats, Message, Role, TokenEvent
from core.utils import retry_async

log = get_logger(__name__)

try:  # httpx is the single required HTTP dependency; urllib is the escape hatch.
    import httpx

    _HAS_HTTPX = True
except Exception:  # pragma: no cover - exercised only on a broken install
    httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False


@register(KIND_LLM, "llamacpp")
@register(KIND_LLM, "openai")
@register(KIND_LLM, "ollama")
@register(KIND_LLM, "vllm")
class OpenAICompatBackend:
    """Streaming chat client for any OpenAI-compatible ``/v1`` endpoint.

    The three registered aliases exist so configuration reads naturally
    (``llm.backend: ollama``) while the implementation stays single-sourced.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "local-model",
        api_key: str = "sk-local",
        *,
        timeout_s: float = 300.0,
        connect_timeout_s: float = 8.0,
        max_retries: int = 2,
        extra_body: Optional[Dict[str, Any]] = None,
        name: str = "llamacpp",
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "sk-local"
        self.timeout_s = float(timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.max_retries = int(max_retries)
        self.extra_body = dict(extra_body or {})
        self._client: Optional[Any] = None
        self._lock = asyncio.Lock()

    # -- plumbing ----------------------------------------------------------------------
    async def _get_client(self) -> Any:
        if not _HAS_HTTPX:
            raise BackendUnavailable(
                "缺少 httpx 依赖，请先运行 scripts/install.ps1 安装 Python 依赖",
                detail={"hint": "pip install httpx"},
            )
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        base_url=self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s),
                        limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
# trust_env=False is REQUIRED, not cosmetic: httpx reads the system proxy
                        # (e.g. Clash on 127.0.0.1:7890) and would route requests to
                        # *local* services through it. The proxy cannot reach a
                        # localhost port and answers 502, so the app reports
                        # "model server broken" when the real problem is the proxy.
                        # `proxy=None` does NOT help - measured: it still returned 502;
                        # only trust_env=False produced a real ConnectError.
                        trust_env=False,
                    )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    def _payload(self, request: LLMRequest, *, stream: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if stream:
            # Ask llama.cpp/vLLM to include the final usage block when supported.
            body["stream_options"] = {"include_usage": True}
            # llama.cpp extension: reuse the KV prefix between turns. This is the
            # single biggest latency win in a chat app and is safe because every
            # turn extends the same conversation prefix.
            body["cache_prompt"] = True
        if request.stop:
            body["stop"] = list(request.stop)
        if request.json_schema:
            # llama.cpp honours a raw JSON schema; OpenAI-style servers want the
            # wrapped response_format. Send both: each server ignores the other.
            body["json_schema"] = request.json_schema
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "strict": False, "schema": request.json_schema},
            }
        body.update(self.extra_body)
        body.update(request.extra or {})
        return body

    # -- streaming ---------------------------------------------------------------------
    async def stream(self, request: LLMRequest) -> AsyncIterator[TokenEvent]:
        """Yield :class:`TokenEvent` deltas.

        Retries only cover *opening* the stream. Once the first byte arrives a
        retry would duplicate visible text, so mid-stream failures propagate.
        """
        client = await self._get_client()
        payload = self._payload(request, stream=True)
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        usage: Dict[str, Any] = {}
        timings: Dict[str, Any] = {}
        finish_reason: Optional[str] = None
        produced_any = False

        async def _open() -> Any:
            # build_request + send (instead of post) is what makes the response
            # headers available before the body finishes streaming.
            req = client.build_request("POST", "/chat/completions", json=payload)
            return await client.send(req, stream=True)

        try:
            response = await retry_async(
                _open,
                attempts=self.max_retries + 1,
                on_retry=lambda attempt, exc: log.warning(
                    "llm stream open retry", extra={"attempt": attempt, "error": str(exc)}
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc, self.base_url) from exc

        try:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")[:800]
                raise UpstreamError(
                    f"模型服务返回 {response.status_code}",
                    detail={"status": response.status_code, "body": body, "url": self.base_url},
                )

            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    log.debug("skipping malformed SSE payload", extra={"raw": data[:200]})
                    continue

                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                if isinstance(chunk.get("timings"), dict):
                    timings = chunk["timings"]

                for choice in chunk.get("choices") or ():
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        produced_any = True
                        yield TokenEvent("reasoning", str(reasoning))
                    content = delta.get("content")
                    if content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        produced_any = True
                        yield TokenEvent("text", str(content))
                    for call in delta.get("tool_calls") or ():
                        yield TokenEvent("tool", data={"tool_call": call})

            stats = self._build_stats(usage, timings, started, first_token_at)
            if finish_reason:
                stats.model = stats.model or self.model
            yield TokenEvent("usage", data=stats.to_public())
            yield TokenEvent("done", data={"finish_reason": finish_reason, "stats": stats.to_public()})
        except (BackendUnavailable, BackendTimeout, UpstreamError):
            raise
        except Exception as exc:  # noqa: BLE001
            if not produced_any:
                raise _translate(exc, self.base_url) from exc
            # Mid-stream break: report it as a terminal event rather than an
            # exception, so the caller can keep the partial answer it already has.
            log.warning("llm stream interrupted", extra={"error": str(exc)})
            yield TokenEvent("error", data={"message": f"生成中断：{exc}"})
            yield TokenEvent("done", data={"finish_reason": "interrupted"})
        finally:
            try:
                await response.aclose()
            except Exception:  # pragma: no cover
                pass

    # -- non-streaming -----------------------------------------------------------------
    async def complete(self, request: LLMRequest) -> Completion:
        client = await self._get_client()
        payload = self._payload(request, stream=False)
        started = time.perf_counter()

        async def _post() -> Any:
            return await client.post("/chat/completions", json=payload)

        try:
            response = await retry_async(_post, attempts=self.max_retries + 1)
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc, self.base_url) from exc

        if response.status_code >= 400:
            raise UpstreamError(
                f"模型服务返回 {response.status_code}",
                detail={"status": response.status_code, "body": response.text[:800]},
            )
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        stats = self._build_stats(body.get("usage") or {}, body.get("timings") or {}, started, None)
        message = Message(role=Role.ASSISTANT, content=text, meta={"model": self.model, "stats": stats.to_public()})
        return Completion(text=text, message=message, stats=stats)

    # -- diagnostics -------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        """Probe the server without burning GPU time on a generation."""
        if not _HAS_HTTPX:
            return {"ok": False, "name": self.name, "error": "httpx 未安装"}
        try:
            client = await self._get_client()
            started = time.perf_counter()
            response = await client.get("/models", timeout=min(self.connect_timeout_s, 8.0))
            latency = (time.perf_counter() - started) * 1000
            info: Dict[str, Any] = {
                "ok": response.status_code < 400,
                "name": self.name,
                "base_url": self.base_url,
                "latency_ms": round(latency, 1),
            }
            if response.status_code < 400:
                try:
                    data = response.json()
                    served = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
                    info["models"] = served
                    if served and self.model not in served:
                        info["ok"] = False
                        info["error"] = f"服务已启动但未加载模型 {self.model}（可用：{', '.join(served)}）"
                except Exception:  # pragma: no cover
                    pass
            else:
                info["error"] = f"HTTP {response.status_code}"
            return info
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "name": self.name, "base_url": self.base_url, "error": str(exc)}

    # -- internals ---------------------------------------------------------------------
    def _build_stats(
        self,
        usage: Dict[str, Any],
        timings: Dict[str, Any],
        started: float,
        first_token_at: Optional[float],
    ) -> LLMStats:
        stats = LLMStats(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            model=self.model,
        )
        if first_token_at is not None:
            stats.ttft_ms = round((first_token_at - started) * 1000, 1)
        stats.total_ms = round((time.perf_counter() - started) * 1000, 1)
        if timings:
            # llama.cpp reports its own speed counters; they are the ground truth
            # for deciding how many layers fit on the GPU.
            timings_out: Dict[str, Any] = {}
            for key in ("prompt_ms", "prompt_per_second", "predicted_ms", "predicted_per_second", "cache_n"):
                if key in timings:
                    timings_out[key] = timings[key]
            if timings.get("model"):
                stats.model = str(timings["model"])
            stats.timings = timings_out
        return stats


def _translate(exc: BaseException, base_url: str) -> Exception:
    """Map transport exceptions onto the project's error taxonomy."""
    text = str(exc)
    lowered = text.lower()
    if isinstance(exc, (asyncio.TimeoutError,)) or "timeout" in lowered or "timed out" in lowered:
        return BackendTimeout(
            "模型服务响应超时。若是首次请求，通常是在加载模型，请稍候重试。",
            detail={"base_url": base_url, "error": text},
        )
    if "connection refused" in lowered or "connect" in lowered or "econnrefused" in lowered:
        return BackendUnavailable(
            f"无法连接模型服务 {base_url}。请确认已运行 scripts/start-all.ps1 启动 llama-server。",
            detail={"base_url": base_url, "error": text},
        )
    return UpstreamError(f"调用模型服务失败：{text}", detail={"base_url": base_url})


__all__ = ["OpenAICompatBackend"]
