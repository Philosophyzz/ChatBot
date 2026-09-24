"""Validated local/API connection profiles shared by the pet and web settings."""
from urllib.parse import urlsplit

from core.errors import BadRequest


def managed_local(config):
    parsed = urlsplit(config.base_url)
    return config.mode == "local" and parsed.hostname in {"localhost", "127.0.0.1", "::1"} and parsed.port == 8080


def validate_connection(values):
    mode = values.get("mode", "local")
    url = str(values.get("base_url", "")).strip().rstrip("/")
    model = str(values.get("model", "")).strip()
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise BadRequest("模型地址格式不正确") from exc
    if mode not in {"local", "api"}:
        raise BadRequest("调用模式必须为 local 或 api")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BadRequest("请填写不含密钥、查询参数的 HTTP(S) 接口根地址，例如 http://127.0.0.1:8080/v1")
    if not model or len(model) > 200:
        raise BadRequest("请填写模型名称（最多 200 字符）")
    backend = "openai"
    if mode == "local" and parsed.hostname in {"localhost", "127.0.0.1", "::1"} and port == 8080:
        backend = str(values.get("backend") or "vllm")
        if backend not in {"vllm", "llamacpp", "openai"}:
            raise BadRequest("不支持该本地推理后端")
    return {"mode": mode, "backend": backend,
            "base_url": url, "model": model, "mock": False,
            "api_key": str(values.get("api_key") or ("sk-local" if mode == "local" else ""))}
