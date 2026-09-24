"""GPT-SoVITS 上游 WebUI 的兼容层测试。

为什么要测一个"启动器"里的补丁：上游 WebUI 用 gradio 4.44，而本项目环境里是 starlette 1.6，
``TemplateResponse`` 的签名在两代之间反了：旧 ``(name, context)``、新 ``(request, name, context)``。
结果是**端口在监听、页面却 500**（jinja2 收到 dict 当模板名）。这个补丁不生效的话，
用户看到的就是一个打不开的"可视化训练页面"。

另外还有一条：gradio 启动时会自检"能不能连上自己的 localhost"，而本机 Clash 在 7890，
代理在场时自检失败，gradio 直接拒绝启动。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_shim():
    spec = importlib.util.spec_from_file_location("gsv_webui_shim", ROOT / "tools" / "gsv_webui_shim.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def templates(tmp_path):
    from starlette.templating import Jinja2Templates

    (tmp_path / "hello.html").write_text("version={{ config.version }}", encoding="utf-8")
    return Jinja2Templates(directory=str(tmp_path))


def _request():
    from starlette.requests import Request

    return Request({"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""})


def test_legacy_gradio_call_now_works(templates) -> None:
    """gradio 4.x 的调用方式：(模板名, context) —— 这正是页面 500 的原因。"""
    shim = _load_shim()
    assert shim.patch_starlette_template_response() is True

    response = templates.TemplateResponse(
        "hello.html", {"request": _request(), "config": {"version": "v2ProPlus"}}
    )
    assert response.status_code == 200
    assert b"version=v2ProPlus" in response.body


def test_new_starlette_call_still_works(templates) -> None:
    """补丁不能把新签名弄坏（本项目自己的 FastAPI 用的是新签名）。"""
    shim = _load_shim()
    shim.patch_starlette_template_response()

    response = templates.TemplateResponse(_request(), "hello.html", {"config": {"version": "v2"}})
    assert response.status_code == 200
    assert b"version=v2" in response.body


def test_patching_twice_is_harmless(templates) -> None:
    shim = _load_shim()
    assert shim.patch_starlette_template_response() is True
    assert shim.patch_starlette_template_response() is True
    response = templates.TemplateResponse("hello.html", {"request": _request(), "config": {"version": "ok"}})
    assert b"version=ok" in response.body


def test_proxy_env_is_cleared_for_localhost(monkeypatch) -> None:
    """代理在场时 gradio 会以为 localhost 不可达，直接拒绝启动。"""
    shim = _load_shim()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.delenv("NO_PROXY", raising=False)

    shim.patch_gradio_localhost_check()

    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    assert "127.0.0.1" in os.environ.get("NO_PROXY", "")
