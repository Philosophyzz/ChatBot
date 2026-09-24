"""启动 GPT-SoVITS 上游 WebUI，并补一个 starlette 兼容层。

为什么需要这个文件：上游 WebUI 用的是 gradio 4.44，而本项目环境里的
**starlette 1.6** 改了 ``TemplateResponse`` 的签名：

    旧（gradio 4.44 的调用方式）  TemplateResponse(name, {"request": request, ...})
    新（starlette ≥ 0.42）        TemplateResponse(request, name, context)

于是首参 ``name``（字符串）被当成 request、context（dict）被当成模板名，jinja2 直接抛
``TypeError: unhashable type: 'dict'``，页面 500 —— 服务在监听，但打不开。

可选的三条路：
  1. 降级 starlette —— 会影响本项目自己的 FastAPI 0.141（不划算）；
  2. 把 WebUI 装进独立环境 —— 又要一份 torch，4GB 起；
  3. **在这里加一层兼容**（本文件）：只影响这个进程，主程序一行不动。

这个文件只做两件事：打补丁，然后用 ``__main__`` 语义跑上游的 webui.py。
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1] / "vendor" / "GPT-SoVITS"
WEBUI = REPO / "webui.py"


def patch_starlette_template_response() -> bool:
    """让 starlette 的新签名也接受 gradio 4.x 的旧调用方式。返回是否打了补丁。"""
    try:
        from starlette.templating import Jinja2Templates
    except Exception as exc:  # noqa: BLE001
        print(f"[shim] 无法导入 starlette.templating：{exc}")
        return False

    original = Jinja2Templates.TemplateResponse
    if getattr(original, "_gsv_compat", False):
        return True

    seen: set = set()

    def TemplateResponse(self: Any, *args: Any, **kwargs: Any):  # noqa: N802 - 对齐上游命名
        """按**类型**认参数，而不是按位置猜 —— 新旧两种调用顺序都能接。

        旧：TemplateResponse(name: str, context: dict)
        新：TemplateResponse(request: Request, name: str, context: dict | None)
        """
        request = kwargs.pop("request", None)
        name = kwargs.pop("name", None)
        context = kwargs.pop("context", None)
        for value in args:
            if isinstance(value, str) and name is None:
                name = value
            elif hasattr(value, "scope") and request is None:  # starlette 的 Request
                request = value
            elif isinstance(value, dict) and context is None:
                context = value
        context = dict(context or {})
        if request is None:
            request = context.get("request")

        if name is None or request is None:
            # 认不出来就原样交回，让它自己报清晰的错，别在这里吞掉
            if "unrecognized" not in seen:
                seen.add("unrecognized")
                print(f"[shim] 无法识别的 TemplateResponse 调用：args={args!r} kwargs={kwargs!r}")
            return original(self, *args, **kwargs)

        if "applied" not in seen:
            seen.add("applied")
            print(f"[shim] 兼容层命中：template={name!r}（按类型识别参数）")
        return original(self, request, name, context, **kwargs)

    TemplateResponse._gsv_compat = True  # type: ignore[attr-defined]
    Jinja2Templates.TemplateResponse = TemplateResponse  # type: ignore[assignment]
    # 如果 gradio 的 templates 对象自己也挂了同名属性，那是个**实例属性**：属性上放普通函数
    # 时 Python 不会自动绑定 self（踩过：函数收到的第一个参数其实是模板名）。所以这里显式
    # 绑定，别再让它自己去猜。
    try:
        import gradio.routes as gradio_routes

        templates = getattr(gradio_routes, "templates", None)
        if templates is not None:
            def _bound(*args: Any, **kwargs: Any):  # noqa: ANN202
                return TemplateResponse(templates, *args, **kwargs)

            _bound._gsv_compat = True  # type: ignore[attr-defined]
            templates.TemplateResponse = _bound  # type: ignore[assignment]
    except Exception:  # noqa: BLE001 - gradio 尚未导入也无妨
        pass
    print("[shim] 已给 starlette 的 TemplateResponse 加上旧签名兼容（gradio 4.x 需要）")
    return True


def patch_gradio_localhost_check() -> None:
    """让 gradio 的自检别被系统代理坑了。

    gradio 启动时会请求 ``http://127.0.0.1:<port>/`` 确认自己能连上；本机 Clash 在
    7890，如果 HTTP_PROXY 之类还留在环境里，这个请求会被代理接走并失败，于是 gradio
    直接拒绝启动：``ValueError: When localhost is not accessible, a shareable link must
    be created``。本项目所有本机调用都用同一套办法解决（见 speech/llm 客户端的
    ``trust_env=False``）：本机地址不走代理。
    """
    import os

    cleared = [key for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy") if os.environ.pop(key, None)]
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
    os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])
    if cleared:
        print(f"[shim] 已清掉代理环境变量 {'、'.join(cleared)}（本机地址不走代理）")


def main() -> int:
    if not WEBUI.exists():
        print(f"找不到上游 WebUI：{WEBUI}")
        print("先运行：powershell -ExecutionPolicy Bypass -File scripts\\install-gptsovits.ps1")
        return 1

    patch_starlette_template_response()
    patch_gradio_localhost_check()

    language = sys.argv[-1] if len(sys.argv) > 1 else "zh_CN"
    print(f"[shim] 启动 {WEBUI.name}（语言 {language}，工作目录 {Path.cwd()}）")
    sys.argv = [str(WEBUI), language]
    runpy.run_path(str(WEBUI), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
