"""ASGI server assembly.

Run it either way — the module makes both work by putting ``src/`` on ``sys.path``
itself, so the working directory does not matter::

    python -m api.server --port 8077          # from src/
    python src\\api\\server.py --port 8077      # from the project root

or through ``scripts/start-all.ps1``, which also starts the model server.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Optional

# --- import bootstrap -----------------------------------------------------------------
# The application's top-level packages (``core``, ``memory``, ``api``…) live in ``src``,
# not at the project root. Without this, ``python src/api/server.py`` from the project
# root fails with "No module named 'core'" and ``python -m api.server`` from the root
# fails with "No module named 'api'" — both confusing, neither the user's fault.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.app import ChatBotApp  # noqa: E402
from core.config import load_config  # noqa: E402
from core.logging import get_logger  # noqa: E402

log = get_logger(__name__)

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "FastAPI 未安装，请先运行 scripts/install.ps1（或 pip install fastapi uvicorn）"
    ) from exc


def create_asgi_app(app: Optional[ChatBotApp] = None) -> FastAPI:
    """Build the FastAPI application around a :class:`ChatBotApp`."""
    chatbot = app or ChatBotApp(load_config())

    @contextlib.asynccontextmanager
    async def lifespan(instance: FastAPI) -> AsyncIterator[None]:
        await chatbot.start()
        instance.state.chatbot = chatbot
        try:
            yield
        finally:
            await chatbot.stop()

    api = FastAPI(
        title="本地 LLM 聊天机器人",
        version="0.1.0",
        description="本地大模型 + 长期记忆 + 语音输入输出的个人助手",
        lifespan=lifespan,
    )
    api.state.chatbot = chatbot

    api.add_middleware(
        CORSMiddleware,
        allow_origins=chatbot.config.server.cors_origins + (["*"] if chatbot.config.server.allow_lan else []),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-TTS-Chunks", "X-TTS-Backend"],
    )

    from api.routes import router

    api.include_router(router)

    @api.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        log.exception("unhandled error", extra={"path": str(request.url.path)})
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal", "message": str(exc)[:400]}},
        )

    web_root = Path(chatbot.config.paths.web_dir)
    if chatbot.config.server.serve_static and web_root.exists():
        assets = web_root / "assets"
        if assets.exists():
            # ``no-cache`` means "revalidate before reuse", not "do not cache": the ETag
            # still spares the transfer, but a restarted server or an edited app.js can
            # never be masked by a stale browser copy. Without any Cache-Control at all
            # browsers apply heuristic freshness, and a user then keeps seeing
            # yesterday's UI while the backend has already changed — indistinguishable,
            # from their side, from "the fix did not work".
            api.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

            @api.middleware("http")
            async def _shell_revalidation(request: Request, call_next: Any) -> Any:
                response = await call_next(request)
                if request.url.path == "/" or request.url.path.startswith("/assets/"):
                    response.headers.setdefault("Cache-Control", _SHELL_CACHE_CONTROL)
                return response

        @api.get("/", include_in_schema=False)
        async def index() -> Any:
            # Registered only when the file exists: FileResponse would otherwise raise
            # "File at path ... does not exist" *after* routing, surfacing as a 500
            # with a traceback that says nothing about a missing web build.
            if not (web_root / "index.html").exists():
                return Response(status_code=404)
            return FileResponse(
                str(web_root / "index.html"), headers={"Cache-Control": _SHELL_CACHE_CONTROL}
            )

        @api.get("/favicon.ico", include_in_schema=False)
        async def favicon() -> Any:
            """Serve the site icon, or an empty 204 when there is no file.

            Returning ``JSONResponse(status_code=204, content=None)`` here was a bug:
            204 means "no content", but the response still carried a body, so uvicorn
            rejected it with "Response content longer than Content-Length". Every page
            load triggered it (browsers request /favicon.ico automatically) and the
            traceback in logs/api.err.log pointed at an unrelated-looking failure.
            ``Response(status_code=204)`` is the correct empty reply.
            """
            icon = web_root / "favicon.ico"
            if icon.exists():
                return FileResponse(str(icon))
            # Inline SVG keeps the deployment to a single file with no binary asset.
            return Response(
                content=_FAVICON_SVG,
                media_type="image/svg+xml",
                headers={"Cache-Control": "public, max-age=86400"},
            )

        @api.get("/favicon.svg", include_in_schema=False)
        async def favicon_svg() -> Any:
            return Response(
                content=_FAVICON_SVG,
                media_type="image/svg+xml",
                headers={"Cache-Control": "public, max-age=86400"},
            )

    return api


#: Cache policy for the HTML shell and its JS/CSS (see the note inside ``create_app``).
_SHELL_CACHE_CONTROL = "no-cache, must-revalidate"

#: A tiny speech-bubble mark; inlined so there is no binary asset to ship or 404.
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<rect width="64" height="64" rx="14" fill="#7c9cff"/>'
    '<path d="M16 22h32a4 4 0 0 1 4 4v14a4 4 0 0 1-4 4H30l-9 8v-8h-5a4 4 0 0 1-4-4V26a4 4 0 0 1 4-4z" fill="#12131a"/>'
    '<circle cx="26" cy="33" r="3" fill="#7c9cff"/><circle cx="34" cy="33" r="3" fill="#7c9cff"/>'
    '<circle cx="42" cy="33" r="3" fill="#7c9cff"/></svg>'
)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="启动本地聊天机器人服务")
    parser.add_argument("--host", default=None, help="监听地址（默认取配置）")
    parser.add_argument("--port", type=int, default=None, help="监听端口（默认取配置）")
    parser.add_argument("--root", default=None, help="项目根目录（默认自动推断）")
    parser.add_argument("--mock", action="store_true", help="使用内置模拟模型，不连接本地大模型")
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更自动重启")
    parser.add_argument("--log-level", default=None, help="日志级别")
    args = parser.parse_args(argv)

    config = load_config(Path(args.root) if args.root else None)
    if args.log_level:
        config.log_level = args.log_level
    host = args.host or config.server.host
    port = args.port or config.server.port
    if args.mock:
        config.llm.mock = True

    uvicorn = _import_uvicorn()
    if args.reload:
        # Reload mode must re-exec the module: construct the app from an import string.
        os.environ["CHATBOT_MOCK"] = "1" if config.llm.mock else "0"
        uvicorn.run(
            "api.server:build_from_env",
            host=host,
            port=port,
            reload=True,
            log_level=config.log_level.lower(),
            app_dir=str(config.paths.root / "src"),
        )
        return 0

    app = create_asgi_app(ChatBotApp(config))
    log.info("starting server", extra={"host": host, "port": port, "mock": config.llm.mock})
    # An explicit Server object (rather than ``uvicorn.run``) so ``POST /api/shutdown``
    # can flip ``should_exit``: that path runs the lifespan teardown, which closes the
    # SQLite connections and checkpoints the WAL. ``Stop-Process -Force`` cannot.
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level=config.log_level.lower(), access_log=False)
    )
    app.state.uvicorn_server = server
    server.run()
    return 0


def build_from_env() -> FastAPI:
    """Factory used by ``uvicorn --reload`` (needs an import string, not an object)."""
    config = load_config()
    config.llm.mock = os.environ.get("CHATBOT_MOCK") == "1"
    return create_asgi_app(ChatBotApp(config))


def _import_uvicorn() -> Any:
    try:
        import uvicorn  # type: ignore

        return uvicorn
    except Exception as exc:  # pragma: no cover
        raise ImportError("uvicorn 未安装，请运行 scripts/install.ps1") from exc


if __name__ == "__main__":
    sys.exit(main())
