"""End-to-end HTTP smoke test: boot the real ASGI app and exercise every endpoint.

``scripts/selftest.ps1`` used to embed this as a ``python -c`` one-liner. That is
unreliable on Windows: the command line is passed through the ANSI code page, so the
Chinese characters in the request bodies arrived corrupted and Python raised
SyntaxError — the same class of failure that already broke ``install.ps1`` and
``verify.ps1``. Keeping it in a real file removes the whole problem, and makes the
checks readable and individually assertable.

What it proves, with a real uvicorn server on a real port:

* ``/api/health`` responds and reports the components
* ``/api/personas`` lists the catalogue
* ``/api/chat`` answers a Chinese message (through the mock model, so no GPU needed)
* ``/api/memory/stats`` responds
* ``/api/voice/tts`` returns genuine audio bytes — either WAV (``RIFF``) or MP3, but
  never the 46-byte header-only container that a broken backend produces

Usage::

    python tests/smoke_http.py            # picks a free port, exits 0/1
    python tests/smoke_http.py --port 8099
"""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import load_config  # noqa: E402


def cleanup_scratch(scratch: tempfile.TemporaryDirectory, attempts: int = 8) -> None:
    """Delete the scratch data dir, tolerating Windows file locks.

    The server is asked to exit, but uvicorn's own teardown may still hold the SQLite
    handle for a moment; on Windows that turns ``cleanup()`` into ``PermissionError:
    [WinError 32]`` and kills the script *after* the checks have passed. A failed
    cleanup is cosmetic — say so and move on instead of failing the smoke test.
    """
    for attempt in range(attempts):
        try:
            scratch.cleanup()
            return
        except (PermissionError, OSError):
            time.sleep(0.4)
    print(f"  提示：临时目录暂时删不掉（{scratch.name}），可稍后手动清理")


def free_port(preferred: int) -> int:
    """Return ``preferred`` if it is free, otherwise an ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def describe_audio(data: bytes) -> Tuple[bool, str]:
    """Classify a TTS payload. Returns ``(ok, description)``."""
    if len(data) <= 44:
        return False, f"过短（{len(data)} 字节）—— 可能是仅含文件头的空音频"
    if data[:4] == b"RIFF":
        import struct

        data_size = struct.unpack_from("<I", data, 40)[0]
        return (data_size > 0), f"WAV data块={data_size} 字节"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return True, f"MP3 {len(data)} 字节"
    return True, f"未知容器但非空（{len(data)} 字节，前 4 字节 {data[:4]!r}）"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动真实服务并做端到端接口冒烟")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args(argv)

    port = free_port(args.port)
    base = f"http://127.0.0.1:{port}"

    config = load_config(ROOT)
    config.llm.mock = True                    # no GPU, no model weights needed
    config.server.port = port
    config.memory.embedder = "hash"           # do not probe the real embedding server
    config.memory.reranker = "heuristic"
    config.memory.extract_async = False
    config.memory.consolidate_enabled = False
    config.speech.tts_preference = ["tone"]   # deterministic, offline, no network
    config.debug = True

    # Throwaway data directory.
    #
    # This script chats ("我叫小明，我喜欢不加糖的美式咖啡") to exercise extraction, so
    # pointing it at the real ``data/`` plants those sentences in the user's permanent
    # memory — the assistant then confidently tells its owner their name is 小明. Every
    # run of ``selftest.ps1`` used to do exactly that. Only the database location moves;
    # config, personas and models still come from the project so the smoke stays real.
    scratch = tempfile.TemporaryDirectory(prefix="chatbot-smoke-")
    config.paths = replace(config.paths, data_dir=Path(scratch.name))
    print(f"  临时数据目录（真实记忆库不受影响）：{scratch.name}")

    from api.server import create_asgi_app
    from core.app import ChatBotApp
    import uvicorn

    asgi = create_asgi_app(ChatBotApp(config))
    server = uvicorn.Server(uvicorn.Config(asgi, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    started = False
    for _ in range(120):
        time.sleep(0.5)
        if getattr(server, "started", False):
            started = True
            break
    if not started:
        print("  [X] 服务未能在 60 秒内启动")
        return 1

    failures: List[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        mark = "OK  " if condition else "FAIL"
        print(f"  [{mark}] {label}{('  ' + detail) if detail else ''}")
        if not condition:
            failures.append(label)

    def request_json(path: str, payload: Dict[str, Any] | None = None, timeout: int = 120):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{base}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET"
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def request_bytes(path: str, payload: Dict[str, Any], timeout: int = 90) -> bytes:
        request = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    print(f"  服务已在 {base} 启动（模拟模型）")

    try:
        status, health = request_json("/api/health")
        check("/api/health", status == 200 and health.get("ok") is True, f"llm={health.get('llm', {}).get('name')}")

        status, personas = request_json("/api/personas")
        count = len(personas.get("personas") or [])
        check("/api/personas", status == 200 and count >= 4, f"{count} 个人设")

        status, chat = request_json("/api/chat", {"message": "你好，我叫小明", "stream": False})
        answer = (chat.get("text") or "").strip()
        check("/api/chat", status == 200 and bool(answer), repr(answer[:36]))

        status, stats = request_json("/api/memory/stats")
        check("/api/memory/stats", status == 200, f"记忆={stats.get('store', {}).get('memories')}")

        audio = request_bytes("/api/voice/tts", {"text": "测试语音合成"})
        ok, detail = describe_audio(audio)
        check("/api/voice/tts", ok, detail)

        status, _single = request_json("/api/memory/items")
        check("/api/memory/items", status == 200)

        # A missing session must be a clean 404, not a 500.
        try:
            urllib.request.urlopen(f"{base}/api/sessions/does-not-exist", timeout=15)
            check("未知会话返回 404", False, "居然返回了 200")
        except urllib.error.HTTPError as exc:
            check("未知会话返回 404", exc.code == 404, f"HTTP {exc.code}")

        # Chat history search must not blow up on an empty store.
        # NOTE: the query string must be percent-encoded. Passing raw Chinese in a URL
        # raises "UnicodeEncodeError: 'ascii' codec can't encode characters" from
        # urllib, which has nothing to do with the server.
        status, search = request_json(f"/api/memory/search?q={quote('咖啡')}")
        check("/api/memory/search", status == 200, f"命中 {len(search.get('memories') or [])} 条")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] 冒烟过程中出现异常：{type(exc).__name__}: {exc}")
        failures.append("未处理异常")
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        cleanup_scratch(scratch)

    print("")
    if failures:
        print(f"  SMOKE FAIL（{len(failures)} 项）：{', '.join(failures)}")
        return 1
    print("  SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
