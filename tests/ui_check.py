"""Real-browser UI check (not a pytest test: it needs Chrome and a running server).

Why this file exists: two bugs in this project were invisible to source-level review and
to the HTTP smoke test, because both were about what the browser actually *paints*:

* ``.modal { display: flex }`` beat the user-agent ``[hidden] { display: none }``, so
  ``element.hidden = true`` did nothing. The page opened with the 编辑人设 and 编辑记忆
  dialogs covering the whole app and 取消 could not dismiss them.
* the same cascade problem kept a red "正在录音…" banner under the composer.

Both were only ever confirmed by looking at a rendered screenshot. This script makes
that check repeatable: it drives headless Chrome over the DevTools protocol, asks for
*computed* styles (what is really on screen, not what the markup intends), clicks the
buttons, and reports pass/fail per assertion.

Usage::

    python tests/ui_check.py                    # default http://127.0.0.1:8077/
    python tests/ui_check.py --url http://127.0.0.1:8099/ --keep-open
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
)


def find_browser() -> Optional[str]:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ("chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def launch_browser(executable: str, port: int, profile_dir: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            executable,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            "--window-size=1400,900",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_for_target(port: int, timeout_s: float = 30.0) -> str:
    """Wait for the debugger *we* just launched.

    A fixed port is a trap: if a browser from a previous run is still shutting down, the
    new one silently fails to bind and the client happily attaches to the dying browser
    instead — every assertion then evaluates against about:blank and reports ``None``,
    which looks like eleven broken UI features rather than one broken test harness.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as response:
                targets = json.loads(response.read().decode("utf-8"))
            for target in targets:
                if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                    return target["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001 - browser still starting
            pass
        time.sleep(0.4)
    raise RuntimeError(f"浏览器调试端口 {port} 没有就绪")


class Session:
    """Minimal CDP client: enough to evaluate JS and take a screenshot."""

    def __init__(self, socket: Any) -> None:
        self._socket = socket
        self._id = 0

    async def send(self, method: str, **params: Any) -> Dict[str, Any]:
        self._id += 1
        message_id = self._id
        await self._socket.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            raw = json.loads(await self._socket.recv())
            if raw.get("id") == message_id:
                if "error" in raw:
                    raise RuntimeError(f"{method} 失败：{raw['error']}")
                return raw.get("result", {})

    async def js(self, expression: str) -> Any:
        result = await self.send(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
            awaitPromise=True,
        )
        return result.get("result", {}).get("value")


VISIBLE = "getComputedStyle(document.querySelector({sel})).display"


def build_checks() -> List[Tuple[str, str, Any]]:
    """(label, JS expression, expected value)."""
    return [
        ("页面加载后：编辑记忆弹窗不可见", VISIBLE.format(sel="'#modal-memory'"), "none"),
        ("页面加载后：编辑人设弹窗不可见", VISIBLE.format(sel="'#modal-persona'"), "none"),
        ("页面加载后：录音提示不可见", VISIBLE.format(sel="'#recording-hint'"), "none"),
        ("页面加载后：停止按钮不可见", VISIBLE.format(sel="'#btn-stop'"), "none"),
        (
            "打开记忆弹窗：可见",
            "openMemoryModal({id:'mem_check', content:'x', importance:0.5, confidence:0.5, tags:[]});"
            + VISIBLE.format(sel="'#modal-memory'"),
            "flex",
        ),
        (
            "点“取消”：弹窗关闭",
            "document.querySelector('[data-action=\"memory-cancel\"]').click();"
            + VISIBLE.format(sel="'#modal-memory'"),
            "none",
        ),
        (
            "打开后按 Esc：弹窗关闭",
            "openMemoryModal({id:'mem_check', content:'x', importance:0.5, confidence:0.5, tags:[]});"
            "document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}));"
            + VISIBLE.format(sel="'#modal-memory'"),
            "none",
        ),
        (
            "打开后点遮罩：弹窗关闭",
            "openMemoryModal({id:'mem_check', content:'x', importance:0.5, confidence:0.5, tags:[]});"
            "document.querySelector('#modal-memory').dispatchEvent(new MouseEvent('mousedown',{bubbles:true}));"
            + VISIBLE.format(sel="'#modal-memory'"),
            "none",
        ),
        (
            "打开人设弹窗再取消：关闭",
            "openPersonaModal(null);"
            "document.querySelector('[data-action=\"persona-cancel\"]').click();"
            + VISIBLE.format(sel="'#modal-persona'"),
            "none",
        ),
        (
            "默认人设是樱樱",
            "(document.querySelector('#persona-name')||{}).textContent",
            "樱樱",
        ),
        (
            "角色头像使用对应立绘",
            "document.querySelector('#persona-avatar img').getAttribute('src')",
            "/pet-assets/sakura_cat.png",
        ),
        (
            "系统提供本地与 API 模式",
            "Array.from(document.querySelector('#connection-mode').options).map(o=>o.value).join(',')",
            "local,api",
        ),
        (
            "编辑器保留角色初始记忆与专属形象",
            "openPersonaModal(currentPersona()); var valid=document.querySelector('#pf-memory').value.includes('樱樱') && document.querySelector('#pf-skin').value==='sakura_cat'; document.querySelector('[data-action=persona-cancel]').click(); valid",
            True,
        ),
        (
            "记忆面板顶部有“自动提取”说明",
            # The panel loads asynchronously, so poll instead of sampling once.
            "(async function(){document.querySelector('[data-tab=\"memory\"]').click();"
            "for (var i=0;i<25;i++){await new Promise(function(r){setTimeout(r,150)});"
            "if (document.querySelector('.memory-notice')) return true;} return false;})()",
            True,
        ),
        (
            "情绪页区分用户情绪与桌宠心情，并保留消息快照",
            "(async function(){await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},"
            "body:JSON.stringify({message:'我很难过',persona_id:state.personaId,stream:false})});"
            "state.memoryTab='emotion';await loadMemory();const t=document.querySelector('#memory-body').textContent;"
            "return t.includes('难过')&&t.includes('关心')&&t.includes('快照')&&t.includes('未估计');})()",
            True,
        ),
        (
            "朗读走流式接口并把音频排进队列",
            # Exercises the page's own speakStreaming(). Playback itself cannot be the
            # signal: headless Chrome blocks autoplay (no user gesture), so the audio is
            # shifted out of the queue and play() rejects. Counting frames as they are
            # handed to the player is what actually proves audio arrived.
            "(async function(){var frames=0;var orig=window.enqueueAudioUrl;"
            "window.enqueueAudioUrl=function(u,n){frames++;return orig(u,n);};"
            "var ok=null,err=null;"
            "try{ok=await speakStreaming('第一句先出声。第二句随后就到。',null);}"
            "catch(e){err=String(e);}finally{window.enqueueAudioUrl=orig;}"
            "for(var i=0;i<160&&frames===0;i++){await new Promise(function(r){setTimeout(r,250)});}"
            "return {ok:ok,frames:frames,err:err};})()",
            lambda value: isinstance(value, dict) and value.get("frames", 0) > 0,
        ),
        (
            "整段接口仍然可用（回退路径）",
            "(async function(){const r=await fetch('/api/voice/tts',{method:'POST',"
            "headers:{'Content-Type':'application/json'},"
            "body:JSON.stringify({text:'回退路径测试。',persona_id:'gentle_sister'})});"
            "if(!r.ok) return 'HTTP '+r.status; const b=await r.blob(); return b.size>1000;})()",
            True,
        ),
    ]


async def run(args: argparse.Namespace) -> int:
    try:
        from websockets.asyncio.client import connect
    except ImportError:  # websockets 12 remains supported by requirements.txt.
        from websockets.client import connect

    executable = args.browser or find_browser()
    if not executable:
        print("  找不到 Chrome/Edge，跳过浏览器检查")
        return 0

    profile = Path(tempfile.mkdtemp(prefix="chatbot-ui-"))
    process = launch_browser(executable, args.cdp_port, profile)
    failures: List[str] = []
    try:
        ws_url = wait_for_target(args.cdp_port)
        async with connect(ws_url, max_size=32 * 1024 * 1024) as socket:
            session = Session(socket)
            await session.send("Page.enable")
            await session.send("Runtime.enable")
            await session.send("Page.navigate", url=args.url)
            # Wait for *our* page: the app boots asynchronously (personas, sessions,
            # greeting), and a blank target must not be mistaken for a broken UI.
            loaded = False
            for _ in range(60):
                await asyncio.sleep(0.4)
                current = await session.js("location.href + '|' + document.readyState")
                if isinstance(current, str) and current.startswith(args.url) and current.endswith("complete"):
                    loaded = True
                    break
            if not loaded:
                print(f"  [FAIL] 页面没有加载成功（当前 {current!r}）")
                return 1
            for _ in range(40):
                await asyncio.sleep(0.3)
                if await session.js("!!document.querySelector('#persona-name')"):
                    break

            print(f"  目标：{args.url}")
            for label, expression, expected in build_checks():
                try:
                    actual = await session.js(expression)
                except Exception as exc:  # noqa: BLE001
                    actual = f"异常 {exc}"
                # ``expected`` may be a predicate when the useful diagnostic is a whole
                # object rather than a single value (it is printed on failure).
                if callable(expected):
                    ok = bool(expected(actual))
                else:
                    ok = actual == expected
                mark = "OK  " if ok else "FAIL"
                detail = "" if ok else f"（期望 {getattr(expected, '__name__', expected)!r}，实际 {actual!r}）"
                print(f"  [{mark}] {label}{detail}")
                if not ok:
                    failures.append(label)

            if args.screenshot:
                shot = await session.send("Page.captureScreenshot", format="png")
                import base64

                Path(args.screenshot).write_bytes(base64.b64decode(shot["data"]))
                print(f"  截图：{args.screenshot}")

            if not args.keep_open:
                await session.send("Browser.close")
    finally:
        if not args.keep_open:
            process.terminate()
        try:
            process.wait(timeout=10)
        except Exception:  # noqa: BLE001
            process.kill()
        shutil.rmtree(profile, ignore_errors=True)

    print("")
    if failures:
        print(f"  UI CHECK FAIL（{len(failures)} 项）：{', '.join(failures)}")
        return 1
    print("  UI CHECK PASS")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="用真实浏览器检查界面（计算样式 + 交互）")
    parser.add_argument("--url", default="http://127.0.0.1:8077/")
    parser.add_argument("--browser", default=None, help="浏览器可执行文件路径")
    parser.add_argument(
        "--cdp-port",
        type=int,
        default=0,
        help="调试端口；默认自动挑一个空闲端口（固定端口会和上一轮残留的浏览器打架）",
    )
    parser.add_argument("--screenshot", default=None, help="把截图写到这个路径")
    parser.add_argument("--keep-open", action="store_true", help="检查完不关闭浏览器")
    args = parser.parse_args(argv)
    if not args.cdp_port:
        args.cdp_port = free_port()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
