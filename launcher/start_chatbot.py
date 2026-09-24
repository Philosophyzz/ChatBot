"""双击启动：模型服务 + 网页界面 + 桌宠。

为什么要有这个 exe：日常使用不该记 PowerShell 命令。双击它，它会

  1. 找到项目根目录（exe 所在目录或其上一级，都可以）
  2. 检查后端是否已在运行 —— 已经在跑就直接进第 4 步，不会重复启动
  3. 调 scripts\\start-all.ps1 起模型服务与网页服务（输出原样显示，方便看进度）
  4. 等 /api/health 就绪，不自动打开浏览器
  5. 顺手把桌宠也拉起来（除非 -NoPet）

参数（给愿意用命令行的人；双击时全部走默认）：
  --no-pet        只起服务，不开桌宠
  --no-browser    兼容旧参数；现在所有启动方式都不自动开浏览器
  --mock          演示模式：不加载模型，秒开（界面功能完整）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 8077
HEALTH_URL = f"http://127.0.0.1:{PORT}/api/health"
APP_URL = f"http://127.0.0.1:{PORT}/"


def setup_console() -> None:
    """Windows 控制台默认代码页是 GBK，中文提示会显示成乱码。

    chcp 65001 把当前窗口切到 UTF-8，再把 stdout/stderr 固定成 UTF-8 ——
    双击运行时（真实控制台）和从管道捕获时都能正确显示中文。
    """
    if os.name == "nt":
        os.system("chcp 65001 >nul 2>&1")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass


def pause(message: str = "\n  按回车键退出…") -> None:
    """等用户看完再关窗口；没有控制台时（被重定向/管道）安静跳过而不是崩溃。"""

    try:
        input(message)  # noqa: S322 - 交互确认，不是安全输入
    except EOFError:
        pass


def find_root(explicit: str | None = None) -> Path | None:
    """项目根目录：exe 所在目录、其上一级、或同级配置文件里写明的路径。"""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    candidates += [here, here.parent, here.parent.parent, Path(r"D:\Harness\ChatBot")]
    for directory in candidates:
        try:
            if (directory / "scripts" / "start-all.ps1").is_file():
                return directory
        except OSError:
            continue
    return None


def port_open(host: str = "127.0.0.1", port: int = PORT, timeout_s: float = 1.5) -> bool:
    """服务是否在监听 —— 用 TCP 连接，而不是 HTTP 请求。

    两个理由：

    * socket 完全不经过代理。urllib 会读环境变量与注册表里的代理设置，这台机器上 Clash
      把连 127.0.0.1 的请求也拦了下来，结果是 ``start-all`` 已经打印"接口已就绪"，
      启动器却一直探测失败、白等到超时（这个 bug 实测踩到过）。
    * 我们只关心"端口通没通"，不需要解析 HTTP 响应体，中间层越少越可靠。
    """
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def backend_ready(timeout_s: float = 2.0) -> bool:
    return port_open(timeout_s=timeout_s)


def wait_ready(timeout_s: float = 240.0) -> bool:
    """Wait for the API to answer. The model server may still be loading weights."""
    deadline = time.time() + timeout_s
    dots = 0
    while time.time() < deadline:
        if backend_ready():
            return True
        dots += 1
        print(".", end="", flush=True)
        time.sleep(2.0)
    return False


def run_script(root: Path, script: str, *args: str) -> int:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(root / "scripts" / script),
        *args,
    ]
    print(f"  > {script} {' '.join(args)}".rstrip())
    print("")
    return subprocess.call(command)


def start_pet(root: Path) -> None:
    pet = root / "启动聊天机器人.exe"
    if not pet.is_file():
        print(f"  （没找到桌宠：{pet}，跳过）")
        return
    print("  正在打开桌宠…")
    try:
        subprocess.Popen([str(pet)], creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    except OSError as exc:
        print(f"  桌宠启动失败：{exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动本地聊天机器人")
    parser.add_argument("--root", default=None)
    parser.add_argument("--no-pet", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    setup_console()
    print("=" * 60)
    print("  本地聊天机器人 —— 启动")
    print("=" * 60)
    print("")

    root = find_root(args.root)
    if root is None:
        print("  [X] 找不到项目根目录（应该包含 scripts\\start-all.ps1）")
        print("      把本程序放回 D:\\Harness\\ChatBot\\ 下再运行，")
        print("      或用 --root D:\\Harness\\ChatBot 指定。")
        pause()
        return 2
    print(f"  项目目录：{root}")
    print("")

    if backend_ready():
        print("  [OK] 服务已经在运行，直接使用")
    else:
        print("  正在启动模型服务与网页服务（首次加载模型需要 10~60 秒）…")
        print("")
        script_args = ["-NoBrowser"]
        if args.mock:
            script_args.append("-Mock")
        code = run_script(root, "start-all.ps1", *script_args)
        if code != 0:
            print("")
            print(f"  [X] 启动脚本返回错误（exit {code}）")
            print(f"      日志：{root / 'logs' / 'api.err.log'}")
            print(f"            {root / 'logs' / 'llama-chat.log'}")
            pause()
            return code

        print("")
        print("  等待接口就绪", end="", flush=True)
        if not wait_ready():
            print("")
            print("  [!] 等待超时：服务可能还在加载模型，或启动失败")
            print(f"      看日志：{root / 'logs' / 'llama-chat.log'}")
            pause()
            return 1
        print("")
        print("  [OK] 服务已就绪")

    print("")
    if not args.no_pet:
        start_pet(root)

    print("")
    print("=" * 60)
    print("  可以开始聊天了")
    print(f"    网页界面：{APP_URL}")
    print("    桌宠    ：按住麦克风按钮说话；右键打开菜单")
    print(f"    停止服务：双击 桌宠右键 → 退出并停止全部服务（或 scripts\\stop.ps1）")
    print("=" * 60)
    print("")
    print("  本窗口可以直接关掉 —— 服务和桌宠会继续在后台运行。")
    time.sleep(6)  # let the user read the summary before the console closes
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
