"""双击停止：模型服务 + 网页服务 + 桌宠，全部优雅退出。

调用 ``scripts\\stop.ps1``：它会先请求 ``POST /api/shutdown``，
让服务跑完 lifespan 收尾（关闭 SQLite 连接、合并 WAL），失败才强杀。
直接杀进程会留下未合并的 WAL 和 Windows 文件锁 —— 所以别用任务管理器。

桌宠是独立进程，会先收到关闭消息（Qt 正常退出），必要时才强制结束。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


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
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    candidates += [here, here.parent, here.parent.parent, Path(r"D:\Harness\ChatBot")]
    for directory in candidates:
        try:
            if (directory / "scripts" / "stop.ps1").is_file():
                return directory
        except OSError:
            continue
    return None


def stop_pet() -> int:
    """Close the pet politely first (Qt handles WM_CLOSE), then force if needed."""
    stopped = 0
    try:
        listing = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq ChatBotPet.exe", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
    except OSError:
        return 0
    if "ChatBotPet.exe" not in listing:
        return 0

    print("  关闭桌宠…")
    subprocess.run(["taskkill", "/IM", "ChatBotPet.exe"], capture_output=True)
    time.sleep(1.5)
    for _ in range(6):
        still = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq ChatBotPet.exe", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
        if "ChatBotPet.exe" not in still:
            print("    已关闭")
            return 1
        stopped += 1
        time.sleep(0.5)
    print("    还在运行，强制结束")
    subprocess.run(["taskkill", "/F", "/IM", "ChatBotPet.exe"], capture_output=True)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="停止本地聊天机器人")
    parser.add_argument("--root", default=None)
    parser.add_argument("--keep-pet", action="store_true", help="只停服务，留着桌宠")
    args = parser.parse_args(argv)

    setup_console()
    print("=" * 60)
    print("  本地聊天机器人 —— 停止")
    print("=" * 60)
    print("")

    root = find_root(args.root)
    if root is None:
        print("  [X] 找不到项目根目录（应该包含 scripts\\stop.ps1）")
        pause()
        return 2

    if not args.keep_pet:
        stop_pet()
        print("")

    print("  停止模型服务与网页服务（优雅退出，合并数据库 WAL）…")
    print("")
    code = subprocess.call(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / "scripts" / "stop.ps1"),
        ]
    )

    print("")
    if code == 0:
        print("  [OK] 已全部停止")
    else:
        print(f"  [!] 停止脚本返回 exit {code}，请检查是否有残留进程")
    print("")
    print("  下次使用：双击 dist\\启动聊天机器人.exe")
    time.sleep(6)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
