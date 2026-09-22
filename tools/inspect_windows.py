"""列出桌面上某个进程的顶层窗口（标题 / 类名 / 位置），用来核对桌宠窗口到底在哪。

为什么需要它：桌宠是**无边框置顶**窗口，`--selftest` 跑的是离屏渲染，只能证明"画出来了"，
证明不了"画在屏幕右下角"。气泡（SpeechBubble）更是另一个独立顶层窗口，标题为空，
只能靠位置和尺寸认出来 —— 曾经它就一直出现在屏幕正中间。

    python tools\\inspect_windows.py                 # 所有可见窗口
    python tools\\inspect_windows.py --process ChatBotPet
    python tools\\inspect_windows.py --process ChatBotPet --rects
    python tools\\inspect_windows.py --json logs\\windows.json

坐标注意：本工具是普通 Python 进程（**不感知 DPI**），所以 Windows 会把坐标换算成逻辑像素
返回 —— 正好和 Qt 的 `widget.x()` 同一套单位，可以直接比。但在**感知 DPI** 的进程里
（例如桌宠自己，Qt 会把它设成 per-monitor aware），`GetWindowRect` 返回的是物理像素，
在这个 175% 缩放的屏幕上两者差 1.75 倍，混着用就会得出"两只桌宠隔了 1500 像素"这种结论。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from ctypes import wintypes
from typing import Dict, List, Optional

if sys.platform != "win32":  # pragma: no cover - 桌宠本身只在 Windows 上跑
    print("这个工具只在 Windows 上有意义。")
    raise SystemExit(1)

# Windows 控制台默认是 GBK，窗口标题里常有 emoji / 特殊字符（例如 U+200B），直接 print
# 会 UnicodeEncodeError 把整个工具打挂 —— 明明查到了窗口却看不到结果。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 老 Python / 被重定向的流
        pass


def _titles_for_pids(pids: set) -> Dict[int, str]:
    """进程号 -> 进程名，用来按名字筛窗口。"""
    import subprocess

    names: Dict[int, str] = {}
    if not pids:
        return names
    try:
        out = subprocess.run(  # noqa: S603 - 固定命令，无用户输入
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
        ).stdout
    except Exception:  # noqa: BLE001
        return names
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 2 and parts[1].isdigit():
            names[int(parts[1])] = parts[0]
    return names


def windows(visible_only: bool = True) -> List[Dict[str, object]]:
    user32 = ctypes.windll.user32

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):  # noqa: ANN001
        if visible_only and not user32.IsWindowVisible(hwnd):
            return True
        buffer = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buffer, 512)
        klass = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, klass, 256)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        found.append(
            {
                "hwnd": int(hwnd),
                "pid": int(pid.value),
                "title": buffer.value,
                "class": klass.value,
                "x": rect.left,
                "y": rect.top,
                "w": rect.right - rect.left,
                "h": rect.bottom - rect.top,
            }
        )
        return True

    found: List[Dict[str, object]] = []
    user32.EnumWindows(callback, 0)
    names = _titles_for_pids({int(w["pid"]) for w in found})
    for window in found:
        window["process"] = names.get(int(window["pid"]), "")
    return found


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="列出桌面顶层窗口，核对桌宠/气泡位置")
    parser.add_argument("--process", default=None, help="只看进程名包含这段文字的窗口")
    parser.add_argument("--title", default=None, help="只看标题包含这段文字的窗口")
    parser.add_argument("--all", action="store_true", help="包含隐藏窗口")
    parser.add_argument("--rects", action="store_true", help="只打印 hwnd 和位置（方便脚本比对）")
    parser.add_argument("--json", default=None, help="把结果写到 JSON 文件")
    args = parser.parse_args(argv)

    items = windows(visible_only=not args.all)
    if args.process:
        items = [w for w in items if args.process.lower() in str(w["process"]).lower()]
    if args.title:
        items = [w for w in items if args.title in str(w["title"])]

    if args.rects:
        for window in items:
            print(f"{window['hwnd']}\t{window['x']},{window['y']}\t{window['w']}x{window['h']}\t{window['title']}")
    else:
        print(f"共 {len(items)} 个窗口")
        for window in items:
            print(
                f"  pid={window['pid']:<7} {str(window['process'])[:18]:<18} "
                f"{window['x']:>5},{window['y']:>5} {window['w']:>5}x{window['h']:<5} "
                f"[{window['class']}] {window['title']!r}"
            )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(items, handle, ensure_ascii=False, indent=2)
        print(f"已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
