"""读出一个卡住的对话框里的文字 —— 专治"exe 双击没反应，任务管理器里还活着"。

打包成 `--windowed` 的 exe 崩在启动阶段时（缺 DLL、缺模块），PyInstaller 和 Qt 都会弹一个
**模态错误框**，然后一直等人点确定。构建脚本、自动化脚本这时候就会卡死在 `-Wait` 上，
而屏幕上那个框里其实写着真正的原因。

    python tools\\read_dialog.py                     # 默认找标题含 Unhandled exception 的框
    python tools\\read_dialog.py 桌宠启动失败          # 按标题关键字找
    python tools\\read_dialog.py --list              # 先看看屏幕上都有哪些对话框

只读，不点任何按钮。
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from ctypes import wintypes
from typing import List, Optional

if sys.platform != "win32":  # pragma: no cover - 桌宠本身只在 Windows 上跑
    print("这个工具只在 Windows 上有意义。")
    raise SystemExit(1)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 被重定向的流
        pass


def _visible_windows(needle: Optional[str]) -> List[int]:
    """标题匹配（None = 全部）的可见顶层窗口句柄。"""
    user32 = ctypes.windll.user32
    found: List[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):  # noqa: ANN001
        if not user32.IsWindowVisible(hwnd):
            return True
        buffer = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buffer, 512)
        if not buffer.value.strip():
            return True
        if needle is None or needle.lower() in buffer.value.lower():
            found.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return found


def _texts(hwnd: int) -> List[str]:
    """对话框里所有子控件的文字（正文 + 按钮）。"""
    user32 = ctypes.windll.user32
    parts: List[str] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def child_callback(child, _lparam):  # noqa: ANN001
        buffer = ctypes.create_unicode_buffer(16384)
        user32.GetWindowTextW(child, buffer, 16384)
        if buffer.value.strip():
            parts.append(buffer.value)
        return True

    user32.EnumChildWindows(hwnd, child_callback, 0)
    return parts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="读出卡住的对话框里的文字")
    parser.add_argument("needle", nargs="?", default="Unhandled exception", help="标题关键字")
    parser.add_argument("--list", action="store_true", help="只列出屏幕上的对话框标题")
    args = parser.parse_args(argv)

    if args.list:
        handles = _visible_windows(None)
        print(f"屏幕上共有 {len(handles)} 个有标题的窗口：")
        for hwnd in handles:
            titles = _texts(hwnd)
            print(f"  hwnd={hwnd}  {titles[0][:90] if titles else '(无文字)'}")
        return 0

    handles = _visible_windows(args.needle)
    print(f"匹配「{args.needle}」的窗口：{len(handles)} 个")
    for hwnd in handles:
        print(f"--- hwnd={hwnd} ---")
        for text in _texts(hwnd):
            print(text)
    if not handles:
        print("（没有找到。换个关键字，或先 --list 看看屏幕上都有什么）")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
