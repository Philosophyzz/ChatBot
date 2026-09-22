"""截一张桌面图（可选只截右下角一块），用来肉眼核对桌宠/气泡的位置。

    python tools\\grab_screen.py --out logs\\desktop.png
    python tools\\grab_screen.py --out logs\\corner.png --corner 760x560

截图是给别人（或视觉模型）看的证据：桌宠是无边框窗口，`--selftest` 只能证明它画出来了，
证明不了它有没有跑到屏幕正中间 —— 那正是气泡出过的 bug。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="截取桌面，用于检查桌宠窗口位置")
    parser.add_argument("--out", default="logs/desktop.png", help="输出 PNG 路径")
    parser.add_argument("--screen", type=int, default=0, help="第几块屏幕（0 = 主屏）")
    parser.add_argument("--corner", default=None, help="只截右下角，格式 WxH（如 760x560）")
    parser.add_argument("--delay", type=float, default=0.5, help="截图前等几秒，让窗口画完")
    args = parser.parse_args(argv)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    screens = QApplication.screens()
    if not screens:
        print("没有找到屏幕（是不是跑在无桌面会话里？）")
        return 1
    screen = screens[min(args.screen, len(screens) - 1)]

    def capture() -> None:
        pixmap = screen.grabWindow(0)  # 0 = 整个屏幕
        if args.corner:
            try:
                width, height = (int(v) for v in args.corner.lower().split("x"))
            except ValueError:
                print(f"--corner 格式应为 WxH，收到 {args.corner!r}")
                QApplication.quit()
                return
            import math

            # 屏幕坐标与设备像素可能不一致（缩放），按比例换算，否则截出来的是屏幕中间。
            ratio = pixmap.width() / max(1, screen.geometry().width())
            width_px = int(width * ratio)
            height_px = int(height * ratio)
            pixmap = pixmap.copy(
                pixmap.width() - width_px,
                pixmap.height() - height_px,
                width_px,
                height_px,
            )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not pixmap.save(str(out)):
            print(f"保存失败：{out}")
            QApplication.quit()
            return
        geometry = screen.geometry()
        print(f"已保存 {out}（{pixmap.width()}x{pixmap.height()}，屏幕 {geometry.width()}x{geometry.height()}）")
        QApplication.quit()

    QTimer.singleShot(int(max(0.0, args.delay) * 1000), capture)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
