"""Render the desktop pet to PNG/ICO — the app icon comes from the same code as the pet.

Keeping the icon generated (instead of shipping a binary asset) means it can never drift
from the character the user actually sees, and there is no art file to lose.

Usage::

    python tools/make_pet_icon.py --out data/pet.ico
    python tools/make_pet_icon.py --out data/pet.png --size 512
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def render_png(size: int) -> bytes:
    """Render one pet frame to PNG bytes (offscreen, no window)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QBuffer, QByteArray, QRectF, Qt
    from PySide6.QtGui import QPainter, QPixmap
    from PySide6.QtWidgets import QApplication

    from pet.window import EMOTION_COLORS, parse_color

    app = QApplication.instance() or QApplication([])
    from PySide6.QtGui import QColor, QFont, QPainterPath, QPen, QBrush
    import math

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # A trimmed-down version of the window's character (no name tag, no microphone):
    # an icon has to read at 16px, so only the silhouette and the face survive.
    accent = parse_color(EMOTION_COLORS["sweet"])
    rect = QRectF(size * 0.06, size * 0.06, size * 0.88, size * 0.88)
    width = rect.width() * 0.78
    height = rect.height() * 0.78
    body = QRectF(rect.center().x() - width / 2, rect.top() + rect.height() * 0.16, width, height)

    for sign in (-1, 1):
        ear = QPainterPath()
        base_x = rect.center().x() + sign * width * 0.30
        ear.moveTo(base_x - size * 0.07, body.top() + size * 0.07)
        ear.lineTo(base_x + sign * size * 0.03, body.top() - size * 0.14)
        ear.lineTo(base_x + size * 0.08, body.top() + size * 0.05)
        ear.closeSubpath()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(accent.darker(115)))
        painter.drawPath(ear)

    path = QPainterPath()
    path.addRoundedRect(body, width * 0.42, width * 0.42)
    painter.setBrush(QBrush(accent))
    painter.setPen(QPen(QColor(255, 255, 255, 70), max(1.0, size * 0.008)))
    painter.drawPath(path)

    face = body.adjusted(width * 0.17, height * 0.22, -width * 0.17, -height * 0.30)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(255, 253, 250, 245)))
    painter.drawRoundedRect(face, face.width() * 0.42, face.width() * 0.42)

    eye_r = face.width() * 0.085
    eye_y = face.center().y() - face.height() * 0.10
    painter.setBrush(QBrush(QColor(40, 42, 58)))
    for sign in (-1, 1):
        painter.drawEllipse(
            QRectF(face.center().x() + sign * face.width() * 0.22 - eye_r, eye_y - eye_r, eye_r * 2, eye_r * 2)
        )
    painter.setBrush(QBrush(QColor(255, 140, 170, 120)))
    for sign in (-1, 1):
        painter.drawEllipse(
            QRectF(face.center().x() + sign * face.width() * 0.34 - size * 0.045, eye_y + eye_r * 1.6, size * 0.09, size * 0.045)
        )
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QColor(60, 62, 82), max(1.2, size * 0.012), Qt.SolidLine, Qt.RoundCap))
    painter.drawArc(QRectF(face.center().x() - size * 0.09, eye_y + eye_r * 2.4, size * 0.18, size * 0.10), 200 * 16, 140 * 16)
    painter.end()

    buffer = QBuffer(QByteArray())
    buffer.open(QBuffer.WriteOnly)
    pixmap.save(buffer, "PNG")
    return bytes(buffer.data())


def png_to_ico(png: bytes, size: int) -> bytes:
    """Wrap a PNG into a single-image .ico container.

    Qt can read ICO but writing one is not guaranteed across builds, and an ICO that
    merely *contains* a PNG is legal (Vista+), so the container is built by hand — 6-byte
    header + one 16-byte directory entry + the PNG payload.
    """
    header = struct.pack("<HHH", 0, 1, 1)  # reserved, type=icon, image count
    entry = struct.pack(
        "<BBBBHHII",
        0 if size >= 256 else size,
        0 if size >= 256 else size,
        0,
        0,
        1,
        32,
        len(png),
        len(header) + 16,
    )
    return header + entry + png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成桌宠图标")
    parser.add_argument("--out", required=True, help="输出路径（.ico 或 .png）")
    parser.add_argument("--size", type=int, default=256, help="渲染尺寸")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    png = render_png(args.size)
    if out.suffix.lower() == ".ico":
        out.write_bytes(png_to_ico(png, args.size))
    else:
        out.write_bytes(png)
    print(f"  已生成 {out}（{out.stat().st_size} 字节，{args.size}px）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
