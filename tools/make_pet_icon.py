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

    from pet.skin import builtin_skin_path
    app = QApplication.instance() or QApplication([])
    from core.config import load_config
    path = builtin_skin_path(load_config().default_persona) or builtin_skin_path("hiyori")
    if path is None:
        raise RuntimeError("Missing default character artwork")
    source = QPixmap(str(path))
    pixmap = source.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
