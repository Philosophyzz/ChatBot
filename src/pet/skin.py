"""Custom character images ("skins") for the desktop pet.

The pet draws its character with QPainter so it can ship without art assets — but the user
wants an anime girl, so the drawn character has to be replaceable. This module owns that:
import an image, normalise it once (square canvas, transparent background, sane size), store it
under ``data/pet_skins/``, and remember which persona uses it.

Normalising on import matters: a raw 4K JPEG would be re-decoded and re-scaled on *every*
repaint (the pet redraws at 30 fps), which would burn CPU for nothing. One 512×512 PNG per
persona is cheap to draw and looks identical at 210 px.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from core.logging import get_logger
from pet.settings import PetSettings, get_settings, project_root

log = get_logger(__name__)

SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
#: Rendered size of the character area is ~200 px, so 512 keeps it crisp on high-DPI screens.
DEFAULT_SIZE = 512


class SkinError(RuntimeError):
    """Readable failure (bad file, unsupported format, no Qt) for the UI to show."""


def skins_dir() -> Path:
    root = project_root()
    base = root / "data" / "pet_skins" if root else Path.cwd() / "pet_skins"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _require_qt():
    try:
        from PySide6.QtCore import Qt  # noqa: F401
        from PySide6.QtGui import QImage
    except Exception as exc:  # noqa: BLE001
        raise SkinError(f"需要 Qt 才能处理图片（PySide6 不可用）：{exc}") from exc
    return QImage


def install_skin(
    source: str | Path,
    persona_id: Optional[str] = None,
    *,
    size: int = DEFAULT_SIZE,
    settings: Optional[PetSettings] = None,
    keep_original: bool = False,
    directory: Optional[Path] = None,
) -> Path:
    """Copy an image in as this persona's character; returns the stored file.

    ``keep_original`` stores the untouched bytes (for a future animated/photo skin) instead of
    the normalised square PNG. ``directory`` overrides where the file lands — the tests pass a
    temp dir so they never touch the real ``data/pet_skins``.
    """
    QImage = _require_qt()
    src = Path(source)
    if not src.exists():
        raise SkinError(f"找不到图片：{src}")
    if src.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise SkinError(f"不支持的图片格式：{src.suffix}（支持 {'/'.join(SUPPORTED_SUFFIXES)}）")

    store = settings or get_settings()
    persona_key = persona_id or "default"
    target_dir = Path(directory) if directory else skins_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:10]

    if keep_original:
        target = target_dir / f"{persona_key}_{digest}{src.suffix.lower()}"
        shutil.copy2(src, target)
    else:
        image = QImage(str(src))
        if image.isNull():
            raise SkinError(f"读不出这张图片（可能已损坏）：{src}")
        normalised = normalise(image, QImage, size=size)
        target = target_dir / f"{persona_key}_{digest}.png"
        if not normalised.save(str(target), "PNG"):
            raise SkinError(f"写入失败：{target}")

    # Store a project-relative path when possible so the setting survives moving the folder.
    root = project_root()
    stored: str
    try:
        stored = str(target.relative_to(root)) if root else str(target)
    except ValueError:
        stored = str(target)
    store.set_skin(persona_id, stored)

    # Remove the previous image for this persona so the folder does not grow forever.
    for old in target_dir.glob(f"{persona_key}_*"):
        if old != target:
            try:
                old.unlink()
            except OSError:
                pass
    log.info("pet skin installed", extra={"persona": persona_key, "path": str(target)})
    return target


def normalise(image, QImage, *, size: int = DEFAULT_SIZE):
    """Scale to fit ``size`` and centre on a transparent square (keeps aspect ratio)."""
    from PySide6.QtCore import Qt

    image = image.convertToFormat(QImage.Format_ARGB32)
    scaled = image.scaled(
        size,
        size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
    canvas = QImage(size, size, QImage.Format_ARGB32)
    canvas.fill(0)  # fully transparent
    from PySide6.QtGui import QPainter

    painter = QPainter(canvas)
    painter.drawImage((size - scaled.width()) // 2, (size - scaled.height()) // 2, scaled)
    painter.end()
    return canvas


def clear_skin(persona_id: Optional[str] = None, *, settings: Optional[PetSettings] = None) -> List[Path]:
    """Forget this persona's custom image (and delete the file)."""
    store = settings or get_settings()
    removed: List[Path] = []
    path = store.skin_for(persona_id)
    store.set_skin(persona_id, None)
    if path:
        try:
            Path(path).unlink()
            removed.append(Path(path))
        except OSError:
            pass
    return removed


def list_skins(*, settings: Optional[PetSettings] = None) -> List[Tuple[str, str]]:
    """``[(persona_id, path), ...]`` for everything currently configured."""
    store = settings or get_settings()
    out: List[Tuple[str, str]] = []
    if store.get("default_skin"):
        resolved = store.skin_for(None)
        if resolved:
            out.append(("(所有人设)", resolved))
    for persona_id in sorted((store.get("skins") or {}).keys()):
        resolved = store.skin_for(persona_id)
        if resolved:
            out.append((persona_id, resolved))
    return out


def placeholder_image(size: int = DEFAULT_SIZE):
    """A generated sample image, for tests and ``--demo``."""
    QImage = _require_qt()
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath

    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    gradient = QLinearGradient(QPointF(0, 0), QPointF(size, size))
    gradient.setColorAt(0.0, QColor("#6ad4ff"))
    gradient.setColorAt(1.0, QColor("#ff8fb1"))
    painter.setBrush(QBrush(gradient))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(20, 20, size - 40, size - 40)
    path = QPainterPath()
    path.moveTo(size * 0.5, size * 0.15)
    path.lineTo(size * 0.85, size * 0.8)
    path.lineTo(size * 0.15, size * 0.8)
    path.closeSubpath()
    painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
    painter.drawPath(path)
    painter.end()
    return image
