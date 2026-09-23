"""Bundled artwork must survive switching, clearing and offline rendering."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pet.client import BackendClient
from pet.settings import PetSettings
from pet.skin import BUILTIN_SKINS, builtin_skin_path, clear_skin
from pet.window import PET_SIZE, PetWindow


@pytest.fixture
def pet(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = PetSettings(tmp_path / "settings.json")
    monkeypatch.setattr("pet.window.get_settings", lambda: settings)
    # Rendering and changing clothes must work without an LLM or audio service.
    monkeypatch.setattr(PetWindow, "refresh_personas", lambda self: None)
    monkeypatch.setattr(PetWindow, "refresh_models", lambda self: None)
    window = PetWindow(BackendClient("http://127.0.0.1:1"), persona_id="test", with_tray=False)
    window._anim.stop()
    yield window, app
    window.bubble.close()
    window.close()
    window.client.close()


@pytest.mark.parametrize("skin_id,label", BUILTIN_SKINS)
def test_bundled_skin_menu_persistence_and_render(pet, skin_id, label):
    window, app = pet
    path = builtin_skin_path(skin_id)
    original = path.read_bytes()
    image = QImage(str(path))
    assert not image.isNull() and image.hasAlphaChannel()
    assert image.pixelColor(0, 0).alpha() == 0
    menu = window._menu()
    looks = next(a.menu() for a in menu.actions() if a.text() == "更换形象")
    next(a for a in looks.actions() if a.text() == label).trigger()
    assert window.skin_path == str(path)
    restored = PetSettings(window.settings.path)
    assert restored.skin_for("test") == str(path)
    assert restored.skin_for("another") is None
    reopened = PetWindow(BackendClient("http://127.0.0.1:1"), persona_id="test", with_tray=False)
    try:
        assert reopened.skin_path == str(path), "Skin must load even when the backend is offline"
    finally:
        reopened.bubble.close()
        reopened.close()
        reopened.client.close()
    assert window._icon().pixmap(64, 64).toImage() == QPixmap(str(path)).scaled(
        512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation
    ).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation).toImage()

    frames = []
    window._affection = 0
    for state in ("idle", "listening", "thinking", "speaking", "offline"):
        window.state = state
        canvas = QPixmap(PET_SIZE, PET_SIZE)
        canvas.fill(Qt.transparent)
        window.render(canvas)
        frame = canvas.toImage()
        assert sum(frame.pixelColor(x, y).alpha() > 0 for x in range(0, PET_SIZE, 4)
                   for y in range(0, PET_SIZE, 4)) > 200
        assert frame.pixelColor(PET_SIZE // 2, PET_SIZE - 17).alpha() > 200
        frames.append(frame)
    assert frames[0] != frames[1]
    assert frames[0] != frames[2]
    window.clear_skin()
    assert window.skin_pixmap is None
    assert path.read_bytes() == original, "Reset must never delete bundled artwork"


def test_hover_and_click_animate_without_breaking_chat(pet):
    window, app = pet
    window.choose_builtin_skin("sakura_cat")
    window._affection = 0
    window.state = "idle"
    window._hovered = True
    before = QPixmap(PET_SIZE, PET_SIZE)
    before.fill(Qt.transparent)
    window.render(before)
    for _ in range(10):
        window._tick()
    after = QPixmap(PET_SIZE, PET_SIZE)
    after.fill(Qt.transparent)
    window.render(after)
    assert before.toImage() != after.toImage()
    QTest.mouseClick(window, Qt.LeftButton, pos=QPoint(100, 90))
    assert window._affection == 1
    assert not window.bubble.input.isHidden()
    for _ in range(45):
        window._tick()
    assert window._affection == 0


def test_clear_inherited_skin_preserves_shared_file(tmp_path):
    settings = PetSettings(tmp_path / "settings.json")
    shared = tmp_path / "shared.png"
    shared.write_bytes(b"shared file")
    settings.set_skin(None, str(shared))
    clear_skin("no_override", settings=settings)
    assert shared.exists()
    assert settings.skin_for("no_override") == str(shared)
    assert builtin_skin_path("../outside") is None
