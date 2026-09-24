"""Real Qt/Cubism render and character-switch check using isolated pet settings."""
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication
from pet.client import BackendClient
from pet.settings import PetSettings
import pet.window as window_module


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    client = BackendClient()
    personas, _ = client.personas()
    sandbox = tempfile.TemporaryDirectory()
    settings = PetSettings(Path(sandbox.name) / "settings.json")
    settings.values.update(awareness_enabled=False, muted=True, live2d_models={
        "hiyori": "D:/Models/Live2D/Hiyori/Hiyori.model3.json", "mao": "D:/Models/Live2D/Mao/Mao.model3.json"})
    window_module.get_settings = lambda: settings
    pet = window_module.PetWindow(client, persona_id="hiyori", with_tray=False)
    pet.personas = personas
    pet.set_scale(2, persist=False)
    pet._apply_persona()
    pet.show()
    errors = []
    def first():
        try:
            assert pet.live2d_view and pet.live2d_view.model and not pet.live2d_view.failed
            assert pet.live2d_view.capabilities
            pet.live2d_view.set_plan({"frames": [{"duration_ms": 1000, "parameters": {"ParamAngleX": 25, "ParamAngleZ": -20}}]})
            print("Hiyori GPU render OK", flush=True)
        except Exception as exc:
            errors.append(str(exc))
        QTimer.singleShot(800, switch)
    def switch():
        pet.grab().save("logs/companion-hiyori.png")
        pet.select_persona("mao")
        QTimer.singleShot(1500, final)
    def final():
        try:
            assert pet.live2d_view and pet.live2d_view.model and not pet.live2d_view.failed
            image = pet.live2d_view.grabFramebuffer()
            colored = sum(1 for y in range(0,image.height(),8) for x in range(0,image.width(),8)
                          if image.pixelColor(x,y).alpha()>200 and image.pixelColor(x,y).saturation()>30)
            assert colored > 100, f"model lost textures after context switch: {colored}"
            menu = pet._menu(from_pet=True)
            names = [a.text() for a in menu.actions()]
            assert "陪伴场景" in names and "主动感知（屏幕 / 活动 / 媒体）" in names
            assert not pet.awareness.enabled
            pet.grab().save("logs/companion-mao.png")
            print("Mao GPU render after switch OK; scene menu and awareness switch OK", flush=True)
            menu.deleteLater()
        except Exception as exc:
            errors.append(str(exc))
        pet.awareness.close()
        if pet.live2d_view:
            pet.live2d_view.release()
            pet.live2d_view = None
        pet.bubble.close()
        pet.close()
        app.quit()
    QTimer.singleShot(1500, first)
    app.exec()
    client.close()
    sandbox.cleanup()
    if errors:
        raise RuntimeError(str(errors))


if __name__ == "__main__":
    main()
