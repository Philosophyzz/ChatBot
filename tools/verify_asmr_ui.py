"""Actual Qt menu/controller + stereo device, silent during this automated check."""
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from pet.client import BackendClient
from pet.settings import PetSettings
import pet.window as wm


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    sandbox = tempfile.TemporaryDirectory()
    settings = PetSettings(Path(sandbox.name) / 'settings.json')
    settings.values.update(awareness_enabled=False, muted=True, live2d_models={'hiyori':'D:/Models/Live2D/Hiyori/Hiyori.model3.json'})
    wm.get_settings = lambda: settings
    client = BackendClient()
    pet = wm.PetWindow(client, persona_id='hiyori', with_tray=False)
    pet.personas, _ = client.personas()
    pet._apply_persona()
    pet.show()
    errors = []
    def start():
        try:
            menu = pet._menu()
            assert 'ASMR（无语音）' in [a.text() for a in menu.actions()]
            pet.asmr.volume = 0  # real stereo device runs silently
            pet.asmr.start('paper', 4)
            assert pet.asmr.active
            assert not pet.hands_free
        except Exception as exc:
            errors.append(str(exc))
        QTimer.singleShot(1000, pause)
    checkpoint = []
    def pause():
        pet.asmr.toggle_pause()
        checkpoint.append(pet.asmr.elapsed)
        pet.grab().save('logs/asmr-ui.png')
        QTimer.singleShot(600, resume)
    def resume():
        try:
            assert pet.asmr.active and pet.asmr.paused
            assert abs(pet.asmr.elapsed-checkpoint[0]) < .2
        except Exception as exc:
            errors.append(str(exc))
        pet.asmr.toggle_pause()
        QTimer.singleShot(500, stop)
    def stop():
        pet.asmr.stop()
        QTimer.singleShot(600, finish)
    def finish():
        try:
            assert not pet.asmr.active
            assert pet.asmr.elapsed > .2
            assert not pet.asmr.thread.is_alive()
        except Exception as exc:
            errors.append(str(exc))
        pet.awareness.close()
        if pet.live2d_view:
            pet.live2d_view.release();pet.live2d_view=None
        pet.bubble.close();pet.close();app.quit()
    QTimer.singleShot(1800, start)
    app.exec()
    client.close()
    sandbox.cleanup()
    if errors:
        raise RuntimeError(errors)
    print('ASMR_QT_STEREO_PAUSE_CANCEL_OK', flush=True)


if __name__ == '__main__':
    main()
