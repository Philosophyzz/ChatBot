"""Render installed sample models through the actual Qt/Cubism GPU path."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QWidget
from pet.live2d import create_view, validate_model


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication([])
    errors = []
    entries = [("hiyori", "Hiyori/Hiyori.model3.json"), ("mao", "Mao/Mao.model3.json")]
    previews = []
    def next_model():
        if not entries:
            app.quit()
            return
        name, model = entries.pop(0)
        parent = QWidget()
        parent.setAttribute(Qt.WA_TranslucentBackground)
        parent.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool)
        parent.resize(512, 512)
        parent.state = "idle"
        parent.awareness = None
        path = validate_model(Path("D:/Models/Live2D") / model)
        view = create_view(parent, path, errors.append)
        view.setGeometry(0, 0, 512, 512)
        parent.show()
        view.show()
        previews.append((parent, view))
        def save():
            dest = Path(__file__).resolve().parents[1] / "src/pet/assets" / (name + ".png")
            if view.failed or not view.model:
                errors.append("model did not load: " + name)
            else:
                frame = view.grabFramebuffer()
                if not frame.save(str(dest)):
                    errors.append("failed saving: " + name)
                print(name, "parameters", len(view.capabilities), "saved", dest, flush=True)
            view.release()
            parent.close()
            QTimer.singleShot(300, next_model)
        QTimer.singleShot(1700, save)
    app.setQuitOnLastWindowClosed(False)
    QTimer.singleShot(0, next_model)
    app.exec()
    if errors:
        raise RuntimeError(str(errors))


if __name__ == "__main__":
    main()
