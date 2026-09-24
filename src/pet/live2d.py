"""Optional native Cubism 3+ renderer. PNG pets have no Live2D dependency."""
import json
import math
import random
import time
from pathlib import Path
from core.companion import bounded_parameters


class ActionTimeline:
    """Bounded keyframes, cubic transitions and automatic return to neutral."""
    def __init__(self, capabilities):
        self.capabilities = capabilities
        self.frames = []
        self.started = 0.0
        self.origin = {}

    def set(self, plan, now=None):
        stamp = time.monotonic() if now is None else now
        self.origin = self.values(stamp)
        frames = []
        for frame in (plan.get("frames", []) if isinstance(plan, dict) else [])[:4]:
            try:
                duration = float(frame.get("duration_ms", 1500)) / 1000
                if not math.isfinite(duration):
                    continue
                params = bounded_parameters(frame.get("parameters"), self.capabilities)
                if params:
                    frames.append((max(.4, min(3, duration)), params))
            except (ValueError, TypeError, AttributeError):
                continue
        keys = set(self.origin)
        for _, params in frames:
            keys.update(params)
        neutral = {k: self.capabilities[k]["default"] for k in keys}
        self.frames = [(duration, {**neutral, **params}) for duration, params in frames] + ([(1.5, neutral)] if frames else [])
        self.started = stamp

    def values(self, now=None):
        elapsed = (time.monotonic() if now is None else now) - self.started
        origin = self.origin
        for duration, target in self.frames:
            if elapsed <= duration:
                t = max(0.0, elapsed / duration)
                t = 4*t*t*t if t < .5 else 1 - (-2*t + 2)**3 / 2
                return {key: origin.get(key, self.capabilities[key]["default"]) * (1-t) + value*t for key, value in target.items()}
            elapsed -= duration
            origin = target
        return {}


def validate_model(path):
    path = Path(path).resolve()
    if not path.name.endswith(".model3.json"):
        raise ValueError("请选择 Cubism 3+ 的 .model3.json 文件")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    refs = data.get("FileReferences", {})
    if not refs.get("Moc") or not refs.get("Textures"):
        raise ValueError("模型缺少 Moc 或 Textures，PNG 不能直接作为 Live2D 模型")

    def files(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, list):
            for child in node:
                yield from files(child)
        elif isinstance(node, dict):
            for key, child in node.items():
                if key not in {"Name", "FadeInTime", "FadeOutTime"}:
                    yield from files(child)

    for ref in files(refs):
        target = (path.parent / ref).resolve()
        if not target.is_relative_to(path.parent) or not target.is_file():
            raise ValueError(f"模型资源缺失或超出模型目录：{ref}")
    return str(path)


def create_view(parent, path, on_error):
    """Import lazily so optional native libraries cannot break ordinary startup."""
    import live2d.v3 as runtime
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtOpenGLWidgets import QOpenGLWidget

    if not getattr(create_view, "initialized", False):
        runtime.init()
        create_view.initialized = True

    class View(QOpenGLWidget):
        def __init__(self):
            super().__init__(parent)
            fmt = QSurfaceFormat()
            fmt.setAlphaBufferSize(8)
            fmt.setDepthBufferSize(24)
            fmt.setVersion(2, 1)
            self.setFormat(fmt)
            self.setAttribute(Qt.WA_AlwaysStackOnTop)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_TransparentForMouseEvents)
            self.model = None
            self.failed = False
            self.capabilities = {}
            self.timeline = ActionTimeline({})
            self.next_idle = time.monotonic() + 6
            self.mouth = 0.0
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update)

        def fail(self, exc):
            if self.failed:
                return
            self.failed = True
            self.timer.stop()
            QTimer.singleShot(0, lambda: on_error(str(exc)))

        def initializeGL(self):
            try:
                runtime.glInit()
                self.model = runtime.LAppModel()
                self.model.LoadModelJson(path)
                self.model.Resize(self.width(), self.height())
                for index in range(self.model.GetParameterCount()):
                    p = self.model.GetParameter(index)
                    if p.id.startswith(("ParamAngle", "ParamBodyAngle", "ParamEye", "ParamBrow", "ParamCheek", "ParamArm", "ParamHand")):
                        self.capabilities[p.id] = {"min": p.min, "max": p.max, "default": p.default}
                self.timeline = ActionTimeline(self.capabilities)
                self.model.SetAutoBlinkEnable(True)
                self.model.SetAutoBreathEnable(True)
                self.timer.start(33)
                if hasattr(parent, "_live2d_ready"):
                    QTimer.singleShot(700, parent._live2d_ready)
            except Exception as exc:
                self.fail(exc)

        def resizeGL(self, width, height):
            if self.model and not self.failed:
                self.model.Resize(width, height)

        def paintGL(self):
            if not self.model or self.failed:
                return
            try:
                runtime.clearBuffer()
                self.model.Update()
                now = time.monotonic()
                if now > self.next_idle and not self.timeline.values(now):
                    self.next_idle = now + random.uniform(8, 16)
                    self.set_plan({"frames": [{"duration_ms": 2000, "parameters": {
                        "ParamAngleX": random.uniform(-10, 10), "ParamAngleZ": random.uniform(-4, 4),
                        "ParamBodyAngleX": random.uniform(-3, 3), "ParamEyeBallX": random.uniform(-.4, .4)}}]})
                for key, value in self.timeline.values(now).items():
                    self.model.SetParameterValue(key, value)
                awareness = getattr(parent, "awareness", None)
                if awareness and awareness.enabled and awareness.scene == "music":
                    energy = min(1, awareness.audio.rms * 7 + awareness.audio.pulse)
                    for key, value in bounded_parameters({"ParamBodyAngleZ": math.sin(now * 3) * energy * 5,
                                                         "ParamAngleZ": math.sin(now * 3 + .4) * energy * 7}, self.capabilities).items():
                        self.model.SetParameterValue(key, value)
                if parent.state == "speaking":
                    target = getattr(getattr(parent, "player", None), "level", 0)
                    self.mouth += (target - self.mouth) * .65
                    self.model.SetParameterValue("ParamMouthOpenY", min(1, self.mouth * 1.6))
                else:
                    self.mouth = 0.0
                self.model.Draw()
            except Exception as exc:
                self.fail(exc)

        def look_at(self, x, y):
            if self.model and not self.failed:
                self.model.Drag(x, y)

        def react(self, kind):
            if self.model and not self.failed:
                presets = {
                    "head": {"ParamAngleZ": 12, "ParamEyeLOpen": .65, "ParamEyeROpen": .65, "ParamCheek": .6},
                    "face": {"ParamAngleX": -16, "ParamBrowLY": .6, "ParamBrowRY": .6},
                    "hold": {"ParamAngleY": -8, "ParamCheek": .4, "ParamEyeLOpen": .65, "ParamEyeROpen": .65},
                    "feed": {"ParamAngleY": 10, "ParamBodyAngleX": 4},
                    "body": {"ParamAngleX": 12, "ParamBodyAngleZ": 4},
                }
                self.set_plan({"frames": [{"parameters": presets.get(kind, presets["body"]), "duration_ms": 1200}]})

        def set_plan(self, plan):
            self.timeline.set(plan)
            self.next_idle = time.monotonic() + 12

        def release(self):
            self.timer.stop()
            if self.context() and self.context().isValid():
                self.makeCurrent()
                self.model = None
                self.doneCurrent()
            self.hide()
            self.deleteLater()

    return View()
