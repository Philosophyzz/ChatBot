"""Opt-out local awareness: aggregate input activity, ephemeral screen, WASAPI output."""
from __future__ import annotations
import asyncio
import ctypes
from ctypes import wintypes
import math
import time

from core.companion import InterventionGate, SCENES


class DesktopSignals:
    def __init__(self):
        self.last_tick = None
        self.last_cursor = None
        self.activity = 0
        self.distance = 0.0
        self.last_disk = None

    def sample(self):
        import psutil
        user = ctypes.windll.user32
        user.GetForegroundWindow.restype = wintypes.HWND
        class LastInput(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]
        value = LastInput()
        value.cbSize = ctypes.sizeof(value)
        if not user.GetLastInputInfo(ctypes.byref(value)):
            return {"protected": True}
        idle = ((ctypes.windll.kernel32.GetTickCount() & 0xffffffff) - value.dwTime) & 0xffffffff
        if self.last_tick is not None and value.dwTime != self.last_tick:
            self.activity += 1
        self.last_tick = value.dwTime
        point = wintypes.POINT()
        user.GetCursorPos(ctypes.byref(point))
        if self.last_cursor:
            self.distance += math.hypot(point.x - self.last_cursor[0], point.y - self.last_cursor[1])
        self.last_cursor = (point.x, point.y)
        hwnd = user.GetForegroundWindow()
        if not hwnd:
            return {"protected": True}
        title = ctypes.create_unicode_buffer(512)
        user.GetWindowTextW(hwnd, title, 512)
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            app = psutil.Process(pid.value).name()
        except (psutil.Error, OSError):
            app = "unknown"
        # Metadata filter is a best effort, not a claim of complete privacy detection.
        blocked = ("password", "密码", "银行", "bank", "incognito", "inprivate", "无痕", "keepass", "1password", "bitwarden")
        protected = any(word in (title.value + " " + app).lower() for word in blocked)
        disk = psutil.disk_io_counters()
        stamp = time.monotonic()
        count = (disk.read_bytes + disk.write_bytes) if disk else 0
        rate = max(0, (count - self.last_disk[0]) / max(.1, stamp - self.last_disk[1]) / 1024) if self.last_disk else 0
        self.last_disk = (count, stamp)
        return {"idle_s": idle / 1000, "activity_ticks": min(self.activity, 10000), "mouse_distance": min(self.distance, 1e7),
                "disk_kbps": min(rate, 1e9), "title": "" if protected else title.value[:200], "app": app[:120],
                "protected": protected, "hwnd": int(hwnd)}

    def reset(self):
        self.activity = 0
        self.distance = 0


async def _media():
    from winsdk.windows.media.control import GlobalSystemMediaTransportControlsSessionManager
    manager = await GlobalSystemMediaTransportControlsSessionManager.request_async()
    session = manager.get_current_session()
    if not session:
        return {"media_status": "unavailable", "media_playing": False}
    properties = await session.try_get_media_properties_async()
    status = int(session.get_playback_info().playback_status)
    return {"media_title": (properties.title or "")[:160], "media_artist": (properties.artist or "")[:100],
            "media_playing": status == 4, "media_status": {4: "playing", 5: "paused"}.get(status, "stopped")}


def media_snapshot():
    try:
        return asyncio.run(asyncio.wait_for(_media(), timeout=3))
    except Exception:
        return {"media_status": "unavailable", "media_playing": False}


class LoopbackMeter:
    """Output-device only. PCM is reduced to levels immediately; no file/clip retained."""
    def __init__(self):
        self.stream = self.audio = None
        self.rms = self.peak = self.pulse = 0.0
        self.status = "关闭"

    def start(self):
        if self.stream:
            return
        try:
            import pyaudiowpatch as pa
            import numpy as np
            self.audio = pa.PyAudio()
            device = self.audio.get_default_wasapi_loopback()
            channels = device["maxInputChannels"]
            def callback(data, frames, time_info, flags):
                samples = np.frombuffer(data, dtype=np.float32)
                level = min(1.0, float(np.sqrt(np.mean(samples ** 2)))) if len(samples) else 0
                self.pulse = min(1.0, max(0.0, (level - self.rms) * 12))
                self.rms = level
                self.peak = min(1.0, float(np.max(np.abs(samples)))) if len(samples) else 0
                return None, pa.paContinue
            self.stream = self.audio.open(format=pa.paFloat32, channels=channels, rate=int(device["defaultSampleRate"]),
                                          input=True, input_device_index=device["index"], frames_per_buffer=1024, stream_callback=callback)
            self.status = "电脑输出音量 / 节奏"
        except Exception:
            self.stop()
            self.status = "当前输出设备无法回环采集"

    def stop(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        if self.audio:
            self.audio.terminate()
        self.stream = self.audio = None
        self.rms = self.peak = self.pulse = 0.0
        self.status = "关闭"


class AwarenessController:
    def __init__(self, pet):
        from PySide6.QtCore import QTimer
        self.pet = pet
        self.enabled = bool(pet.settings.get("awareness_enabled", True))
        self.scene = pet.settings.get("companion_scene", "normal")
        if self.scene not in SCENES:
            self.scene = "normal"
        self.screen_enabled = bool(pet.settings.get("screen_awareness", True))
        self.signals = DesktopSignals()
        self.audio = LoopbackMeter()
        self.gate = InterventionGate()
        self.last_interaction = time.monotonic()
        self.epoch = 0
        self.inflight = False
        self.last_reason = "等待用户空闲"
        self.last_media_at = 0
        self.media = {}
        self.media_pending = False
        self.timer = QTimer(pet)
        self.timer.timeout.connect(self.tick)
        self.timer.start(2000)

    @property
    def label(self):
        return f"感知开 · {SCENES[self.scene]['label']}" if self.enabled and self.scene != "quiet" else "感知暂停"

    def interact(self):
        self.last_interaction = time.monotonic()
        self.epoch += 1

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self.pet.settings.set("awareness_enabled", self.enabled)
        self.interact()
        if not self.enabled:
            self.audio.stop()
            self.media = {}
            self.signals.reset()
        self.last_reason = "等待用户空闲" if self.enabled else "感知已关闭"
        self.pet.update()

    def set_scene(self, scene):
        self.scene = scene if scene in SCENES else "normal"
        self.pet.settings.set("companion_scene", self.scene)
        self.interact()
        self.audio.stop()
        self.pet.bubble.show_text(f"场景：{SCENES[self.scene]['label']}\n电影播放时优先无声反应；主动发言每小时最多 4 次。", autohide_s=6)
        self.pet.update()

    def toggle_screen(self):
        self.screen_enabled = not self.screen_enabled
        self.pet.settings.set("screen_awareness", self.screen_enabled)
        self.interact()

    def tick(self):
        pet = self.pet
        if not self.enabled or self.scene == "quiet" or not pet.isVisible():
            self.audio.stop()
            return
        from core.companion import local_connection
        from types import SimpleNamespace
        if not local_connection(SimpleNamespace(mode=pet.model_connection.get("mode", "api"), base_url=pet.model_connection.get("base_url", ""))):
            self.last_reason = "外部 API 模式下屏幕感知暂停"
            self.audio.stop()
            return
        busy = pet._chat_busy or pet.state != "idle" or pet._recording or pet.bubble.input.isVisible() or (getattr(pet, 'asmr', None) and pet.asmr.active)
        if busy:
            self.audio.stop()
            self.last_reason = "用户对话优先"
            return
        try:
            obs = self.signals.sample()
        except Exception:
            self.last_reason = "无法读取桌面活动"
            return
        if obs.get("protected"):
            self.audio.stop()
            self.last_reason = "受保护窗口，暂停采集"
            return
        if self.scene in {"movie", "music"}:
            self.audio.start()
        if time.monotonic() - self.last_media_at > 10 and not self.media_pending:
            self.media_pending = True
            epoch = self.epoch
            def got_media(value):
                self.media_pending = False
                self.last_media_at = time.monotonic()
                if self.epoch == epoch and self.enabled:
                    self.media = value
            pet._run_async(media_snapshot, got_media, lambda exc: got_media({"media_status": "unavailable"}))
        obs.update(self.media)
        # Some local players do not expose SMTC. Audible output still suppresses movie speech.
        obs["media_playing"] = bool(obs.get("media_playing") or (self.scene != "normal" and self.audio.rms > .003))
        obs.update(enabled=True, scene=self.scene, busy=busy, pet_idle_s=time.monotonic() - self.last_interaction,
                   audio_rms=self.audio.rms, audio_peak=self.audio.peak)
        reason = self.gate.check(obs)
        if reason or self.inflight:
            self.last_reason = reason or "正在判断是否适合说话"
            return
        self.gate.reserve()
        hwnd = obs.pop("hwnd", 0)
        image = ""
        if self.screen_enabled:
            from PySide6.QtCore import QByteArray, QBuffer, QIODevice, Qt
            from PySide6.QtGui import QGuiApplication
            import base64
            # Capture only the foreground window's monitor; reduced image stays in RAM.
            screen = QGuiApplication.screenAt(pet.cursor().pos()) or QGuiApplication.primaryScreen()
            pixmap = screen.grabWindow(hwnd) if hwnd else None
            if pixmap is not None and not pixmap.isNull():
                pixmap = pixmap.scaled(960, 640, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                data = QByteArray()
                buffer = QBuffer(data)
                buffer.open(QIODevice.WriteOnly)
                pixmap.save(buffer, "JPG", 65)
                image = base64.b64encode(bytes(data)).decode("ascii")
        epoch, persona = self.epoch, pet.persona_id
        capabilities = pet.live2d_view.capabilities if pet.live2d_view else {}
        payload = {"observation": obs, "image": image, "persona_id": persona or "", "capabilities": capabilities}
        self.inflight = True
        self.signals.reset()
        def work():
            if self.epoch != epoch or not self.enabled:
                return {"speak": False, "reason": "观察已取消"}
            response = pet.client._client.post("/api/pet/observe", json=payload, timeout=25)
            response.raise_for_status()
            return response.json()
        def done(result):
            self.inflight = False
            if self.epoch != epoch or not self.enabled or pet.persona_id != persona or pet.state != "idle":
                return
            self.last_reason = result.get("reason", "保持安静")
            if pet.live2d_view and result.get("parameters"):
                pet.live2d_view.set_plan({"frames": [{"parameters": result["parameters"], "duration_ms": 1800}]})
            if result.get("speak") and result.get("text"):
                self.gate.record()
                pet.bubble.show_text(result["text"], autohide_s=12)
                if result.get("voice") and not pet.muted and not (self.scene == "movie" and self.audio.rms > .003):
                    pet.speak(result["text"])
        def failed(exc):
            self.inflight = False
            self.last_reason = "感知服务暂不可用，保持安静"
        pet._run_async(work, done, failed)

    def close(self):
        self.enabled = False
        self.epoch += 1
        self.timer.stop()
        self.audio.stop()
