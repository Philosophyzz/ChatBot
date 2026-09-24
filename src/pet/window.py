"""The desktop pet: an always-on-top character that listens and talks.

Design notes, because a desktop pet fails in ways a normal window does not:

* **Frameless + translucent + always-on-top**, dragged by the body. It must never steal
  focus while the user is typing elsewhere, so it is created with ``Qt.Tool`` and
  ``WindowDoesNotAcceptFocus``; only clicking it (or opening the input box) focuses it.
* **Bundled anime sprites or custom images**, with a drawn fallback. Images are cached
  when the skin changes; animation uses painter transforms instead of re-decoding files.
* **Every network call runs on a worker thread** and reports back through Qt signals. A
  blocking ``httpx`` call inside ``paintEvent``/a slot freezes the whole pet — including
  its animation — and looks like a crash.
* **Push-to-talk**: hold the drawn mic button (or Ctrl+Space while the pet has focus) to
  record; release to send. Optional hands-free mode keeps listening and cuts on silence.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QPointF, QPoint, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from core.logging import get_logger
from pet.audio import AudioUnavailable, Player, Recorder
from pet.client import BackendClient, BackendError
from pet.hands_free import (
    EVENT_SEND,
    EVENT_SPEECH_STARTED,
    PHASE_SPEECH,
    SENSITIVITY_LABELS,
    HandsFreeListener,
    describe,
)
from pet.settings import get_settings

log = get_logger(__name__)

PET_SIZE = 210
MIN_SCALE = 0.35
MAX_SCALE = 3.0
BUBBLE_MAX_WIDTH = 380
#: Persona accent colours, keyed by the persona's dominant emotion.
EMOTION_COLORS = {
    "sweet": "#ff8fb1",
    "gentle": "#f4a3c0",
    "cheerful": "#ffb35c",
    "excited": "#ff9d5c",
    "calm": "#7fc6d9",
    "serious": "#8ea2c6",
    "shy": "#d8a3e8",
    "sad": "#8f9fd0",
    "neutral": "#9fb3d9",
}


def parse_color(value: str) -> QColor:
    color = QColor(value)
    return color if color.isValid() else QColor("#ff8fb1")


def _rects_overlap(a: tuple, b: tuple, gap: int = 4) -> bool:
    """Do two ``(x, y, w, h)`` rects overlap (with a small gap so pets do not touch)?"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + gap <= bx or bx + bw + gap <= ax or ay + ah + gap <= by or by + bh + gap <= ay
    )


def _to_logical(rects: List[tuple], ratio: float) -> List[tuple]:
    """Convert Win32 window rects (physical pixels) into Qt's logical coordinates.

    This mismatch is invisible on a 100% display and silently breaks everything on a
    scaled one: Qt reports the pet at ``(1943, 915)`` while ``GetWindowRect`` reports the
    same window at ``(3400, 1601)`` (175% scaling), so two pets that are stacked on top of
    each other look like they are 1500px apart and the "find a free slot" loop never fires.
    """
    if not ratio or ratio == 1.0:
        return [tuple(rect) for rect in rects]
    return [(x / ratio, y / ratio, w / ratio, h / ratio) for x, y, w, h in rects]


class Worker:
    """Legacy helper kept for non-Qt callers (see ``PetWindow._run_async``).

    Qt widgets may only be touched from the GUI thread. Running a callback directly from
    a worker thread "works" until it starts a QTimer or resizes a widget, at which point
    Qt prints ``QObject::startTimer: Timers cannot be started from another thread`` and
    the UI silently misbehaves. ``PetWindow`` therefore marshals results through signals.
    """

    @staticmethod
    def run(fn: Callable[[], Any], on_done: Callable[[Any], None], on_error: Callable[[Exception], None]) -> None:
        def target() -> None:
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 - surfaced in the bubble
                on_error(exc)
                return
            on_done(result)

        threading.Thread(target=target, name="pet-worker", daemon=True).start()


class SpeechBubble(QWidget):
    """Frameless translucent bubble with the assistant's text and a text input.

    Placement belongs to the bubble's owner (it must sit above the pet, whose position
    only :class:`PetWindow` knows), so the bubble just announces "I am shown and sized"
    through :attr:`on_shown`. Without that hook, repositioning has to be repeated at every
    ``show_text`` call site — and any site that forgets leaves a frameless window where
    Windows drops an unpositioned one: the middle of the screen.
    """

    submitted = Signal(str)

    def __init__(self) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFixedWidth(BUBBLE_MAX_WIDTH)
        #: Invoked after every show/resize so the owner can move the bubble into place.
        self.on_shown: Optional[Callable[[], None]] = None

        self.label = QLabel("", self)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("color:#f2f4ff; font-size:13.5px; background:transparent;")
        self.label.setTextInteractionFlags(Qt.NoTextInteraction)

        self.input = QLineEdit(self)
        self.input.setPlaceholderText("说点什么…（Enter 发送，Esc 收起）")
        self.input.setStyleSheet(
            "background:rgba(255,255,255,0.10); color:#eef1ff; border:1px solid rgba(255,255,255,0.18);"
            "border-radius:9px; padding:7px 9px; font-size:13px;"
        )
        self.input.returnPressed.connect(self._submit)
        self.input.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        layout.addWidget(self.label)
        layout.addWidget(self.input)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self._typing_target = ""
        self._typing_shown = 0
        self._typing_timer = QTimer(self)
        self._typing_timer.timeout.connect(self._type_step)

    # -- content -----------------------------------------------------------------------
    def _notify_shown(self) -> None:
        """Tell the owner the bubble is on screen with its final size."""
        callback = self.on_shown
        if callback is None or not self.isVisible():
            return
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 - placement must never break speech
            log.warning("bubble placement failed", extra={"error": str(exc)})

    def show_text(self, text: str, *, autohide_s: float = 0.0, typewriter: bool = False) -> None:
        self._typing_timer.stop()
        if typewriter:
            self._typing_target = text
            self._typing_shown = 0
            self.label.setText("")
            self._typing_timer.start(28)
        else:
            self.label.setText(text)
        self.adjustSize()
        self.show()
        self.raise_()
        self._notify_shown()
        self._hide_timer.stop()
        if autohide_s > 0:
            self._hide_timer.start(int(autohide_s * 1000))

    def append_text(self, delta: str) -> None:
        self.show_text(self.label.text() + delta)

    def _type_step(self) -> None:
        self._typing_shown = min(len(self._typing_target), self._typing_shown + 2)
        self.label.setText(self._typing_target[: self._typing_shown])
        self.adjustSize()
        # The bubble grows while typing; without this it would drift off the pet.
        self._notify_shown()
        if self._typing_shown >= len(self._typing_target):
            self._typing_timer.stop()

    def ask(self) -> None:
        self.input.show()
        self.show()
        self.raise_()
        self.adjustSize()
        self._notify_shown()
        self.activateWindow()
        self.input.setFocus()
        self._hide_timer.stop()

    def _submit(self) -> None:
        text = self.input.text().strip()
        self.input.clear()
        if text:
            self.submitted.emit(text)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key_Escape:
            self.input.hide()
            self.adjustSize()
            self._notify_shown()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -7)
        path = QPainterPath()
        path.addRoundedRect(rect, 14, 14)
        # little tail pointing down at the pet
        tail = QPainterPath()
        tail.moveTo(rect.center().x() - 10, rect.bottom() - 1)
        tail.lineTo(rect.center().x(), rect.bottom() + 8)
        tail.lineTo(rect.center().x() + 10, rect.bottom() - 1)
        path = path.united(tail)
        painter.setBrush(QBrush(QColor(22, 24, 34, 238)))
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.drawPath(path)


class PetWindow(QWidget):
    """The character: animation, push-to-talk, menus, and all the state wiring."""

    state_changed = Signal(str)
    reply_ready = Signal(str)
    partial_reply = Signal(str)
    error_raised = Signal(str)
    #: Worker results are marshalled through these: emitting from any thread is safe and
    #: the connected slot always runs on the GUI thread.
    worker_ok = Signal(object)
    worker_fail = Signal(object)
    speaking = Signal(bool)

    def __init__(
        self,
        client: BackendClient,
        *,
        persona_id: Optional[str] = None,
        with_tray: bool = True,
    ) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFixedSize(PET_SIZE, PET_SIZE)
        self.setWindowTitle("本地聊天机器人 · 桌宠")
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)

        self.client = client
        self.persona_id = persona_id
        #: Persona whose single-instance guard this pet holds (claimed by run_pet.run_gui
        #: before any window exists). Tracked so switching persona can hand the old name
        #: back and keep the "one pet per persona" rule true.
        self._guard_key: Optional[str] = persona_id
        self.personas: List[Dict[str, Any]] = []
        self.persona_name = "本地助手"
        self.accent = QColor("#ff8fb1")

        self.recorder = Recorder()
        self.player = Player()
        self._voice_epoch = 0
        self._chat_busy = False
        # The player runs on its own thread; hop back to the GUI thread via a signal.
        self.player.on_state = lambda speaking: self.speaking.emit(bool(speaking))
        self.worker_ok.connect(self._dispatch_ok)
        self.worker_fail.connect(self._dispatch_fail)
        self.speaking.connect(self._on_speaking)

        self.state = "idle"          # idle | listening | thinking | speaking | offline
        self.muted = False
        self.hands_free = False
        self.listener = HandsFreeListener()
        #: The "I can't hear you at all" hint is worth showing once per hands-free session.
        self._hands_free_hint_shown = False
        #: Character image for the current persona (None → draw the built-in character).
        self.skin_path: Optional[str] = None
        self.skin_pixmap: Optional[QPixmap] = None
        self.settings = get_settings()
        from pet.asmr import ASMRController
        self.asmr = ASMRController(self)
        self.scale_factor = 1.0
        self.emotion_state = {}
        self.pet_mood = "calm"
        self._scale_target = 1.0
        self._scale_anchor = None
        self._scale_timer = QTimer(self)
        self._scale_timer.setInterval(16)
        self._scale_timer.timeout.connect(self._animate_scale)
        self._scale_save_timer = QTimer(self)
        self._scale_save_timer.setSingleShot(True)
        self._scale_save_timer.timeout.connect(lambda: self.settings.set("scale", self._scale_target))
        self.live2d_view = None
        self._live2d_path = None
        self._touch_kind = "body"
        self._held = False
        self._last_touch_at = 0.0
        self.model_connection = {}
        self.runtime_migration = {}
        self.muted = bool(self.settings.get("muted"))
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._hold_touch)
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(lambda: self.touch(self._touch_kind))
        self.setMouseTracking(True)
        #: Base-model tiers (fetched from /api/models) and whether a switch is in flight.
        self.model_tiers: List[Dict[str, Any]] = []
        self.model_current: Optional[str] = None
        self._switching_model = False
        #: Hands-free preset name ("quiet"/"normal"/"noisy"), persisted between runs.
        self.sensitivity = str(self.settings.get("hands_free_sensitivity") or "normal")
        if self.sensitivity not in SENSITIVITY_LABELS:
            self.sensitivity = "normal"
        self.listener.set_sensitivity(self.sensitivity)
        self._phase = 0.0
        self._blink = 0.0
        self._hovered = False
        self._hover_amount = 0.0
        self._affection = 0.0
        self._press_pos: Optional[QPoint] = None
        self._press_moved = False
        self._recording = False
        self._mic_rect = QRectF()
        self.bubble = SpeechBubble()
        self.bubble.submitted.connect(self.send_text)
        # Every show/resize repositions the bubble above the pet (see SpeechBubble.on_shown).
        self.bubble.on_shown = self._reposition_bubble
        from pet.awareness import AwarenessController
        self.awareness = AwarenessController(self)
        self._action_epoch = 0

        self._anim = QTimer(self)
        self._anim.timeout.connect(self._tick)
        self._anim.start(33)  # ~30 fps: smooth enough, negligible CPU

        self._connect_timer = QTimer(self)
        self._connect_timer.timeout.connect(self.refresh_personas)
        self._connect_timer.start(30000)
        # Model tiers change rarely (downloads/switches), so a slower poll is enough — but it
        # must be polled, because a switch can also be started from the web UI.
        self._model_timer = QTimer(self)
        self._model_timer.timeout.connect(self.refresh_models)
        self._model_timer.start(60000)

        self._silence_timer = QTimer(self)
        self._silence_timer.timeout.connect(self._hands_free_tick)

        self.tray: Optional[QSystemTrayIcon] = None
        if with_tray:
            self._build_tray()
        self._load_skin()
        self.set_scale(self.settings.get("scale", 1.0), persist=False)
        self._place_bottom_right()
        self.refresh_personas()
        self.refresh_models()
        self._set_state("thinking")

    # -- lifecycle ---------------------------------------------------------------------
    def _run_async(
        self,
        work: Callable[[], Any],
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Run ``work`` on a worker thread, deliver the outcome on the GUI thread."""

        def target() -> None:
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001 - shown in the speech bubble
                # The window can be gone by now (app quitting, or a script that closes it):
                # emitting into a deleted QObject raises in *this* thread and would print an
                # unhandled-exception traceback nobody can act on.
                try:
                    self.worker_fail.emit((on_error, exc))
                except RuntimeError:
                    pass
                return
            try:
                self.worker_ok.emit((on_done, result))
            except RuntimeError:
                pass

        threading.Thread(target=target, name="pet-worker", daemon=True).start()

    def _dispatch_ok(self, payload: object) -> None:
        callback, value = payload  # type: ignore[misc]
        callback(value)

    def _dispatch_fail(self, payload: object) -> None:
        callback, exc = payload  # type: ignore[misc]
        callback(exc)

    def _on_speaking(self, speaking: bool) -> None:
        self._set_state("speaking" if speaking else "idle")

    def _place_bottom_right(self) -> None:
        """Park the pet in the bottom-right corner, in a slot no other pet occupies.

        One pet per persona is enforced, but different personas may run side by side —
        and they all asked for the same corner, so the second pet sat exactly on top of
        the first. Walk left until the candidate spot is free.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        x = area.right() - self.width() - 40
        y = area.bottom() - self.height() - 60
        try:
            from pet import single_instance

            # pet_window_rects() comes from Win32 and is in *physical* pixels, while
            # everything here is Qt logical coordinates (see _to_logical).
            taken = _to_logical(single_instance.pet_window_rects(), float(screen.devicePixelRatio() or 1.0))
        except Exception as exc:  # noqa: BLE001 - placement must never stop the pet
            log.warning("cannot list existing pets", extra={"error": str(exc)})
            taken = []
        for _ in range(8):
            candidate = (x, y, self.width(), self.height())
            if not any(_rects_overlap(candidate, rect) for rect in taken):
                break
            x -= self.width() + 16
        self.move(max(area.left() + 8, x), y)

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(self._icon(), self)
        self.tray.setToolTip("本地聊天机器人 · 桌宠")
        self.tray.setContextMenu(self._menu())
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

    def _icon(self) -> QIcon:
        if self.skin_pixmap is not None:
            return QIcon(self.skin_pixmap)
        pixmap = QPixmap(PET_SIZE, PET_SIZE)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._draw_character(painter, QRectF(0, 0, PET_SIZE, PET_SIZE), mood="idle")
        painter.end()
        return QIcon(pixmap)

    def _refresh_tray_appearance(self) -> None:
        self.setWindowIcon(self._icon())
        if self.tray is not None:
            self.tray.setIcon(self._icon())
            previous = self.tray.contextMenu()
            self.tray.setContextMenu(self._menu())
            if previous is not None:
                previous.deleteLater()

    # -- state -------------------------------------------------------------------------
    def _set_state(self, state: str) -> None:
        if self.state == state:
            return
        self.state = state
        if state in {"thinking", "speaking"}:
            self._pause_hands_free()
        elif state == "idle":
            # Every path back to idle re-arms hands-free (reply finished, speech ended, an
            # error was shown). The old code re-armed once from a timer and lost the mode
            # whenever the reply was still generating at that moment.
            QTimer.singleShot(250, self._resume_hands_free)
        self.state_changed.emit(state)
        self.update()

    def set_status_text(self, text: str) -> None:
        self.bubble.show_text(text, autohide_s=6)

    # -- personas ----------------------------------------------------------------------
    def refresh_personas(self) -> None:
        def work():
            personas, default = self.client.personas()
            return personas, default

        def done(result) -> None:
            personas, default = result
            self.personas = personas
            from persona.catalog import LEGACY_IDS
            if self.persona_id in LEGACY_IDS and LEGACY_IDS[self.persona_id] in {p["id"] for p in personas}:
                self.select_persona(LEGACY_IDS[self.persona_id])
            if not self.persona_id:
                self.persona_id = default or (personas[0]["id"] if personas else None)
            self._apply_persona()
            if self.state in {"thinking", "offline"}:
                self._set_state("idle")
                self.bubble.show_text(f"{self.persona_name}在呢，按住麦克风跟我说话吧。", autohide_s=8)

        def failed(exc: Exception) -> None:
            self._set_state("offline")
            self.bubble.show_text(f"连不上本地服务：{exc}", autohide_s=10)

        self._run_async(work, done, failed)

    def _apply_persona(self) -> None:
        from persona.catalog import CATALOG
        persona = next((p for p in self.personas if p.get("id") == self.persona_id),
                       CATALOG.get(self.persona_id, {}))
        self.persona_name = str(persona.get("name") or self.persona_id or "本地助手")
        emotion = str((persona.get("voice") or {}).get("emotion") or "neutral")
        self.accent = parse_color(EMOTION_COLORS.get(emotion, "#ff8fb1"))
        self.setWindowTitle(f"{self.persona_name} · 桌宠")
        self._load_skin()
        self._load_live2d()
        self.update()
        self.refresh_emotion()

    def refresh_emotion(self) -> None:
        persona_id = self.persona_id
        self._run_async(lambda: self.client._client.get("/api/emotions", params={"persona_id": persona_id}).json(),
                        lambda payload: self._apply_emotion(payload) if self.persona_id == persona_id else None,
                        lambda exc: None)

    def _apply_emotion(self, payload) -> None:
        from core.emotion import PET_LABELS
        previous_mood = self.pet_mood
        self.emotion_state = payload or {}
        self.pet_mood = (self.emotion_state.get("pet") or {}).get("label", "calm")
        colors = {"caring": "#bfa7e9", "curious": "#90d5e8", "encouraging": "#8fd5b0",
                  "playful": "#ffaacc", "cheerful": "#ffc779", "calm": "#7fc6d9"}
        self.accent = parse_color(colors.get(self.pet_mood, "#7fc6d9"))
        self.setToolTip(f"{self.persona_name} · {PET_LABELS.get(self.pet_mood, '平静')}\n"
                        "点头部摸头 · 点脸颊戳脸 · 点嘴巴投喂\n长按拥抱 · 双击聊天 · 滚轮缩放")
        if self.live2d_view is not None and previous_mood != self.pet_mood:
            self.live2d_view.react("head" if self.pet_mood in {"cheerful", "playful"} else "hold")
        self.update()

    def _load_skin(self, *, force: bool = False) -> None:
        """Pick up this persona's custom image, if the user installed one.

        Called on every persona refresh (30 s), so replacing the image does not require
        restarting the pet — at most one refresh interval.

        Invariant: ``skin_pixmap is None`` exactly when ``skin_path is None``. The early
        return below used to compare only the paths, so 「换回默认形象」 (which clears the
        path and then asks for a reload) hit ``None == None`` and returned *before* dropping
        the pixmap — the old picture stayed on screen and the menu item looked broken.
        """
        path = None
        try:
            path = self.settings.skin_for(self.persona_id)
            if not (self.settings.get("skins") or {}).get(self.persona_id):
                from pet.skin import builtin_skin_path
                persona = next((p for p in self.personas if p.get("id") == self.persona_id), {})
                bundled = builtin_skin_path(persona.get("skin_id", ""))
                if bundled:
                    path = str(bundled)
        except Exception as exc:  # noqa: BLE001 - a broken setting must not kill the pet
            log.warning("could not resolve pet skin", extra={"error": str(exc)})
        same_path = path == self.skin_path
        consistent = (path is None) == (self.skin_pixmap is None)
        if same_path and consistent and not force:
            return
        self.skin_path = path
        self.skin_pixmap = None
        if not path:
            self._refresh_tray_appearance()
            self.update()
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            log.warning("pet skin could not be loaded", extra={"path": path})
            self.skin_path = None
            self._refresh_tray_appearance()
            self.update()
            return
        self.skin_pixmap = pixmap.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # The tray icon should look like the character the user actually sees.
        self._refresh_tray_appearance()
        log.info("pet skin applied", extra={"persona": self.persona_id, "path": path})

    def choose_builtin_skin(self, skin_id: str) -> None:
        from pet.skin import BUILTIN_SKINS, builtin_skin_path

        if builtin_skin_path(skin_id) is None:
            self.bubble.show_text("这款内置形象暂时不可用", autohide_s=4)
            return
        if skin_id != self.persona_id:
            self.select_persona(skin_id)
            if self.persona_id != skin_id:
                return
        self.settings.set_skin(self.persona_id, f"builtin:{skin_id}")
        self._load_skin(force=True)
        self._affection = 1.0
        self.update()
        self.bubble.show_text(f"{dict(BUILTIN_SKINS)[skin_id]}来陪你啦～", autohide_s=4)

    def set_skin(self, image_path: str) -> bool:
        """Install ``image_path`` as the current persona's character (tray menu entry)."""
        from pet.skin import SkinError, install_skin

        try:
            stored = install_skin(image_path, self.persona_id, settings=self.settings)
        except SkinError as exc:
            self.bubble.show_text(f"换形象失败：{exc}", autohide_s=8)
            return False
        except Exception as exc:  # noqa: BLE001
            self.bubble.show_text(f"换形象失败：{exc}", autohide_s=8)
            return False
        self._load_skin(force=True)
        self.update()
        self.bubble.show_text(f"「{self.persona_name}」换好形象了：{stored.name}", autohide_s=5)
        return True

    def clear_skin(self) -> None:
        """Drop this persona's custom image and go back to the drawn character."""
        from pet.skin import clear_skin

        clear_skin(self.persona_id, settings=self.settings)
        # force=True: the path is already gone from settings, so a "nothing changed" early
        # return would leave the old picture on screen (that was the bug).
        self._load_skin(force=True)
        self.update()
        self.bubble.show_text("已经换回默认形象", autohide_s=4)

    def choose_skin_file(self) -> None:
        """Ask for an image with a file dialog, then install it.

        This is the "操作入口" the user asked for: one click in the tray menu, pick a picture,
        done — no config editing, no restart.
        """
        from PySide6.QtWidgets import QFileDialog

        start_dir = ""
        try:
            from pet.skin import skins_dir

            start_dir = str(skins_dir().parent)
        except Exception:  # noqa: BLE001
            pass
        path, _filter = QFileDialog.getOpenFileName(
            None,
            f"给「{self.persona_name}」选一张形象图（PNG 透明背景效果最好）",
            start_dir,
            "图片 (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;所有文件 (*)",
        )
        if path:
            self.set_skin(path)

    def _persona_label(self, persona_id: str) -> str:
        for persona in self.personas:
            if persona.get("id") == persona_id:
                return str(persona.get("name") or persona_id)
        return persona_id

    def select_persona(self, persona_id: str) -> None:
        """Switch persona, respecting the one-pet-per-persona rule.

        Starting a pet twice already refuses to open a second identical pet (see
        ``run_pet.run_gui``); switching persona here is the other way two copies could
        collide, so the same guard is claimed before the switch and the old one released
        after it.
        """
        if self._chat_busy:
            self.bubble.show_text("请等这句话回复完再切换角色。", autohide_s=4)
            return
        if not persona_id or persona_id == self.persona_id:
            return
        label = self._persona_label(persona_id)
        from pet import single_instance

        if not single_instance.acquire(persona_id):
            self.bubble.show_text(f"「{label}」的桌宠已经开着啦，先关掉那个再切过来。", autohide_s=7)
            return
        previous = self._guard_key
        if previous and previous != persona_id:
            single_instance.release(previous)
        self._guard_key = persona_id

        self._voice_epoch += 1
        self.awareness.interact()
        self._action_epoch += 1
        self.player.stop()
        self.persona_id = persona_id
        self._apply_persona()
        self._apply_emotion({})
        persona = next((p for p in self.personas if p["id"] == persona_id), {})
        self.bubble.show_text(persona.get("greeting") or f"已切换到「{self.persona_name}」", autohide_s=8)

    # -- animation ---------------------------------------------------------------------
    def _tick(self) -> None:
        self._phase += 0.06
        self._hover_amount += (float(self._hovered) - self._hover_amount) * 0.16
        self._affection = max(0.0, self._affection - 0.025)
        if self.state == "thinking":
            self._phase += 0.06
        if self._blink > 0:
            self._blink -= 0.08
        elif int(self._phase * 10) % 90 == 0:
            self._blink = 1.0
        self.update()

    # -- painting ----------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.scale(self.scale_factor, self.scale_factor)
        rect = QRectF(6, 6, PET_SIZE - 12, PET_SIZE - 26)
        if self.live2d_view is not None and self.live2d_view.isVisible():
            painter.setPen(QColor("#f0eef8"))
            painter.setFont(QFont("Microsoft YaHei UI", 8))
            painter.drawText(QRectF(4, PET_SIZE * .76, PET_SIZE - 8, 16), Qt.AlignCenter, self.persona_name)
        elif self.skin_pixmap is not None:
            self._draw_skin(painter, rect)
        else:
            self._draw_character(painter, rect, mood=self.state)
        self._draw_mic(painter)
        if hasattr(self, "awareness"):
            painter.setPen(QColor("#cce9dd") if self.awareness.enabled and self.awareness.scene != "quiet" else QColor("#bbbbbb"))
            font = QFont("Microsoft YaHei UI")
            font.setPointSizeF(7.5)
            painter.setFont(font)
            painter.setBrush(QColor(30, 33, 45, 200))
            painter.drawRoundedRect(QRectF(4, PET_SIZE - 33, 78, 30), 6, 6)
            painter.drawText(QRectF(5, PET_SIZE - 33, 76, 30), Qt.AlignCenter, self.asmr.label if self.asmr.active else self.awareness.label.replace(" · ", "\n"))
        if self.hands_free:
            painter.setPen(QPen(QColor(120, 230, 170, 220), 2))
            painter.drawText(QRectF(0, PET_SIZE - 20, PET_SIZE, 16), Qt.AlignCenter, "一直听着")

    def _draw_skin(self, painter: QPainter, rect: QRectF) -> None:
        """Draw the user's image where the built-in character would go.

        The animation and the state feedback survive the swap: the image bobs like the drawn
        character, and thinking/speaking/listening get the same ring and status text, so a
        static picture still looks alive and still shows what the pet is doing.
        """
        pixmap = self.skin_pixmap
        if pixmap is None:
            return
        bob = math.sin(self._phase) * (2.5 if self.state != "thinking" else 4.0)
        # Leave space for the name and mic; reserve a margin for sway and the hop.
        target = rect.adjusted(12, 8, -12, -38).translated(0, bob)
        ratio = min(target.width() / pixmap.width(), target.height() / pixmap.height())
        width, height = pixmap.width() * ratio, pixmap.height() * ratio
        x = target.center().x() - width / 2
        y = target.center().y() - height / 2
        glow = QColor(self.accent)
        glow.setAlpha(0)
        if self.state == "speaking":
            pulse = 0.5 + 0.5 * math.sin(self._phase * 3.2)
            glow.setAlpha(int(60 + 90 * pulse))
        elif self.state in {"listening", "thinking"}:
            glow.setAlpha(110 if self.state == "listening" else 70)
        if glow.alpha():
            gradient = QRadialGradient(target.center(), max(width, height) * 0.58)
            gradient.setColorAt(0, glow)
            gradient.setColorAt(0.65, glow)
            gradient.setColorAt(1, QColor(glow.red(), glow.green(), glow.blue(), 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QRectF(x, y, width, height).adjusted(-8, -8, 8, 8))
        painter.save()
        hop = math.sin((1.0 - self._affection) * math.pi) * 7 * self._affection
        painter.translate(target.center().x(), target.bottom() - hop)
        energy = {"cheerful": 1.4, "playful": 1.6, "caring": 0.55, "calm": 0.8}.get(self.pet_mood, 1.0)
        sway = math.sin(self._phase * 0.7) * 1.8 * energy - self._hover_amount * 3
        if self.state == "speaking":
            sway += math.sin(self._phase * 3.2) * 1.5
        painter.rotate(sway)
        breath = 1 + math.sin(self._phase) * 0.008 * energy
        painter.scale(1 + self._hover_amount * 0.025, breath)
        # Reuse the exact painting transform for mouse hit tests at every zoom level.
        self._sprite_transform = painter.worldTransform()
        self._sprite_size = (width, height)
        painter.drawPixmap(QRectF(-width / 2, -height, width, height), pixmap, QRectF(pixmap.rect()))
        painter.restore()

        # Draw hearts as paths so the effect also works without an emoji font.
        if self._affection > 0:
            painter.save()
            painter.setPen(Qt.NoPen)
            for index in range(3):
                painter.save()
                painter.setOpacity(self._affection * 0.9)
                painter.translate(
                    target.center().x() + (index - 1) * 32,
                    target.top() + 24 + index % 2 * 12 - (1 - self._affection) * 24,
                )
                heart = QPainterPath()
                heart.moveTo(0, 3)
                heart.cubicTo(-12, -4, -6, -12, 0, -6)
                heart.cubicTo(6, -12, 12, -4, 0, 3)
                painter.setBrush(QColor("#ff8fb1"))
                painter.drawPath(heart)
                painter.restore()
            painter.restore()

        # Name tag, same place as the drawn character's.
        font = QFont("Microsoft YaHei UI")
        font.setPointSizeF(9.5)
        painter.setFont(font)
        from core.emotion import PET_LABELS
        label = painter.fontMetrics().elidedText(f"{self.persona_name} · {PET_LABELS.get(self.pet_mood, '平静')}", Qt.ElideRight, int(rect.width() - 20))
        tag_width = painter.fontMetrics().horizontalAdvance(label) + 16
        tag = QRectF(rect.center().x() - tag_width / 2, rect.bottom() - 29, tag_width, 18)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(30, 33, 45, 190))
        painter.drawRoundedRect(tag, 9, 9)
        painter.setPen(QColor(245, 243, 250))
        painter.drawText(tag, Qt.AlignCenter, label)

        if self.state == "thinking":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(235, 238, 250, 200)))
            for index in range(3):
                offset = (self._phase * 2 + index * 0.6) % 3
                alpha = int(120 + 120 * (1 - offset / 3))
                painter.setBrush(QBrush(QColor(235, 238, 250, alpha)))
                painter.drawEllipse(QRectF(rect.center().x() - 18 + index * 14, rect.bottom() - 46, 7, 7))

    def _draw_character(self, painter: QPainter, rect: QRectF, *, mood: str) -> None:
        bob = math.sin(self._phase) * (2.5 if mood != "thinking" else 4.0)
        rect = rect.translated(0, bob)
        body = QPainterPath()
        center_x = rect.center().x()
        body_width = rect.width() * 0.72
        body_height = rect.height() * 0.74
        body_rect = QRectF(
            center_x - body_width / 2, rect.top() + rect.height() * 0.20, body_width, body_height
        )
        body.addRoundedRect(body_rect, body_width * 0.42, body_width * 0.42)

        # ears
        for sign in (-1, 1):
            ear = QPainterPath()
            base_x = center_x + sign * body_width * 0.30
            ear.moveTo(base_x - 14, body_rect.top() + 16)
            ear.lineTo(base_x + sign * 6, body_rect.top() - 34)
            ear.lineTo(base_x + 16, body_rect.top() + 12)
            ear.closeSubpath()
            painter.setBrush(QBrush(self.accent.darker(115)))
            painter.setPen(Qt.NoPen)
            painter.drawPath(ear)

        glow = QColor(self.accent)
        glow.setAlpha(90 if mood == "listening" else 45)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(body_rect.adjusted(-8, -8, 8, 8))

        gradient = QColor(self.accent)
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1.5))
        painter.drawPath(body)

        # face plate
        face = body_rect.adjusted(body_width * 0.16, body_height * 0.20, -body_width * 0.16, -body_height * 0.30)
        painter.setBrush(QBrush(QColor(255, 253, 250, 240)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(face, face.width() * 0.42, face.width() * 0.42)

        # eyes
        eye_y = face.center().y() - face.height() * 0.10
        eye_dx = face.width() * 0.22
        eye_r = face.width() * 0.075
        blink = max(0.12, 1.0 - self._blink)
        painter.setBrush(QBrush(QColor(40, 42, 58)))
        for sign in (-1, 1):
            eye = QRectF(
                face.center().x() + sign * eye_dx - eye_r,
                eye_y - eye_r * blink,
                eye_r * 2,
                eye_r * 2 * blink,
            )
            painter.drawEllipse(eye)
            if blink > 0.5:  # catchlight only when the eye is open
                painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
                painter.drawEllipse(QRectF(eye.center().x(), eye.center().y() - eye_r * 0.4, eye_r * 0.6, eye_r * 0.6))
                painter.setBrush(QBrush(QColor(40, 42, 58)))

        # blush
        blush = QColor(255, 140, 170, 120)
        painter.setBrush(QBrush(blush))
        for sign in (-1, 1):
            painter.drawEllipse(
                QRectF(face.center().x() + sign * eye_dx * 1.55 - 9, eye_y + eye_r * 1.4, 18, 9)
            )

        # mouth: shape encodes what the pet is doing
        mouth = QRectF(face.center().x() - 16, eye_y + eye_r * 2.6, 32, 14)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(60, 62, 82), 2.4, Qt.SolidLine, Qt.RoundCap))
        if mood == "speaking":
            openness = 0.5 + 0.5 * math.sin(self._phase * 3.2)
            painter.setBrush(QBrush(QColor(120, 60, 80)))
            painter.drawEllipse(QRectF(mouth.center().x() - 9, mouth.top() - 2, 18, 6 + 10 * openness))
        elif mood == "listening":
            painter.drawEllipse(QRectF(mouth.center().x() - 6, mouth.center().y() - 5, 12, 11))
        elif mood == "thinking":
            painter.drawLine(mouth.topLeft() + QPoint(6, 4), mouth.topRight() - QPoint(6, -2))
        elif mood == "offline":
            painter.drawLine(mouth.topLeft() + QPoint(4, 8), mouth.topRight() - QPoint(4, 8))
        else:
            painter.drawArc(mouth, 200 * 16, 140 * 16)

        # name tag
        painter.setPen(QPen(QColor(235, 238, 250, 235)))
        # Explicit family: the offscreen platform used by --selftest has no system font
        # database, so an unnamed font renders as tofu boxes there.
        font = QFont("Microsoft YaHei UI")
        font.setPointSizeF(9.5)
        painter.setFont(font)
        tag = QRectF(0, rect.bottom() - 26, rect.width(), 18)
        painter.drawText(tag, Qt.AlignCenter, self.persona_name)

        if mood == "thinking":
            painter.setBrush(QBrush(QColor(235, 238, 250, 200)))
            for index in range(3):
                offset = (self._phase * 2 + index * 0.6) % 3
                alpha = int(120 + 120 * (1 - offset / 3))
                dot = QColor(235, 238, 250, alpha)
                painter.setBrush(QBrush(dot))
                painter.drawEllipse(QRectF(rect.center().x() - 18 + index * 14, rect.bottom() - 46, 7, 7))

    def _draw_mic(self, painter: QPainter) -> None:
        radius = 17.0
        # All drawing and hit testing share the unscaled 210px design coordinates.
        center = QPoint(PET_SIZE // 2, PET_SIZE - 17)
        self._mic_rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
        active = self._recording
        painter.setBrush(QBrush(QColor(255, 92, 122, 235) if active else QColor(30, 33, 45, 210)))
        painter.setPen(QPen(QColor(255, 255, 255, 90), 1.4))
        painter.drawEllipse(self._mic_rect)
        painter.setPen(QPen(QColor(240, 243, 255, 235), 2.0, Qt.SolidLine, Qt.RoundCap))
        # simple microphone glyph
        painter.drawRoundedRect(QRectF(center.x() - 4, center.y() - 8, 8, 12), 4, 4)
        painter.drawArc(QRectF(center.x() - 8, center.y() - 4, 16, 12), 200 * 16, 140 * 16)
        painter.drawLine(center.x(), center.y() + 8, center.x(), center.y() + 11)

    # -- interaction -------------------------------------------------------------------
    def enterEvent(self, event) -> None:  # noqa: N802
        self._hovered = True
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovered = False
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        self.awareness.interact()
        if event.button() == Qt.LeftButton:
            self._click_timer.stop()
            self._press_pos = event.position().toPoint()
            self._press_moved = False
            self._held = False
            point = event.position() / self.scale_factor
            self._press_mic = self._mic_rect.contains(point)
            if self._press_mic:
                self.start_recording()
            else:
                self._touch_kind = self._touch_at(point)
                if self._touch_kind is None:
                    self._press_pos = None
                    return
                self._hold_timer.start(650)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.live2d_view is not None:
            self.live2d_view.look_at(event.position().x(), event.position().y())
        if self._press_pos is not None and event.buttons() & Qt.LeftButton and not self._recording:
            delta = event.position().toPoint() - self._press_pos
            if delta.manhattanLength() > QApplication.startDragDistance():
                self._press_moved = True
                self._hold_timer.stop()
                self._click_timer.stop()
            if self._press_moved:
                self.move(self.pos() + delta)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._hold_timer.stop()
        if event.button() == Qt.LeftButton:
            if self._recording:
                self.stop_recording_and_send()
            elif self._press_pos is not None and not self._press_moved and not self._held and not getattr(self, "_press_mic", False):
                self._affection = 1.0
                self._click_timer.start(QApplication.doubleClickInterval())
            elif self._press_moved:
                self._clamp_to_screen()
            self._press_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        point = event.position() / self.scale_factor
        if (event.button() == Qt.LeftButton and not self._mic_rect.contains(point)
                and self._touch_at(point) is not None):
            self._click_timer.stop()
            self._hold_timer.stop()
            self._held = True
            self.touch("double")
            self.bubble.ask()
        event.accept()

    def wheelEvent(self, event) -> None:
        delta = event.pixelDelta().y() / 45 if event.pixelDelta().y() else event.angleDelta().y() / 120
        if delta:
            # Keep the point under the pointer fixed while smoothly approaching the target.
            self._scale_anchor = (event.globalPosition(), event.position() / self.scale_factor)
            self._scale_target = max(MIN_SCALE, min(MAX_SCALE, self._scale_target + delta * 0.02))
            self._scale_timer.start()
            self._scale_save_timer.start(350)
        event.accept()

    def _animate_scale(self) -> None:
        remaining = self._scale_target - self.scale_factor
        value = self._scale_target if abs(remaining) < 0.0008 else self.scale_factor + remaining * 0.28
        self._apply_scale(value, anchor=self._scale_anchor)
        if value == self._scale_target:
            self._scale_timer.stop()

    def _touch_at(self, point):
        if self.skin_pixmap is not None and hasattr(self, "_sprite_transform") and self.live2d_view is None:
            inverse, valid = self._sprite_transform.inverted()
            if valid:
                local = inverse.map(point * self.scale_factor)
                width, height = self._sprite_size
                x, y = (local.x() + width / 2) / width, (local.y() + height) / height
                image = self.skin_pixmap.toImage()
                if not (0 <= x < 1 and 0 <= y < 1) or image.pixelColor(int(x * image.width()), int(y * image.height())).alpha() < 20:
                    return None
                mouth_x, mouth_y = {"sakura_cat": (0.52, 0.38), "mint_bunny": (0.52, 0.425),
                                    "luna_witch": (0.44, 0.39)}.get(self.persona_id, (0.5, 0.4))
                if abs(x - mouth_x) < 0.055 and abs(y - mouth_y) < 0.045:
                    return "feed"
                return "head" if y < mouth_y - 0.1 else "face" if y < mouth_y + 0.07 else "body"
        if point.y() < 65:
            return "head"
        # The mouth is a small central target; cheeks remain on either side.
        if 90 <= point.x() <= 120 and 90 <= point.y() <= 112:
            return "feed"
        return "face" if point.y() < 108 else "body"

    def _hold_touch(self) -> None:
        if self._press_pos is not None and not self._press_moved:
            self._held = True
            self.touch("hold")

    def touch(self, kind: str) -> None:
        if self.state in {"thinking", "speaking"} or self._recording:
            return
        persona = next((p for p in self.personas if p.get("id") == self.persona_id), {})
        if not persona:
            from persona.catalog import CATALOG
            persona = CATALOG.get(self.persona_id, {})
        line = persona.get("options", {}).get("touch", {}).get(kind, "我在这里，陪你一起休息一会儿。")
        self._affection = 1.0
        self.bubble.show_text(line, autohide_s=5)
        self.pet_mood = "playful" if kind in {"face", "feed", "double"} else "caring" if kind == "hold" else "cheerful"
        if self.live2d_view is not None:
            self.live2d_view.react(kind)
        # Limit repeated taps; they must not queue many audio requests.
        if not self.muted and time.monotonic() - self._last_touch_at > 3:
            self._last_touch_at = time.monotonic()
            self.speak(line)

    def set_scale(self, value, *, persist=True) -> None:
        try:
            scale = float(value)
            if not math.isfinite(scale):
                scale = 1.0
        except (TypeError, ValueError):
            scale = 1.0
        self._scale_timer.stop()
        self._scale_save_timer.stop()
        self._scale_target = max(MIN_SCALE, min(MAX_SCALE, scale))
        self._apply_scale(self._scale_target)
        if persist:
            self.settings.set("scale", self._scale_target)

    def _apply_scale(self, scale, *, anchor=None) -> None:
        bottom_right = self.geometry().bottomRight()
        self.scale_factor = max(MIN_SCALE, min(MAX_SCALE, scale))
        size = round(PET_SIZE * self.scale_factor)
        self.setFixedSize(size, size)
        if anchor:
            screen_point, local_point = anchor
            self.move((screen_point - local_point * self.scale_factor).toPoint())
        else:
            self.move(bottom_right.x() - size + 1, bottom_right.y() - size + 1)
        if self.live2d_view is not None:
            self.live2d_view.setGeometry(0, 0, size, round(size * 0.76))
        self._clamp_to_screen()
        self.update()

    def _clamp_to_screen(self) -> None:
        screen = QApplication.screenAt(self.frameGeometry().center()) or QApplication.primaryScreen()
        if screen:
            area = screen.availableGeometry()
            self.move(max(area.left(), min(self.x(), area.right() - self.width() + 1)),
                      max(area.top(), min(self.y(), area.bottom() - self.height() + 1)))
        self._reposition_bubble()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space and event.modifiers() & Qt.ControlModifier:
            self.start_recording()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space and self._recording:
            self.stop_recording_and_send()
        super().keyReleaseEvent(event)

    def moveEvent(self, event) -> None:  # noqa: N802
        self._reposition_bubble()
        super().moveEvent(event)

    def _reposition_bubble(self) -> None:
        """Keep the bubble glued above the pet, inside the screen the pet is on."""
        if not self.bubble.isVisible():
            return
        self.bubble.adjustSize()
        width, height = self.bubble.width(), self.bubble.height()
        # screenAt(), not primaryScreen(): the pet can be dragged to a second monitor, and a
        # bubble clamped to the primary screen would jump to the wrong display.
        screen = QApplication.screenAt(self.frameGeometry().center()) or QApplication.primaryScreen()
        x = self.x() + self.width() // 2 - width // 2
        y = self.y() - height - 4
        if screen is not None:
            area = screen.availableGeometry()
            x = max(area.left() + 8, min(x, area.right() - width - 8))
            y = max(area.top() + 8, min(y, area.bottom() - height - 8))
        self.bubble.move(x, y)

    # -- voice -------------------------------------------------------------------------
    def start_recording(self) -> None:
        self.asmr.stop()
        if self._recording or self.state == "thinking":
            return
        try:
            self.player.stop()
            self.recorder.start()
        except AudioUnavailable as exc:
            self.bubble.show_text(f"麦克风不可用：{exc}", autohide_s=8)
            return
        self._recording = True
        self._set_state("listening")
        self.bubble.show_text("我在听…（松开结束）")
        self.update()

    def stop_recording_and_send(self) -> None:
        if not self._recording:
            return
        self._recording = False
        wav = self.recorder.stop()
        if len(wav) < 2000:
            self._set_state("idle")
            self.bubble.show_text("没听到声音，再说一次？", autohide_s=4)
            return
        self._set_state("thinking")
        self.bubble.show_text("听着呢，让我想想…")

        def work() -> str:
            text = self.client.transcribe(wav)
            if not text:
                raise BackendError("没有识别到内容")
            return text

        self._run_async(work, self._on_transcript, self._on_error)

    def _on_transcript(self, text: str) -> None:
        self.bubble.show_text(f"你：{text}", autohide_s=3)
        self.send_text(text)

    # -- chat --------------------------------------------------------------------------
    def send_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text or self._chat_busy:
            return
        if self.asmr.handle_command(text):
            self.bubble.input.hide()
            return
        self.asmr.stop()
        self._chat_busy = True
        self.awareness.interact()
        persona_id = self.persona_id
        self.bubble.input.hide()
        self.bubble.show_text(text)
        self._set_state("thinking")
        collected: List[str] = []
        emotions = {}

        def work() -> str:
            final = ""
            for event in self.client.chat_stream(text, persona_id=persona_id):
                if event.kind == "text" and event.text:
                    collected.append(event.text)
                elif event.kind == "emotion":
                    emotions.update(event.data.get("emotion") or {})
                elif event.kind == "done":
                    final = event.text or "".join(collected)
            return final or "".join(collected)

        def done(reply: str) -> None:
            self._chat_busy = False
            if self.persona_id == persona_id:
                self._apply_emotion(emotions)
            self.bubble.show_text(reply or "……")
            self._set_state("idle")
            if reply and self.persona_id == persona_id:
                self.plan_actions(reply)
            if reply and not self.muted:
                self.speak(reply)

        self._run_async(work, done, self._on_error)

    def speak(self, text: str) -> None:
        if self.asmr.active:
            return
        persona_id, epoch = self.persona_id, self._voice_epoch
        def work() -> int:
            count = 0
            for audio, _meta in self.client.speak_stream(text, persona_id=persona_id):
                if epoch != self._voice_epoch or self.muted:
                    break
                self.player.enqueue(audio)
                count += 1
            return count

        self._run_async(work, lambda _n: None, lambda exc: log.warning("speech failed", extra={"error": str(exc)}))

    def _on_error(self, exc: Exception) -> None:
        self._chat_busy = False
        message = str(exc)
        if isinstance(exc, BackendError):
            self.bubble.show_text(message, autohide_s=10)
            self._set_state("offline" if "连不上" in message else "idle")
        else:
            self.bubble.show_text(f"出错了：{message}", autohide_s=10)
            self._set_state("idle")

    # -- hands-free --------------------------------------------------------------------
    def toggle_hands_free(self) -> None:
        """One click, then just talk: record continuously, cut sentences on silence.

        The decision logic lives in :mod:`pet.hands_free` (and is unit-tested there); this
        method only starts/stops the machinery and tells the user what happened.
        """
        if not self.hands_free:
            self.asmr.stop()
        self.hands_free = not self.hands_free
        if self.hands_free:
            self._hands_free_hint_shown = False
            self.listener.enable()
            self._silence_timer.start(int(self.listener.config.poll_s * 1000))
            self.bubble.show_text(f"好，我一直听着（直接说话，右键可关闭）\n{describe(self.listener.config)}", autohide_s=6)
            self._resume_hands_free()
        else:
            was_speaking = self.listener.phase == PHASE_SPEECH
            self._silence_timer.stop()
            self.listener.disable()
            # Mid-sentence when switched off? Keep what was said; otherwise drop the take
            # instead of sending a clip that transcribes to nothing.
            if self._recording:
                if was_speaking or self.recorder.peak > self.listener.config.speech_level:
                    self.stop_recording_and_send()
                else:
                    self._discard_recording()
            self.bubble.show_text("不一直听着了", autohide_s=3)

    def _discard_recording(self) -> None:
        """Throw the current take away without sending it."""
        if not self._recording:
            return
        self._recording = False
        self.recorder.stop()
        self._set_state("idle")

    def _resume_hands_free(self) -> None:
        """Open the mic again after a turn — the piece the old code kept losing.

        Restarting used to be a fire-and-forget ``QTimer.singleShot(1200, start_recording)``,
        which silently did nothing whenever the reply was still being generated (that guard is
        the first line of ``start_recording``), so hands-free stopped working after one
        exchange. Now: only re-arm when the pet is genuinely idle, and re-arm *every* time it
        becomes idle while the mode is on.
        """
        if not self.hands_free:
            return
        if self.state in {"thinking", "speaking"}:
            return
        if self._recording:
            self.listener.resume()
            return
        try:
            self.player.stop()
            self.recorder.start()
        except AudioUnavailable as exc:
            self.hands_free = False
            self._silence_timer.stop()
            self.listener.disable()
            self.bubble.show_text(f"麦克风不可用：{exc}", autohide_s=8)
            return
        self._recording = True
        self._set_state("listening")
        self.listener.arm()
        # Same reason as in _pause_hands_free: forget the reply the pet just spoke.
        self.recorder.retain_last(0.3)
        self.update()

    def _hands_free_tick(self) -> None:
        if not self.hands_free or not self._recording:
            return
        if self.listener.paused:
            return
        # A Qt timer slot that raises is fatal (PySide turns it into qFatal), so a bug in the
        # state machine would take the whole pet down instead of just mis-detecting speech —
        # exactly what a user reported as "切换灵敏度之后语音识别失败". Degrade, don't die.
        try:
            level = self.recorder.level()
            event = self.listener.tick(level)
        except Exception as exc:  # noqa: BLE001
            log.warning("hands-free tick failed;重新开始收音", extra={"error": str(exc)})
            self.listener.arm()
            return
        if event == EVENT_SPEECH_STARTED:
            # Trim the silence that came before the sentence: the recogniser is faster and
            # more accurate on a clip that starts at the voice, and the buffer stays bounded
            # however long the pet sits waiting.
            self.recorder.retain_last(0.3)
            self.bubble.show_text("我在听…（停下来就会自动发出去）")
            return
        if event == EVENT_SEND:
            self.stop_recording_and_send()
            return
        # Nothing for a long time: say so, once, instead of listening silently forever. This is
        # the "switched sensitivity and now it never recognises me" case — the thresholds are
        # higher than this microphone's voice level.
        if not self._hands_free_hint_shown and self.listener.seems_inaudible(time.monotonic()):
            self._hands_free_hint_shown = True
            self.bubble.show_text(
                "一直没听到声音。可能麦克风太小/太远，或灵敏度档位偏高：\n"
                f"右键 →「免提灵敏度」（当前触发值 {self.listener.trigger_level():.3f}）\n"
                "也可以先跑：python tools\\check_hands_free.py --calibrate",
                autohide_s=12,
            )

    def _pause_hands_free(self) -> None:
        """Stop reacting while the pet thinks/speaks — otherwise it answers its own voice."""
        if not self.hands_free:
            return
        self.listener.pause()
        # The mic stays open (no device restart gap), so drop the audio the pet is about to
        # hear itself play: what gets recognised must be the user's voice only.
        self.recorder.retain_last(0.3)

    # -- menu / tray -------------------------------------------------------------------
    def _menu(self, *, from_pet: bool = False) -> QMenu:
        menu = QMenu(self)
        self.asmr.add_menu(menu)
        scenes = menu.addMenu("陪伴场景")
        from core.companion import SCENES
        for scene, spec in SCENES.items():
            action = scenes.addAction(spec["label"])
            action.setCheckable(True)
            action.setChecked(self.awareness.scene == scene)
            action.triggered.connect(lambda checked=False, value=scene: self.awareness.set_scene(value))
        aware = menu.addAction("主动感知（屏幕 / 活动 / 媒体）")
        aware.setCheckable(True)
        aware.setChecked(self.awareness.enabled)
        aware.triggered.connect(self.awareness.set_enabled)
        screen_action = menu.addAction("允许观察当前窗口画面（仅本机模型）")
        screen_action.setCheckable(True)
        screen_action.setChecked(self.awareness.screen_enabled)
        screen_action.triggered.connect(self.awareness.toggle_screen)
        menu.addAction("感知状态 / 隐私说明").triggered.connect(self.show_awareness)
        menu.addSeparator()
        type_action = QAction("打字聊天", menu)
        type_action.triggered.connect(self.bubble.ask)
        menu.addAction(type_action)

        talk = QAction("按住说话（Ctrl+Space）", menu)
        talk.setEnabled(False)
        menu.addAction(talk)

        hands = QAction("一直听着（自动断句）", menu)
        hands.setCheckable(True)
        hands.setChecked(self.hands_free)
        hands.triggered.connect(self.toggle_hands_free)
        menu.addAction(hands)

        sensitivity_menu = menu.addMenu("免提灵敏度")
        for key, label in SENSITIVITY_LABELS.items():
            action = QAction(label, sensitivity_menu)
            action.setCheckable(True)
            action.setChecked(self.sensitivity == key)
            action.triggered.connect(lambda _checked=False, name=key: self.set_sensitivity(name))
            sensitivity_menu.addAction(action)

        size_menu = menu.addMenu("桌宠大小（也可滚轮缩放）")
        for value in (0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
            action = size_menu.addAction(f"{value:.0%}")
            action.setCheckable(True)
            action.setChecked(abs(self.scale_factor - value) < 0.01)
            action.triggered.connect(lambda checked=False, v=value: self.set_scale(v))
        menu.addAction("角色介绍 / 初始记忆").triggered.connect(self.show_introduction)
        menu.addAction("角色音色 / 参考音频…").triggered.connect(self.configure_voice)
        menu.addAction("模型调用：本地部署 / 外部 API…").triggered.connect(self.configure_model_connection)
        live_menu = menu.addMenu("Live2D")
        live_menu.addAction("为当前角色导入模型…").triggered.connect(self.choose_live2d)
        live_menu.addAction("恢复角色立绘").triggered.connect(self.clear_live2d)

        # -- 底座模型 --
        model_menu = menu.addMenu("底座模型")
        if self.runtime_migration and (self.runtime_migration.get("state") != "ready" or self.model_connection.get("backend") != "vllm"):
            pending = model_menu.addAction("vLLM 迁移：" + self.runtime_migration.get("detail", "等待安装"))
            pending.setEnabled(False)
            model_menu.addAction("重启电脑后：继续安装 vLLM").triggered.connect(self.resume_vllm_install)
        if not self.model_tiers:
            loading = QAction("外部 API 模式" if self.model_connection.get("mode") == "api" else "（正在读取…）", model_menu)
            loading.setEnabled(False)
            model_menu.addAction(loading)
        else:
            for tier in self.model_tiers:
                downloaded = bool(tier.get("downloaded"))
                label = f"{tier.get('id')}　{tier.get('size_gb')}GB"
                if tier.get("expected_tps"):
                    label += f"　{tier.get('expected_tps')}"
                action = QAction(label, model_menu)
                action.setCheckable(True)
                action.setChecked(bool(tier.get("current")))
                action.setEnabled(downloaded and tier.get("runnable", True) and not self._switching_model)
                if not downloaded:
                    action.setText(f"{label}　（未下载）")
                action.triggered.connect(
                    lambda _checked=False, tier_id=str(tier.get("id")): self.switch_model(tier_id)
                )
                model_menu.addAction(action)
            model_menu.addSeparator()
            refresh = QAction("重新读取档位", model_menu)
            refresh.triggered.connect(self.refresh_models)
            model_menu.addAction(refresh)
            hint = QAction(f"当前：{self.model_current or '未知'}｜切换需 30~120 秒", model_menu)
            hint.setEnabled(False)
            model_menu.addAction(hint)

        # -- 形象 --
        look_menu = menu.addMenu("更换形象")
        from pet.skin import BUILTIN_SKINS, builtin_skin_path

        for skin_id, label in BUILTIN_SKINS:
            path = builtin_skin_path(skin_id)
            action = QAction(QIcon(str(path)) if path else QIcon(), label, look_menu)
            action.setEnabled(path is not None)
            action.setCheckable(True)
            action.setChecked(path is not None and str(path) == self.skin_path)
            action.triggered.connect(lambda checked=False, key=skin_id: self.choose_builtin_skin(key))
            look_menu.addAction(action)
        look_menu.addSeparator()
        pick = QAction("选一张图片…", look_menu)
        pick.triggered.connect(self.choose_skin_file)
        look_menu.addAction(pick)

        open_folder = QAction("打开形象文件夹", look_menu)
        open_folder.triggered.connect(self.open_skin_folder)
        look_menu.addAction(open_folder)

        reset = QAction("换回默认形象", look_menu)
        reset.setEnabled(bool(self.skin_path))
        reset.triggered.connect(self.clear_skin)
        look_menu.addAction(reset)

        look_menu.addSeparator()
        hint = QAction("建议：PNG 透明背景、正方形、512×512 以上", look_menu)
        hint.setEnabled(False)
        look_menu.addAction(hint)

        mute = QAction("静音朗读", menu)
        mute.setCheckable(True)
        mute.setChecked(self.muted)
        mute.triggered.connect(self._toggle_mute)
        menu.addAction(mute)

        menu.addSeparator()
        # 「启动本地服务」原来是个常驻项，而且不看状态就无条件跑 start-all.ps1 —— 用户的
        # 反馈很准：「本身就是错误的，而且很多余」。正常入口是双击 启动聊天机器人.exe，
        # 它已经负责把服务拉起来；对着一个健康的服务再点一次，只会把网页那边的连接掐断
        # （浏览器里就变成 "Failed to fetch"）。所以现在：只在**确实连不上**时才出现，
        # 而且点下去之前还要再探一次健康接口。
        if self.state == "offline":
            start_backend = QAction("本地服务没在运行 → 启动它", menu)
            start_backend.triggered.connect(self.start_backend)
            menu.addAction(start_backend)
            hint = QAction("（平时双击 启动聊天机器人.exe 即可）", menu)
            hint.setEnabled(False)
            menu.addAction(hint)

        reconnect = QAction("重新连接", menu)
        reconnect.triggered.connect(self.refresh_personas)
        menu.addAction(reconnect)

        if from_pet:
            web = QAction("打开网页", menu)
            web.triggered.connect(self.open_web_ui)
            menu.addAction(web)

        hide = QAction("隐藏到托盘", menu)
        hide.triggered.connect(self.hide)
        menu.addAction(hide)

        quit_action = QAction("退出桌宠", menu)
        quit_action.triggered.connect(self.quit)
        menu.addAction(quit_action)
        stop_action = menu.addAction("退出并停止全部服务")
        stop_action.triggered.connect(self.stop_all)
        return menu

    def _show_menu(self, position: QPoint) -> None:
        menu = self._menu(from_pet=True)
        menu.exec(self.mapToGlobal(position))
        menu.deleteLater()

    def _tray_activated(self, reason) -> None:  # noqa: ANN001
        if reason == QSystemTrayIcon.Trigger:
            if self.isVisible():
                self.hide()
                self.bubble.hide()
            else:
                self.show()
                self.raise_()

    # -- base model --------------------------------------------------------------------
    def refresh_models(self) -> None:
        """Read the available base-model tiers (runs on a worker thread)."""

        def work() -> Dict[str, Any]:
            return self.client.models()

        def done(payload: Dict[str, Any]) -> None:
            self.model_connection = dict(payload.get("connection") or {})
            self.runtime_migration = dict(payload.get("runtime_migration") or payload.get("installation") or {})
            self.model_tiers = list(payload.get("tiers") or [])
            self.model_current = payload.get("current")
            self._switching_model = bool(payload.get("busy"))

        def failed(exc: Exception) -> None:
            log.debug("could not read model tiers", extra={"error": str(exc)})

        self._run_async(work, done, failed)

    def switch_model(self, tier_id: str) -> None:
        """Switch the base model from the tray menu.

        The switch stops and restarts llama-server (30–120 s), so it runs on a worker thread
        and the pet says what is happening — a frozen menu with no feedback looks like a crash.
        """
        if self._switching_model:
            self.bubble.show_text("上一次切换还没结束，稍等一下…", autohide_s=5)
            return
        self._switching_model = True
        self.bubble.show_text(f"正在切换到「{tier_id}」…\n需要 30~120 秒（要重新加载权重）", autohide_s=0)

        def work() -> Dict[str, Any]:
            return self.client.switch_model(tier_id)

        def done(result: Dict[str, Any]) -> None:
            self._switching_model = False
            self.bubble.show_text(f"已经切换到「{result.get('tier')}」了，继续聊吧。", autohide_s=6)
            self.refresh_models()

        def failed(exc: Exception) -> None:
            self._switching_model = False
            self.bubble.show_text(f"切换失败：{exc}", autohide_s=15)
            self.refresh_models()

        self._run_async(work, done, failed)

    def open_skin_folder(self) -> None:
        """Show where skins live, so the user can drop images in by hand as well."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from pet.skin import skins_dir

        try:
            folder = skins_dir()
        except Exception as exc:  # noqa: BLE001
            self.bubble.show_text(f"打不开形象文件夹：{exc}", autohide_s=8)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        self.bubble.show_text(
            "把图片放进这个文件夹，然后用右键菜单「选一张图片…」导入（会自动裁剪成方形）",
            autohide_s=8,
        )

    def show_introduction(self) -> None:
        persona = next((p for p in self.personas if p.get("id") == self.persona_id), {})
        intro = persona.get("initial_memory") or persona.get("description") or "正在读取角色介绍…"
        if self.persona_id in {"hiyori", "mao"}:
            intro += "\n\n形象版权：Live2D Inc.，依官方免费素材与角色条款使用；本应用的陪伴设定为独立演绎。许可见 docs/Live2D资源许可.md。"
        self.bubble.show_text(intro)

    def resume_vllm_install(self):
        import subprocess
        from pet.settings import project_root
        root = project_root()
        def work():
            with (root / "logs/vllm-install.log").open("ab") as log_file:
                process = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / "scripts/install-vllm.ps1")],
                                         cwd=root, stdout=log_file, stderr=log_file, stdin=subprocess.DEVNULL,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return process.returncode
        self.bubble.show_text("正在准备 vLLM。安装进度记录在 logs/vllm-install.log；需要重启时会停止等待，不会自行重启电脑。", autohide_s=12)
        def done(code):
            self.bubble.show_text("vLLM 已安装并激活。" if code == 0 else ("请先重启 Windows，再使用此菜单继续安装。" if code == 3010 else "安装未完成，请查看 logs/vllm-install.log。"), autohide_s=12)
            self.refresh_models()
        self._run_async(work, done, self._on_error)

    def configure_voice(self) -> None:
        from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QPushButton
        persona = next((p for p in self.personas if p.get("id") == self.persona_id), None)
        if not persona:
            self.bubble.show_text("连接服务后即可设置角色音色。", autohide_s=5)
            return
        voice = dict(persona.get("voice") or {})
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self.persona_name}的音色")
        dialog.setMinimumWidth(420)
        layout = QFormLayout(dialog)
        backend = QComboBox()
        for label, key in (("神经语音（即用，需联网）", "edge"), ("IndexTTS2（本地参考音色）", "indextts"), ("GPT-SoVITS（本地角色音色）", "gpt_sovits")):
            backend.addItem(label, key)
        backend.setCurrentIndex(max(0, backend.findData(voice.get("backend", "edge"))))
        voice_id = QLineEdit(voice.get("voice_id", "zh-CN-XiaoxiaoNeural"))
        reference = QLineEdit(voice.get("reference_audio") or "")
        transcript = QLineEdit(voice.get("reference_text") or "")
        pick = QPushButton("选择参考音频…")
        def choose():
            path, _ = QFileDialog.getOpenFileName(dialog, "选择角色参考音频", "", "音频 (*.wav *.mp3 *.flac)")
            if path:
                reference.setText(path)
        pick.clicked.connect(choose)
        layout.addRow("语音路径", backend)
        layout.addRow("神经音色名称", voice_id)
        layout.addRow("参考音频", reference)
        layout.addRow(pick)
        layout.addRow("参考音频台词", transcript)
        layout.addRow(QLabel("本地音色需要先安装并启动相应引擎。\n保留当前角色的语速、音高和情感设置。"))
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() == QDialog.Accepted:
            voice.update(backend=backend.currentData(), voice_id=voice_id.text().strip(),
                         reference_audio=reference.text().strip() or None,
                         reference_text=transcript.text().strip() or None)
            payload = dict(persona, voice=voice)
            def done(_):
                self._voice_epoch += 1
                self.player.stop()
                self.refresh_personas()
                self.bubble.show_text("角色音色已保存，下次回复将使用新设置。", autohide_s=5)
            self._run_async(lambda: self.client.update_persona(payload), done, self._on_error)
        dialog.deleteLater()

    def configure_model_connection(self) -> None:
        from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout
        dialog = QDialog(self)
        dialog.setWindowTitle("模型调用模式")
        dialog.setMinimumWidth(430)
        layout = QFormLayout(dialog)
        mode = QComboBox()
        mode.addItem("本地部署（vLLM / 其他兼容服务）", "local")
        mode.addItem("外部 API（兼容 Chat Completions）", "api")
        current = self.model_connection
        mode.setCurrentIndex(1 if current.get("mode") == "api" else 0)
        url = QLineEdit(current.get("base_url") or "http://127.0.0.1:8080/v1")
        model = QLineEdit(current.get("model") or "MiMo-V2.6-Distill-Qwen-9B-INT4-W4A16-AutoRound")
        key = QLineEdit()
        key.setEchoMode(QLineEdit.Password)
        key.setPlaceholderText("同一地址留空保留已保存的密钥")
        def preset(index):
            url.setText("http://127.0.0.1:8080/v1" if index == 0 else "")
            model.setText("MiMo-V2.6-Distill-Qwen-9B-INT4-W4A16-AutoRound" if index == 0 else "")
            key.clear()
        mode.currentIndexChanged.connect(preset)
        layout.addRow("调用模式", mode)
        layout.addRow("接口根地址（含 /v1）", url)
        layout.addRow("模型名称", model)
        layout.addRow("API Key", key)
        note = QLabel("外部 API 会发送聊天上下文和本轮召回记忆；语音仍使用角色音色。\n本地服务需先启动；保存后新对话立即使用该连接。")
        note.setWordWrap(True)
        layout.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() == QDialog.Accepted:
            values = {"mode": mode.currentData(), "base_url": url.text().strip(),
                      "model": model.text().strip(), "api_key": key.text()}
            def done(payload):
                self.model_connection = payload["connection"]
                self.bubble.show_text("模型连接已保存。可以开始聊天了。", autohide_s=5)
                self.refresh_models()
            self._run_async(lambda: self.client.set_model_connection(values), done, self._on_error)
        dialog.deleteLater()

    def _load_live2d(self) -> None:
        path = (self.settings.get("live2d_models") or {}).get(self.persona_id)
        if path == self._live2d_path:
            return
        if self.live2d_view is not None:
            self.live2d_view.release()
            self.live2d_view = None
        self._live2d_path = path
        if not path:
            return
        try:
            from pet.live2d import create_view, validate_model
            valid = validate_model(path)
            self.live2d_view = create_view(self, valid, self._live2d_failed)
            self.live2d_view.setGeometry(0, 0, self.width(), round(self.height() * 0.76))
            self.live2d_view.show()
        except Exception as exc:
            self._live2d_failed(str(exc))

    def _live2d_ready(self):
        if self.live2d_view and not self.live2d_view.failed:
            pixmap = QPixmap.fromImage(self.live2d_view.grabFramebuffer())
            if not pixmap.isNull():
                icon = QIcon(pixmap)
                self.setWindowIcon(icon)
                if self.tray:
                    self.tray.setIcon(icon)

    def plan_actions(self, text):
        if not self.live2d_view or not self.live2d_view.capabilities:
            return
        self._action_epoch += 1
        epoch, persona = self._action_epoch, self.persona_id
        payload = {"text": text[:1000], "persona_id": persona or "", "capabilities": self.live2d_view.capabilities}
        def work():
            response = self.client._client.post("/api/pet/action", json=payload, timeout=12)
            response.raise_for_status()
            return response.json()
        def done(plan):
            if epoch == self._action_epoch and self.persona_id == persona and self.live2d_view:
                self.live2d_view.set_plan(plan)
        self._run_async(work, done, lambda exc: None)

    def show_awareness(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "感知状态", f"{self.awareness.label}\n最近判断：{self.awareness.last_reason}\n系统声音：{self.awareness.audio.status}\n\n"
            "屏幕仅按间隔观察当前前台窗口的缩小截图，保存在内存中并仅发送本机模型。关闭开关立即停止后续采集并丢弃待返回结果。"
            "输入活动只统计空闲时间、活动次数、鼠标移动及磁盘吞吐，不读取键入内容。"
            "电影/音乐模式读取系统媒体标题、播放状态及扬声器输出的音量包络，不开启麦克风，不录制系统声音文件。"
            "密码/银行等窗口标题会暂停采集，这不是完整的敏感内容识别；处理私密内容时请主动暂停感知。"
            "观察与推测不写入聊天历史或长期记忆；屏幕情节不是你的真实情绪。每小时最多主动发言4次，勿扰模式完全暂停。")

    def _live2d_failed(self, error) -> None:
        if self.live2d_view is not None:
            self.live2d_view.release()
            self.live2d_view = None
        self.bubble.show_text(f"Live2D 暂不可用，继续使用角色立绘。\n{error}\n运行库安装：pip install -r requirements-live2d.txt", autohide_s=12)
        self.update()

    def choose_live2d(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        from pet.live2d import validate_model
        path, _ = QFileDialog.getOpenFileName(self, f"为{self.persona_name}导入匹配的 Live2D 模型", "", "Cubism 模型 (*.model3.json)")
        if not path:
            return
        try:
            path = validate_model(path)
        except Exception as exc:
            self.bubble.show_text(str(exc), autohide_s=8)
            return
        models = dict(self.settings.get("live2d_models") or {})
        models[self.persona_id] = path
        self.settings.set("live2d_models", models)
        self._live2d_path = None
        self._load_live2d()

    def clear_live2d(self) -> None:
        models = dict(self.settings.get("live2d_models") or {})
        models.pop(self.persona_id, None)
        self.settings.set("live2d_models", models)
        self._load_live2d()
        self.update()

    def set_sensitivity(self, name: str) -> None:
        """Switch the hands-free noise threshold preset (tray menu)."""
        config = self.listener.set_sensitivity(name)
        self.sensitivity = name
        self.settings.set("hands_free_sensitivity", name)
        self.bubble.show_text(
            f"免提灵敏度：{SENSITIVITY_LABELS.get(name, name)}\n{describe(config)}",
            autohide_s=6,
        )

    def _toggle_mute(self) -> None:
        self._voice_epoch += 1
        self.muted = not self.muted
        self.settings.set("muted", self.muted)
        if self.muted:
            self.player.stop()

    def open_web_ui(self) -> None:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl

        QDesktopServices.openUrl(QUrl(self.client.base_url + "/"))

    def start_backend(self) -> None:
        """Bring the local service up — but only if it is really down.

        The pet can be opened before the backend (or after it crashed, which is how the user
        ends up with "Failed to fetch" in the browser), so it stays able to start it. What it
        must *not* do is run ``start-all.ps1`` against a healthy service: that restarts
        processes and drops the web UI's connection. Hence the health probe first.
        """
        import subprocess
        from pathlib import Path

        try:
            if self.client.online():
                self.bubble.show_text("本地服务已经在运行，不需要重启。", autohide_s=4)
                self._set_state("idle" if self.state == "offline" else self.state)
                self.refresh_personas()
                return
        except Exception as exc:  # noqa: BLE001 - probe failure means "probably down"
            log.debug("health probe before start failed", extra={"error": str(exc)})

        script = Path(__file__).resolve().parents[2] / "scripts" / "start-all.ps1"
        if not script.exists():
            self.bubble.show_text(f"找不到启动脚本：{script}", autohide_s=8)
            return
        self.bubble.show_text("正在启动本地服务（首次加载模型约 1 分钟）…")
        try:
            subprocess.Popen(  # noqa: S603 - fixed local script, no user input
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-WindowStyle",
                    "Hidden",
                    "-File",
                    str(script),
                    "-NoBrowser",
                ],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:  # noqa: BLE001
            self.bubble.show_text(f"启动失败：{exc}", autohide_s=10)
            return
        QTimer.singleShot(25000, self.refresh_personas)

    def quit(self) -> None:
        try:
            self.asmr.stop()
            self.awareness.close()
            self._scale_save_timer.stop()
            self.settings.set("scale", self._scale_target)
            self.player.stop()
            if self._recording:
                self.recorder.stop()
            self.client.close()
        finally:
            if self.tray is not None:
                self.tray.hide()
            QApplication.quit()

    def stop_all(self) -> None:
        from pet.bootstrap import stop_services
        try:
            stop_services()
        except Exception as exc:
            self._on_error(exc)
            return
        self.quit()
