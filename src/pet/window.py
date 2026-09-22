"""The desktop pet: an always-on-top character that listens and talks.

Design notes, because a desktop pet fails in ways a normal window does not:

* **Frameless + translucent + always-on-top**, dragged by the body. It must never steal
  focus while the user is typing elsewhere, so it is created with ``Qt.Tool`` and
  ``WindowDoesNotAcceptFocus``; only clicking it (or opening the input box) focuses it.
* **The character is drawn, not loaded.** No art assets to ship, no licensing question,
  and it scales to any DPI. Personas tint it, so switching persona visibly switches pet.
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

from PySide6.QtCore import QPoint, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
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
        self._press_pos: Optional[QPoint] = None
        self._press_moved = False
        self._recording = False
        self._mic_rect = QRectF()
        self.bubble = SpeechBubble()
        self.bubble.submitted.connect(self.send_text)
        # Every show/resize repositions the bubble above the pet (see SpeechBubble.on_shown).
        self.bubble.on_shown = self._reposition_bubble

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
        x = area.right() - PET_SIZE - 40
        y = area.bottom() - PET_SIZE - 60
        try:
            from pet import single_instance

            # pet_window_rects() comes from Win32 and is in *physical* pixels, while
            # everything here is Qt logical coordinates (see _to_logical).
            taken = _to_logical(single_instance.pet_window_rects(), float(screen.devicePixelRatio() or 1.0))
        except Exception as exc:  # noqa: BLE001 - placement must never stop the pet
            log.warning("cannot list existing pets", extra={"error": str(exc)})
            taken = []
        for _ in range(8):
            candidate = (x, y, PET_SIZE, PET_SIZE)
            if not any(_rects_overlap(candidate, rect) for rect in taken):
                break
            x -= PET_SIZE + 16
        self.move(max(area.left() + 8, x), y)

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(self._icon(), self)
        self.tray.setToolTip("本地聊天机器人 · 桌宠")
        self.tray.setContextMenu(self._menu())
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

    def _icon(self) -> QIcon:
        pixmap = QPixmap(PET_SIZE, PET_SIZE)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._draw_character(painter, QRectF(0, 0, PET_SIZE, PET_SIZE), mood="idle")
        painter.end()
        return QIcon(pixmap)

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
        for persona in self.personas:
            if persona.get("id") == self.persona_id:
                self.persona_name = str(persona.get("name") or persona.get("id"))
                emotion = str(((persona.get("voice") or {}).get("emotion")) or "neutral")
                self.accent = parse_color(EMOTION_COLORS.get(emotion, "#ff8fb1"))
                break
        self.setWindowTitle(f"{self.persona_name} · 桌宠")
        self._load_skin()
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
        except Exception as exc:  # noqa: BLE001 - a broken setting must not kill the pet
            log.warning("could not resolve pet skin", extra={"error": str(exc)})
        same_path = path == self.skin_path
        consistent = (path is None) == (self.skin_pixmap is None)
        if same_path and consistent and not force:
            return
        self.skin_path = path
        self.skin_pixmap = None
        if not path:
            self.update()
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            log.warning("pet skin could not be loaded", extra={"path": path})
            self.skin_path = None
            self.update()
            return
        self.skin_pixmap = pixmap
        # The tray icon should look like the character the user actually sees.
        if self.tray is not None:
            self.tray.setIcon(self._icon())
        log.info("pet skin applied", extra={"persona": self.persona_id, "path": path})

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

        self.persona_id = persona_id
        self._apply_persona()
        self.bubble.show_text(f"已切换到「{self.persona_name}」", autohide_s=4)

    # -- animation ---------------------------------------------------------------------
    def _tick(self) -> None:
        self._phase += 0.06
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
        rect = QRectF(6, 6, PET_SIZE - 12, PET_SIZE - 26)
        if self.skin_pixmap is not None:
            self._draw_skin(painter, rect)
        else:
            self._draw_character(painter, rect, mood=self.state)
        self._draw_mic(painter)
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
        target = QRectF(rect).translated(0, bob)
        # Keep the aspect ratio: a wide image should not be squashed into a square.
        scaled = pixmap.scaled(
            int(target.width()),
            int(target.height()),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        x = target.center().x() - scaled.width() / 2
        y = target.center().y() - scaled.height() / 2
        if self.state == "speaking":
            pulse = 0.5 + 0.5 * math.sin(self._phase * 3.2)
            glow = QColor(self.accent)
            glow.setAlpha(int(60 + 90 * pulse))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QRectF(x, y, scaled.width(), scaled.height()).adjusted(-6, -6, 6, 6))
        elif self.state in {"listening", "thinking"}:
            glow = QColor(self.accent)
            glow.setAlpha(110 if self.state == "listening" else 70)
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QRectF(x, y, scaled.width(), scaled.height()).adjusted(-5, -5, 5, 5))
        painter.drawPixmap(int(x), int(y), scaled)

        # Name tag, same place as the drawn character's.
        painter.setPen(QPen(QColor(235, 238, 250, 235)))
        font = QFont("Microsoft YaHei UI")
        font.setPointSizeF(9.5)
        painter.setFont(font)
        painter.drawText(QRectF(0, rect.bottom() - 26, rect.width(), 18), Qt.AlignCenter, self.persona_name)

        if self.state == "thinking":
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
        center = QPoint(int(self.width() / 2), int(self.height() - 17))
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
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._press_pos = event.position().toPoint()
            self._press_moved = False
            if self._mic_rect.contains(event.position()):
                self.start_recording()
            else:
                self.bubble.input.hide()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press_pos is not None and event.buttons() & Qt.LeftButton and not self._recording:
            delta = event.position().toPoint() - self._press_pos
            if delta.manhattanLength() > 4:
                self._press_moved = True
            if self._press_moved:
                self.move(self.pos() + delta)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            if self._recording:
                self.stop_recording_and_send()
            elif not self._press_moved:
                self.bubble.ask()
            self._press_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.open_web_ui()

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
        if not text:
            return
        self.bubble.input.hide()
        self.bubble.show_text(text)
        self._set_state("thinking")
        collected: List[str] = []

        def work() -> str:
            final = ""
            for event in self.client.chat_stream(text, persona_id=self.persona_id):
                if event.kind == "text" and event.text:
                    collected.append(event.text)
                elif event.kind == "done":
                    final = event.text or "".join(collected)
            return final or "".join(collected)

        def done(reply: str) -> None:
            self.bubble.show_text(reply or "……")
            self._set_state("idle")
            if reply and not self.muted:
                self.speak(reply)

        self._run_async(work, done, self._on_error)

    def speak(self, text: str) -> None:
        def work() -> int:
            count = 0
            for audio, _meta in self.client.speak_stream(text, persona_id=self.persona_id):
                self.player.enqueue(audio)
                count += 1
            return count

        self._run_async(work, lambda _n: None, lambda exc: log.warning("speech failed", extra={"error": str(exc)}))

    def _on_error(self, exc: Exception) -> None:
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
    def _menu(self) -> QMenu:
        menu = QMenu(self)
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

        # -- 底座模型 --
        model_menu = menu.addMenu("底座模型")
        if not self.model_tiers:
            loading = QAction("（正在读取…）", model_menu)
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
                action.setEnabled(downloaded and not self._switching_model)
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

        persona_menu = menu.addMenu("切换人设")
        for persona in self.personas:
            action = QAction(f"{persona.get('avatar', '')} {persona.get('name', persona.get('id'))}", persona_menu)
            action.setCheckable(True)
            action.setChecked(persona.get("id") == self.persona_id)
            action.triggered.connect(lambda _checked=False, pid=persona.get("id"): self.select_persona(str(pid)))
            persona_menu.addAction(action)
        if not self.personas:
            empty = QAction("（还没连上服务）", persona_menu)
            empty.setEnabled(False)
            persona_menu.addAction(empty)

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

        web = QAction("打开完整界面", menu)
        web.triggered.connect(self.open_web_ui)
        menu.addAction(web)

        hide = QAction("隐藏到托盘", menu)
        hide.triggered.connect(self.hide)
        menu.addAction(hide)

        quit_action = QAction("退出桌宠", menu)
        quit_action.triggered.connect(self.quit)
        menu.addAction(quit_action)
        return menu

    def _show_menu(self, position: QPoint) -> None:
        self._menu().exec(self.mapToGlobal(position))

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
        self.muted = not self.muted
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
            self.player.stop()
            if self._recording:
                self.recorder.stop()
            self.client.close()
        finally:
            if self.tray is not None:
                self.tray.hide()
            QApplication.quit()
