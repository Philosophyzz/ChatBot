"""Desktop pet tests: frame protocol, error paths, thread marshalling, rendering.

The pet is a GUI app, so most of it cannot be asserted directly. These tests target the
parts that actually break in practice:

* the ``[length][tag][payload]`` frame parser (an off-by-one there silently truncates
  speech),
* the "backend is down" path (the pet must degrade to a readable message, not freeze),
* Qt thread affinity — worker results used to be delivered straight from a worker thread,
  which makes Qt print "Timers cannot be started from another thread" and misbehave,
* that the character really paints something.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _frame(tag: bytes, payload: bytes) -> bytes:
    return len(payload).to_bytes(4, "big") + tag + payload


class StubBackend(BaseHTTPRequestHandler):
    """Minimal stand-in for the real API: SSE chat + framed TTS stream."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # noqa: D102 - silence the test output
        return

    def _send(self, body: bytes, content_type: str = "application/json", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/health"):
            self._send(json.dumps({"ok": True, "tts": {"last_used": "stub", "healthy": ["stub"]}}).encode())
        elif self.path.startswith("/api/personas"):
            self._send(
                json.dumps(
                    {
                        "personas": [
                            {"id": "gentle_sister", "name": "温柔姐姐", "voice": {"emotion": "gentle"}},
                            {"id": "tsundere_friend", "name": "阿澈", "voice": {"emotion": "cheerful"}},
                        ],
                        "default": "gentle_sister",
                    }
                ).encode()
            )
        elif self.path.startswith("/api/models"):
            self._send(
                json.dumps(
                    {
                        "port": 8080,
                        "listening": True,
                        "current": "fast-9b",
                        "tiers": [
                            {
                                "id": "fast-9b",
                                "label": "速度优先",
                                "size_gb": 5.29,
                                "downloaded": True,
                                "current": True,
                                "expected_tps": "60~90 tok/s",
                                "ctx": 32768,
                                "n_gpu_layers": -1,
                                "notes": "测试用",
                                "server_args": [],
                                "download_command": "",
                            },
                            {
                                "id": "quality-35b",
                                "label": "最强质量",
                                "size_gb": 10.02,
                                "downloaded": False,
                                "current": False,
                                "expected_tps": "5~15 tok/s",
                                "ctx": 8192,
                                "n_gpu_layers": -1,
                                "notes": "测试用",
                                "server_args": ["--n-cpu-moe", "8"],
                                "download_command": "powershell -File scripts\\download-models.ps1 -Tier quality-35b -Mirror",
                            },
                        ],
                        "gpu": {"total_mib": 16303, "free_mib": 2048},
                        "busy": False,
                        "progress": {"state": "idle"},
                    }
                ).encode()
            )
        else:
            self._send(b"{}", status=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.path.startswith("/api/chat"):
            body = (
                'event: text\ndata: {"kind": "text", "text": "你好"}\n\n'
                'event: text\ndata: {"kind": "text", "text": "呀"}\n\n'
                'event: done\ndata: {"kind": "done", "text": "你好呀"}\n\n'
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/voice/tts/stream"):
            pcm = b"\x00\x00" * 2000
            from speech.audio import pcm_to_wav

            audio = pcm_to_wav(pcm)
            body = (
                _frame(b"M", json.dumps({"chunks": 2, "voice_id": "stub"}).encode())
                + _frame(b"A", audio)
                + _frame(b"A", audio)
                + _frame(b"M", json.dumps({"backend": "stub", "frames": 2, "elapsed_ms": 12.5}).encode())
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/voice/stt"):
            # Mirrors the real API exactly: the text is nested under "transcript".
            # A client reading a top-level "text" gets None and then reports
            # "没听到声音" — which looks like a dead microphone, not a parsing bug.
            self._send(
                json.dumps(
                    {
                        "transcript": {
                            "text": "这是识别结果",
                            "language": "zh",
                            "confidence": 1.0,
                            "duration_s": 1.5,
                            "segments": [],
                        },
                        "elapsed_ms": 12.3,
                    }
                ).encode()
            )
        else:
            self._send(b"{}", status=404)


@pytest.fixture
def stub_backend():
    # daemon_threads: ThreadingHTTPServer's handler threads are *non-daemon* by default, and
    # httpx keeps its connection alive. A handler blocked in readline() on such a connection
    # then pins the interpreter at exit — pytest finishes all tests and the process simply
    # never returns (this cost two 10-minute "hangs" that looked like a stuck test).
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubBackend)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_chat_stream_collects_text_and_done(stub_backend) -> None:
    from pet.client import BackendClient

    client = BackendClient(stub_backend)
    events = list(client.chat_stream("你好", persona_id="gentle_sister"))
    kinds = [event.kind for event in events]
    assert kinds[-1] == "done"
    assert "".join(event.text for event in events if event.kind == "text") == "你好呀"
    assert events[-1].text == "你好呀"
    client.close()


def test_speak_stream_parses_frames_and_meta(stub_backend) -> None:
    """Two audio frames must arrive intact — a bad length prefix truncates speech."""
    from pet.client import BackendClient

    client = BackendClient(stub_backend)
    chunks = list(client.speak_stream("你好呀", persona_id="gentle_sister"))
    assert len(chunks) == 2, f"应收到 2 个音频帧，实际 {len(chunks)}"
    for audio, meta in chunks:
        assert audio[:4] == b"RIFF" and len(audio) > 1000
        assert meta.get("chunks") == 2
    # The summary frame arrives after the last audio frame, so it is exposed separately
    # rather than folded into a chunk that does not exist.
    assert client.last_tts_meta.get("backend") == "stub"
    assert client.last_tts_meta.get("frames") == 2
    assert len(client.speak("你好呀")) > 2000
    client.close()


def test_transcribe_reads_the_nested_transcript(stub_backend) -> None:
    """Regression: the STT response nests the text under ``transcript``.

    The pet read a top-level ``text``, got ``None``, and told the user "没听到声音" —
    a parsing bug indistinguishable from a microphone problem. This test pins the real
    response shape.
    """
    from pet.client import BackendClient

    client = BackendClient(stub_backend)
    text = client.transcribe(b"RIFF" + b"\x00" * 200)
    assert text == "这是识别结果", f"应解析 transcript.text，实际 {text!r}"
    client.close()


def test_unreachable_backend_raises_a_readable_error() -> None:
    """A dead service must produce a sentence the pet can show, not a traceback."""
    from pet.client import BackendClient, BackendError

    client = BackendClient("http://127.0.0.1:1")  # nothing listens here
    assert client.online() is False
    with pytest.raises(BackendError) as excinfo:
        list(client.chat_stream("你好"))
    assert "连不上" in str(excinfo.value) or "对话请求失败" in str(excinfo.value)
    client.close()


def test_worker_results_are_delivered_on_the_gui_thread(stub_backend) -> None:
    """Regression: callbacks used to run on the worker thread and touch Qt directly.

    Qt widgets are not thread-safe: starting a QTimer or resizing a widget from another
    thread produces ``QObject::startTimer: Timers cannot be started from another thread``
    and the UI then misbehaves in ways that are very hard to trace back to the cause.
    """
    from PySide6.QtWidgets import QApplication

    from pet.client import BackendClient
    from pet.window import PetWindow

    app = QApplication.instance() or QApplication([])
    window = PetWindow(BackendClient(stub_backend))
    main_thread = threading.get_ident()
    seen: List[Tuple[str, int]] = []

    window._run_async(lambda: 41, lambda value: seen.append(("ok", threading.get_ident())), lambda exc: seen.append(("err", 0)))

    deadline = time.time() + 5
    while not seen and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert seen, "worker 回调没有执行"
    kind, thread_id = seen[0]
    assert kind == "ok"
    assert thread_id == main_thread, "回调必须在 GUI 线程执行，否则 Qt 会出诡异问题"

    window.tray.hide()
    window.client.close()


def test_pet_paints_character_and_microphone(stub_backend) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    from pet.client import BackendClient
    from pet.window import PET_SIZE, PetWindow

    app = QApplication.instance() or QApplication([])
    window = PetWindow(BackendClient(stub_backend))
    window.personas = [{"id": "gentle_sister", "name": "温柔姐姐", "voice": {"emotion": "gentle"}}]
    window.persona_id = "gentle_sister"
    window._apply_persona()

    for mood in ("idle", "listening", "thinking", "speaking", "offline"):
        window.state = mood
        pixmap = QPixmap(PET_SIZE, PET_SIZE)
        pixmap.fill(Qt.transparent)
        window.render(pixmap)
        image = pixmap.toImage()
        painted = sum(
            1 for y in range(0, PET_SIZE, 4) for x in range(0, PET_SIZE, 4) if image.pixelColor(x, y).alpha() > 8
        )
        assert painted > 150, f"{mood} 状态几乎没画出东西（{painted} 点）"
        mic = image.pixelColor(PET_SIZE // 2, PET_SIZE - 17)
        assert mic.alpha() > 200, f"{mood} 状态的麦克风按钮不可见"

    window.tray.hide()
    window.client.close()


def test_bubble_always_lands_above_the_pet(stub_backend) -> None:
    """Regression: the bubble showed up in the middle of the screen.

    The bubble is a frameless top-level window, so until something calls ``move`` Windows
    places it wherever the default position is — the centre of the screen. Only one of the
    ~15 ``show_text`` call sites repositioned it, so the greeting ("…按住麦克风跟我说话吧")
    appeared mid-screen instead of at the pet.
    """
    from PySide6.QtWidgets import QApplication

    from pet.client import BackendClient
    from pet.window import PetWindow

    app = QApplication.instance() or QApplication([])
    window = PetWindow(BackendClient(stub_backend))
    window.show()
    app.processEvents()

    window.bubble.show_text("在呢，按住麦克风跟我说话吧。")
    app.processEvents()
    assert window.bubble.isVisible()
    first = window.bubble.geometry()
    assert first.y() + first.height() <= window.y() + 8, (
        f"气泡应该在桌宠上方：气泡底 {first.y() + first.height()} vs 桌宠顶 {window.y()}"
    )
    # The tail points down at the pet, so the bubble must cover the pet's centre column.
    # (Not "centred within 3px": near a screen edge the bubble is clamped back on screen.)
    pet_center = window.x() + window.width() // 2
    assert first.x() <= pet_center <= first.x() + first.width(), (
        f"气泡没有对准桌宠：气泡 {first.x()}..{first.x() + first.width()}，桌宠中心 {pet_center}"
    )

    # Text grows the bubble; typing and long replies must not walk it off the pet.
    window.bubble.show_text("很长的回复。" * 40)
    app.processEvents()
    grown = window.bubble.geometry()
    assert grown.height() > first.height(), "换成长文本后气泡应长高"
    assert grown.y() + grown.height() <= window.y() + 8, "长文本也不许盖住桌宠"
    assert grown.x() <= pet_center <= grown.x() + grown.width(), "长文本后气泡也要对准桌宠"

    screen = QApplication.primaryScreen()
    if screen is not None:
        area = screen.availableGeometry()
        assert area.contains(grown.topLeft()) and area.contains(grown.bottomRight()), (
            f"气泡跑出屏幕了：{grown} 不在 {area} 内"
        )

    # Dragging the pet takes the bubble along.
    window.move(window.x() - 120, window.y() - 60)
    app.processEvents()
    moved = window.bubble.geometry()
    assert moved.x() != grown.x() or moved.y() != grown.y(), "桌宠移动后气泡应跟随"
    assert moved.y() + moved.height() <= window.y() + 8

    window.bubble.hide()
    window.tray.hide()
    window.client.close()


def test_one_pet_per_persona_guard() -> None:
    """The guard must refuse a second identical pet and still allow other personas.

    Two pets of the same persona would listen on the same microphone and answer the same
    sentence twice. The guard is a named mutex, so it also has to survive being released
    and re-claimed (that is what switching persona does).
    """
    from pet import single_instance

    single_instance.release_all()
    try:
        assert single_instance.acquire("__test_persona_a__") is True, "第一次应该拿到锁"
        assert single_instance.acquire("__test_persona_a__") is False, "同一人设不允许开第二个"
        assert single_instance.acquire("__test_persona_b__") is True, "不同人设可以同时开"
        assert sorted(single_instance.held_keys()) == ["__test_persona_a__", "__test_persona_b__"]

        single_instance.release("__test_persona_a__")
        assert single_instance.acquire("__test_persona_a__") is True, "释放后应该能重新拿到锁"
        assert single_instance.acquire("__test_persona_b__") is False, "b 仍然被自己占着"
    finally:
        single_instance.release_all()
    assert single_instance.held_keys() == []


def test_switching_persona_hands_over_the_guard(stub_backend) -> None:
    """Switching persona inside one pet must claim the new name and free the old one.

    Synthetic persona ids on purpose: the guard is a **session-wide** named mutex, so a real
    pet running on this desktop (holding ``ChatBotPet_gentle_sister``) would make the test
    fail for the wrong reason.
    """
    from PySide6.QtWidgets import QApplication

    from pet import single_instance
    from pet.client import BackendClient
    from pet.window import PetWindow

    app = QApplication.instance() or QApplication([])
    single_instance.release_all()
    try:
        assert single_instance.acquire("__test_persona_a__") is True
        window = PetWindow(BackendClient(stub_backend), persona_id="__test_persona_a__")
        window.personas = [
            {"id": "__test_persona_a__", "name": "温柔姐姐", "voice": {"emotion": "gentle"}},
            {"id": "__test_persona_b__", "name": "阿澈", "voice": {"emotion": "cheerful"}},
        ]
        window.select_persona("__test_persona_b__")
        app.processEvents()
        assert window.persona_id == "__test_persona_b__"
        assert single_instance.held_keys() == ["__test_persona_b__"], "旧人设的锁应该交还，新的人设应该占住"

        # Let the startup greeting land before asserting on bubble text: it writes to the
        # same bubble and would otherwise race with the refusal message below.
        deadline = time.time() + 5
        while window.state != "idle" and time.time() < deadline:
            app.processEvents()
            time.sleep(0.02)

        # Someone else already runs 温柔姐姐 → the switch must be refused, not duplicated.
        assert single_instance.acquire("__test_persona_a__") is True  # simulate the other pet
        window.select_persona("__test_persona_a__")
        assert window.persona_id == "__test_persona_b__", "别人开着的人设不该被抢过来"
        assert "已经开着" in window.bubble.label.text(), window.bubble.label.text()

        window.bubble.hide()
        window.tray.hide()
        window.client.close()
    finally:
        single_instance.release_all()


def test_second_pet_takes_an_empty_slot(stub_backend, monkeypatch) -> None:
    """Different personas may run side by side — but not on the same pixel.

    Found by running two pets for real: 温柔姐姐 and 阿澈 both parked at the same
    bottom-right coordinate, so the second one was completely hidden behind the first.
    """
    from PySide6.QtWidgets import QApplication

    from pet import single_instance
    from pet.client import BackendClient
    from pet.window import PET_SIZE, PetWindow, _rects_overlap

    app = QApplication.instance() or QApplication([])
    screen = QApplication.primaryScreen()
    assert screen is not None
    area = screen.availableGeometry()
    corner = (area.right() - PET_SIZE - 40, area.bottom() - PET_SIZE - 60)

    monkeypatch.setattr(single_instance, "pet_window_rects", lambda: [])
    first = PetWindow(BackendClient(stub_backend), with_tray=False)
    assert (first.x(), first.y()) == corner, f"第一个桌宠应该站在右下角，实际 {(first.x(), first.y())}"

    # Now pretend the first pet is on screen (the offscreen platform has no real windows).
    monkeypatch.setattr(single_instance, "pet_window_rects", lambda: [(*corner, PET_SIZE, PET_SIZE)])
    second = PetWindow(BackendClient(stub_backend), with_tray=False)
    assert second.x() < first.x(), "第二个桌宠应该往左让位"
    assert not _rects_overlap((second.x(), second.y(), PET_SIZE, PET_SIZE), (*corner, PET_SIZE, PET_SIZE))
    assert second.x() + PET_SIZE <= corner[0], "两个桌宠不许重叠"

    assert _rects_overlap((0, 0, 10, 10), (5, 5, 10, 10)) is True
    assert _rects_overlap((0, 0, 10, 10), (20, 0, 10, 10)) is False, "分开放的两个位置不算重叠"
    # The default gap keeps two pets from touching edge to edge.
    assert _rects_overlap((0, 0, 10, 10), (12, 0, 10, 10)) is True

    for window in (first, second):
        window.bubble.hide()
        window.client.close()


def test_pet_stays_a_thin_client() -> None:
    """Importing the pet must not drag in the inference stack (torch / whisper / cv2).

    Two reasons this is a test and not a comment:
    * startup time — a pet that imports torch takes tens of seconds to appear;
    * the frozen exe — PyInstaller bundles whatever the import graph reaches, and it
      reached the whole speech stack through ``plugs``: the build produced a 3.3GB exe.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, r'%s'); "
        "import pet.window, pet.client, pet.audio, pet.single_instance; "
        "heavy = [m for m in ('torch','faster_whisper','transformers','cv2','onnxruntime',"
        "'ctranslate2','librosa','accelerate','plugs') if m in sys.modules]; "
        "print(','.join(heavy))" % (root / "src")
    )
    result = subprocess.run(  # noqa: S603 - fixed local interpreter, no user input
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-800:]
    heavy = [name for name in result.stdout.strip().split(",") if name]
    assert not heavy, f"桌宠启动时不该导入这些重家伙：{heavy}"


def test_existing_pet_rects_are_converted_to_qt_coordinates() -> None:
    """Regression: Win32 rects are physical pixels, Qt geometry is logical.

    On this 175%-scaled display the pet sits at Qt ``(1943, 915)`` while ``GetWindowRect``
    reports the very same window at ``(3400, 1601)``. Comparing the two directly means the
    "is this slot free?" check never sees an overlap — which is exactly how two pets ended
    up stacked on the same pixel even though the unit test (using one coordinate system for
    both sides) passed.
    """
    from pet.window import _rects_overlap, _to_logical

    converted = _to_logical([(3400, 1601, 368, 368)], 1.75)[0]
    assert converted == pytest.approx((1942.857, 914.857, 210.286, 210.286), abs=0.01)
    default_corner = (1943, 915, 210, 210)
    assert _rects_overlap(default_corner, converted) is True, "换算后右下角必须判定为被占用"
    next_slot = (1943 - 210 - 16, 915, 210, 210)
    assert _rects_overlap(next_slot, converted) is False, "左移一格之后应该空出来"
    assert _to_logical([(10, 20, 30, 40)], 1.0) == [(10, 20, 30, 40)], "100% 缩放不该改动数值"
    assert _to_logical([(10, 20, 30, 40)], 0) == [(10, 20, 30, 40)], "拿不到缩放比例时按原样处理"


def test_focusing_an_existing_pet_does_not_move_it() -> None:
    """Regression: focusing used to send SW_RESTORE unconditionally.

    Windows answers SW_RESTORE on a *normal* (never minimised) window by moving it to its
    stored restore position — the pet jumped from (1943, 915) to (1921, 774) every time the
    user double-clicked the shortcut while it was already running.
    """
    from pet.single_instance import SW_RESTORE, SW_SHOW, focus_action

    assert focus_action(iconic=False, visible=True) is None, "正常显示的窗口不该被 ShowWindow 动位置"
    assert focus_action(iconic=True, visible=True) == SW_RESTORE, "最小化了才还原"
    assert focus_action(iconic=False, visible=False) == SW_SHOW, "藏在托盘里的要叫回来"
    assert focus_action(iconic=True, visible=False) == SW_RESTORE


def test_custom_skin_replaces_the_drawn_character(tmp_path, stub_backend) -> None:
    """The pet must be able to wear a user-supplied image instead of the drawn character."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    from pet.client import BackendClient
    from pet.settings import PetSettings
    from pet.skin import install_skin, placeholder_image
    from pet.window import PET_SIZE, PetWindow

    app = QApplication.instance() or QApplication([])
    settings = PetSettings(tmp_path / "pet_settings.json")

    source = tmp_path / "girl.png"
    assert placeholder_image(256).save(str(source), "PNG")
    # directory=tmp_path: never write into the real data/pet_skins from a test.
    stored = install_skin(source, "gentle_sister", size=512, settings=settings, directory=tmp_path / "skins")
    assert stored.exists() and stored.suffix == ".png"
    assert stored.parent.name == "skins"
    assert settings.skin_for("gentle_sister"), "设置里应该记下这个人设的形象"
    assert settings.skin_for("other_persona") is None, "没设置的人设不该共享"

    window = PetWindow(BackendClient(stub_backend), persona_id="gentle_sister", with_tray=False)
    window.settings = settings
    window.skin_path = None
    window._load_skin()
    assert window.skin_path and window.skin_pixmap is not None, "应该加载到自定义形象"

    def render() -> "object":
        pixmap = QPixmap(PET_SIZE, PET_SIZE)
        pixmap.fill(Qt.transparent)
        window.render(pixmap)
        return pixmap.toImage()

    with_skin = render()
    window.skin_pixmap = None  # 回到内置角色
    window.skin_path = None
    drawn = render()
    different = sum(
        1
        for y in range(0, PET_SIZE, 6)
        for x in range(0, PET_SIZE, 6)
        if with_skin.pixelColor(x, y) != drawn.pixelColor(x, y)
    )
    assert different > 50, f"换成自定义形象后画面应该明显不同（{different} 个采样点相同）"

    window.bubble.hide()
    window.client.close()


def test_skin_rejects_junk_with_a_readable_message(tmp_path) -> None:
    from pet.settings import PetSettings
    from pet.skin import SkinError, install_skin

    settings = PetSettings(tmp_path / "pet_settings.json")
    with pytest.raises(SkinError):
        install_skin(tmp_path / "nope.png", "x", settings=settings)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not an image")
    with pytest.raises(SkinError):
        install_skin(broken, "x", settings=settings)
    txt = tmp_path / "notes.txt"
    txt.write_text("hello", encoding="utf-8")
    with pytest.raises(SkinError):
        install_skin(txt, "x", settings=settings)


def test_pet_settings_persist_and_survive_a_corrupt_file(tmp_path) -> None:
    from pet.settings import PetSettings

    path = tmp_path / "pet_settings.json"
    settings = PetSettings(path)
    assert settings.get("hands_free_sensitivity") == "normal", "默认值应该可用"
    settings.set("hands_free_sensitivity", "noisy")
    assert PetSettings(path).get("hands_free_sensitivity") == "noisy", "应该写盘并重新读出来"

    path.write_text("{ this is not json", encoding="utf-8")
    recovered = PetSettings(path)
    assert recovered.get("hands_free_sensitivity") == "normal", "坏文件要回落到默认值而不是崩掉"


def test_pet_menu_offers_model_switch_and_skin(tmp_path, stub_backend) -> None:
    """The tray menu is the "操作入口" for both new features — it must actually list them."""
    from PySide6.QtWidgets import QApplication

    from pet.client import BackendClient
    from pet.settings import PetSettings
    from pet.window import PetWindow

    app = QApplication.instance() or QApplication([])
    window = PetWindow(BackendClient(stub_backend), persona_id="gentle_sister", with_tray=False)
    window.settings = PetSettings(tmp_path / "pet_settings.json")

    # Let the async /api/models fetch land (the menu is rebuilt from it).
    window.refresh_models()
    deadline = time.time() + 5
    while not window.model_tiers and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert window.model_tiers, "应该从 /api/models 读到档位列表"
    assert window.model_current == "fast-9b"

    menu = window._menu()
    labels = [action.text() for action in menu.actions()]
    for expected in ("一直听着（自动断句）", "免提灵敏度", "更换形象", "底座模型"):
        assert any(expected in label for label in labels), f"托盘菜单里缺少「{expected}」：{labels}"

    model_menu = next(action.menu() for action in menu.actions() if action.text() == "底座模型")
    tier_labels = [action.text() for action in model_menu.actions()]
    assert any("fast-9b" in label for label in tier_labels)
    assert any("未下载" in label for label in tier_labels), "未下载的档位要标出来"
    enabled = {action.text(): action.isEnabled() for action in model_menu.actions()}
    assert any("fast-9b" in text and ok for text, ok in enabled.items())
    assert any("quality-35b" in text and not ok for text, ok in enabled.items()), "没下载的档位不该能点"

    skin_menu = next(action.menu() for action in menu.actions() if action.text() == "更换形象")
    assert any("选一张图片" in action.text() for action in skin_menu.actions())

    window.bubble.hide()
    window.client.close()


def test_hands_free_sends_without_touching_the_button(stub_backend, monkeypatch) -> None:
    """The reported bug, end to end inside the window: talk → it sends by itself.

    Two defects made 「一直听着」 useless: the level never fell (so silence was never
    detected) and the listener never re-armed after a turn. This test drives the window's own
    tick with a scripted level sequence and asserts that a clip really goes to the recogniser
    without any button press — then that it is listening again for the next sentence.
    """
    from PySide6.QtWidgets import QApplication

    from pet.client import BackendClient
    from pet.window import PetWindow

    class FakeRecorder:
        """Records nothing; returns silence levels unless told otherwise."""

        def __init__(self) -> None:
            self.recording = False
            self.peak = 0.0
            self.level_value = 0.0
            self.started = 0
            self.retained = []

        def start(self) -> None:
            self.recording = True
            self.started += 1

        def stop(self) -> bytes:
            self.recording = False
            return b"RIFF" + b"\x00" * 4000  # long enough to pass the "heard nothing" check

        def level(self) -> float:
            return self.level_value

        def retain_last(self, seconds: float) -> None:
            self.retained.append(seconds)

    app = QApplication.instance() or QApplication([])
    window = PetWindow(BackendClient(stub_backend), persona_id="gentle_sister", with_tray=False)
    recorder = FakeRecorder()
    window.recorder = recorder

    transcripts = []
    monkeypatch.setattr(window.client, "transcribe", lambda wav: transcripts.append(len(wav)) or "这是识别结果")
    monkeypatch.setattr(window.client, "chat_stream", lambda text, persona_id=None: iter(()))

    # The window's tick polls the real clock; drive a fake one so the test is instant and
    # deterministic instead of sleeping 0.25 s per tick.
    clock = {"now": 1000.0}
    monkeypatch.setattr("pet.hands_free.time.monotonic", lambda: clock["now"])

    def tick(level: float, times: int) -> None:
        recorder.level_value = level
        for _ in range(times):
            clock["now"] += 0.25
            window._hands_free_tick()

    window.toggle_hands_free()
    assert window.hands_free, "应该进入免提模式"
    # A pet that is still thinking (e.g. right after startup) defers opening the mic until it
    # is idle again — otherwise it would record its own reply. Simulate the ready state.
    window._set_state("idle")
    app.processEvents()
    window._resume_hands_free()
    assert recorder.started == 1, "免提一开（桌宠空闲时）就该打开麦克风"

    # 安静 2 秒：不该发任何东西（旧代码在这里就会因为电平不下降而卡死）
    tick(0.0, 8)
    assert transcripts == [], "没说话的时候不该发送"

    # 说 1.5 秒，然后安静 1.5 秒 → 自动断句并发送
    tick(0.25, 6)
    tick(0.0, 8)
    assert transcripts, "说完停下来之后应该自动发出去（不需要按按钮）"

    # 回合结束后要能继续听下一句
    window._set_state("idle")
    window._resume_hands_free()
    assert recorder.started >= 2, "一轮对话之后必须重新开始收音"
    assert window.listener.phase in {"armed", "speech"}

    # 关掉之后不再收音
    window.toggle_hands_free()
    assert not window.hands_free
    assert window.listener.phase == "off"

    window.bubble.hide()
    window.client.close()


def test_switching_back_to_the_default_look_removes_the_image(tmp_path, stub_backend) -> None:
    """Regression: 「换回默认形象」 kept drawing the old picture (user-reported).

    ``_load_skin`` compared only the paths, so after the setting was cleared the comparison
    was ``None == None`` and it returned *before* dropping the loaded pixmap. The assertion is
    on rendered pixels, not on the flag, because the picture on screen is what the user sees.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    from pet.client import BackendClient
    from pet.settings import PetSettings
    from pet.skin import install_skin, placeholder_image
    from pet.window import PET_SIZE, PetWindow

    app = QApplication.instance() or QApplication([])
    settings = PetSettings(tmp_path / "pet_settings.json")
    source = tmp_path / "girl.png"
    assert placeholder_image(256).save(str(source), "PNG")
    install_skin(source, "gentle_sister", settings=settings, directory=tmp_path / "skins")

    window = PetWindow(BackendClient(stub_backend), persona_id="gentle_sister", with_tray=False)
    window.settings = settings
    window._load_skin(force=True)
    assert window.skin_pixmap is not None and window.skin_path

    def render():
        pixmap = QPixmap(PET_SIZE, PET_SIZE)
        pixmap.fill(Qt.transparent)
        window.render(pixmap)
        return pixmap.toImage()

    with_skin = render()
    window.clear_skin()
    app.processEvents()
    after_clear = render()

    assert window.skin_path is None, "设置里的形象应该被清掉"
    assert window.skin_pixmap is None, "缓存的图片也必须清掉（这就是那个 bug）"
    assert settings.skin_for("gentle_sister") is None

    differing = sum(
        1
        for y in range(0, PET_SIZE, 6)
        for x in range(0, PET_SIZE, 6)
        if with_skin.pixelColor(x, y) != after_clear.pixelColor(x, y)
    )
    assert differing > 50, f"画面应该换回绘制角色，实际只差 {differing} 个采样点"

    # 再装一次也应该立刻生效（同一条路径反过来走一遍）
    window.set_skin(str(source))
    app.processEvents()
    assert window.skin_pixmap is not None and window.skin_path
    assert render() != after_clear

    window.bubble.hide()
    window.client.close()


def test_hands_free_tick_never_raises_even_if_the_listener_does(stub_backend) -> None:
    """A raising Qt timer slot is fatal: PySide escalates it and the whole pet disappears.

    That is how a state-machine assertion turned into "切换灵敏度之后语音识别失败". The window
    handler must swallow and recover instead.
    """
    from PySide6.QtWidgets import QApplication

    from pet.client import BackendClient
    from pet.window import PetWindow

    app = QApplication.instance() or QApplication([])
    window = PetWindow(BackendClient(stub_backend), persona_id="gentle_sister", with_tray=False)

    class ExplodingListener:
        paused = False

        def __init__(self) -> None:
            self.armed = 0

        def tick(self, level, now=None):  # noqa: ANN001
            raise RuntimeError("boom")

        def arm(self, now=None):  # noqa: ANN001
            self.armed += 1

    listener = ExplodingListener()
    window.listener = listener
    window.hands_free = True
    window._recording = True

    window._hands_free_tick()  # 不许抛出去
    assert listener.armed == 1, "出错后应该重新开始收音，而不是卡死或崩溃"

    window.bubble.hide()
    window.client.close()


def test_start_backend_is_offered_only_when_the_service_is_down(stub_backend) -> None:
    """User-reported: 「启动本地服务」was "本身就是错误的，而且很多余".

    Two things were wrong with it: it was always listed, and it re-ran start-all.ps1 even when
    the service was healthy — which restarts processes and drops the web UI's connection (the
    browser then shows "Failed to fetch"). Now it appears only while the pet is offline, which
    is exactly when its persona refresh fails, so the test drives both states for real instead
    of poking the state field (which the async refresh would overwrite).
    """
    from PySide6.QtWidgets import QApplication

    from pet.client import BackendClient
    from pet.window import PetWindow

    def wait_for(window, predicate, timeout: float = 6.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            app.processEvents()
            if predicate():
                return True
            time.sleep(0.02)
        return False

    app = QApplication.instance() or QApplication([])

    healthy = PetWindow(BackendClient(stub_backend), persona_id="gentle_sister", with_tray=False)
    assert wait_for(healthy, lambda: healthy.state == "idle"), "桩后端在线时桌宠应进入 idle"
    labels = [action.text() for action in healthy._menu().actions()]
    assert not any("启动它" in label for label in labels), f"服务健康时不该出现启动项：{labels}"
    assert any("重新连接" in label for label in labels), "重新连接要一直在"
    healthy.bubble.hide()
    healthy.client.close()

    # 真的连不上：一个新窗口指向没人监听的端口，它的刷新必然失败 → offline
    offline = PetWindow(BackendClient("http://127.0.0.1:1"), persona_id="gentle_sister", with_tray=False)
    assert wait_for(offline, lambda: offline.state == "offline"), "连不上时应该进入 offline"
    down_labels = [action.text() for action in offline._menu().actions()]
    assert any("启动它" in label for label in down_labels), f"连不上时才该出现：{down_labels}"
    assert any("启动聊天机器人" in label for label in down_labels), "顺便告诉用户正式入口"
    offline.bubble.hide()
    offline.client.close()


def test_start_backend_does_not_restart_a_healthy_service(monkeypatch, stub_backend) -> None:
    """Clicking it must be a no-op when the service answers /api/health."""
    import subprocess

    from PySide6.QtWidgets import QApplication

    from pet.client import BackendClient
    from pet.window import PetWindow

    app = QApplication.instance() or QApplication([])
    window = PetWindow(BackendClient(stub_backend), persona_id="gentle_sister", with_tray=False)
    window._set_state("idle")

    spawned = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: spawned.append(args))

    window.start_backend()
    assert spawned == [], "服务健康时不该启动任何进程（那会掐断网页连接）"
    assert "已经在运行" in window.bubble.label.text(), window.bubble.label.text()

    window.bubble.hide()
    window.client.close()


def test_recorder_and_player_handle_edge_cases() -> None:
    """Empty recordings and undecodable chunks must not raise into the UI."""
    from pet.audio import Player, Recorder

    recorder = Recorder()
    assert recorder.stop() == b"", "没开始录音就停止应该是空字节，而不是抛异常"

    player = Player()
    player.enqueue(b"")  # no-op
    assert not player._queue

    from speech.audio import pcm_to_wav

    samples, rate = player._decode(pcm_to_wav(b"\x00\x00" * 800))
    assert rate == 16000 and len(samples) == 800
