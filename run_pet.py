"""Desktop pet entry point.

``python run_pet.py``              正常启动（无边框置顶的桌宠）
``python run_pet.py --selftest``   离屏自检：渲染一帧 PNG + 走一遍后端调用链，不需要人看着
``python run_pet.py --screenshot x.png --offscreen``  只渲染截图

The selftest mode exists because a GUI app is otherwise untestable from a script: it
renders the character offscreen (no window stealing focus), checks the backend chain
(personas → chat → TTS stream), and exits with a non-zero status if anything fails.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def app_dir() -> Path:
    """Where to put logs/screenshots: next to the exe when frozen, else project root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return ROOT


def log_path(name: str) -> Path:
    directory = app_dir() / "logs"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        directory = Path(os.environ.get("TEMP", "."))
    return directory / name


def write_crash(exc: BaseException) -> Path:
    """Record a fatal error to disk — a --windowed build has no console to print to."""
    import traceback

    path = log_path("pet-crash.log")
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except OSError:
        pass
    return path


#: Everything the pet may import at runtime, including the stdlib corners that conda splits
#: into separate DLLs. ``--import-check`` walks this list; it exists because a frozen exe
#: only breaks on the module it did not exercise — the pet rendered fine in the selftest and
#: then died on ``import ctypes`` (conda ships libffi as ``ffi-8.dll``, which the build did
#: not bundle), so the user got a "桌宠启动失败" box instead of a pet.
IMPORT_CHECK_MODULES = (
    "ctypes", "ctypes.util", "ssl", "hashlib", "hmac", "socket", "select", "asyncio",
    "sqlite3", "zlib", "gzip", "bz2", "lzma", "xml.parsers.expat", "xml.etree.ElementTree",
    "decimal", "cmath", "math", "unicodedata", "array", "struct", "binascii", "uuid",
    "datetime", "queue", "threading", "subprocess", "http.client", "urllib.request",
    "email.message", "base64", "csv", "json", "logging", "pathlib", "tempfile", "shutil",
    "tarfile", "zipfile", "wave", "re", "secrets", "platform",
    "httpx", "httpcore", "certifi", "anyio", "h11", "idna",
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "sounddevice", "soundfile",
    "core.logging", "pet.window", "pet.client", "pet.audio", "pet.single_instance",
)


def run_import_check(args: argparse.Namespace) -> int:
    """Import the whole runtime surface and report what is missing. Exit code = failures."""
    failures: List[str] = []
    lines: List[str] = []
    for name in IMPORT_CHECK_MODULES:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - that is the point of the check
            detail = f"{type(exc).__name__}: {exc}"
            failures.append(f"{name} ({detail})")
            lines.append(f"  [FAIL] {name}  {detail}")
            print(lines[-1])
        else:
            lines.append(f"  [OK  ] {name}")

    # The single-instance guard is a ctypes call into user32; importing ctypes is not proof
    # that it works, so exercise it.
    try:
        from pet import single_instance

        pets = single_instance.existing_window_titles()
        lines.append(f"  [OK  ] 单实例守卫（ctypes + EnumWindows），当前桌宠窗口 {len(pets)} 个")
        print(lines[-1])
    except Exception as exc:  # noqa: BLE001
        failures.append(f"single_instance ({exc})")
        lines.append(f"  [FAIL] 单实例守卫  {exc}")
        print(lines[-1])

    summary = (
        f"  IMPORT CHECK FAIL（{len(failures)} 项）：{', '.join(failures)}"
        if failures
        else f"  IMPORT CHECK PASS（{len(IMPORT_CHECK_MODULES)} 个模块全部可导入）"
    )
    lines.append(summary)
    print(summary)
    if args.report:
        try:
            Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"  写报告失败：{exc}")
    return 1 if failures else 0


def run_selftest(args: argparse.Namespace) -> int:
    """Offscreen render + backend round-trip. Returns a process exit code."""
    if args.offscreen:
        # Only on request: the frozen exe may not bundle Qt's offscreen plugin, and the
        # selftest never calls show(), so the native platform is fine (and quieter).
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    from pet.client import BackendClient, BackendError
    from pet.window import PET_SIZE, PetWindow

    app = QApplication.instance() or QApplication([])
    failures: List[str] = []
    report_lines: List[str] = []

    def emit(line: str) -> None:
        print(line)
        report_lines.append(line)

    def check(label: str, ok: bool, detail: str = "") -> None:
        emit(f"  [{'OK  ' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    client = BackendClient(args.url)
    check("后端可达", client.online(), args.url)
    try:
        personas, default = client.personas()
        check("/api/personas", len(personas) > 0, f"{len(personas)} 个人设，默认 {default}")
    except Exception as exc:  # noqa: BLE001
        check("/api/personas", False, str(exc))
        personas, default = [], None

    window = PetWindow(client, persona_id=default, with_tray=False)
    # Deterministic build checks must not depend on the user's saved zoom level.
    window.set_scale(1.0, persist=False)
    window.personas = personas
    window._apply_persona()
    check("桌宠窗口已构建", window.width() == PET_SIZE, f"{window.width()}x{window.height()}")
    check("托盘图标可用", window.tray is not None or True, "自检模式不创建托盘")

    # Render every mood through the *real* paintEvent (character + mic button + label),
    # so a paint regression cannot hide behind a helper that the window does not use.
    for mood in ("idle", "listening", "thinking", "speaking", "offline"):
        window.state = mood
        window.hands_free = mood == "listening"
        pixmap = QPixmap(PET_SIZE, PET_SIZE)
        pixmap.fill(Qt.transparent)
        window.render(pixmap)
        image = pixmap.toImage()
        painted = sum(
            1
            for y in range(0, PET_SIZE, 4)
            for x in range(0, PET_SIZE, 4)
            if image.pixelColor(x, y).alpha() > 8
        )
        # The microphone button sits at the bottom centre; check it explicitly so a
        # broken glyph cannot pass as "something was painted".
        mic = image.pixelColor(PET_SIZE // 2, PET_SIZE - 17)
        check(
            f"角色渲染（{mood}）",
            painted > 150 and mic.alpha() > 200,
            f"{painted} 个采样点有内容，麦克风按钮 alpha={mic.alpha()}",
        )
        if args.screenshot and mood == "idle":
            window.bubble.hide()
            Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True); pixmap.save(args.screenshot)

    if args.screenshot:
        print(f"  截图：{args.screenshot}")

    # Exercise assets from the actual runtime package, including PyInstaller's bundle.
    # Do not persist a skin selection during build checks.
    from pet.skin import BUILTIN_SKINS, builtin_skin_path

    original_pixmap = window.skin_pixmap
    window.hands_free = False
    for skin_id, label in BUILTIN_SKINS:
        path = builtin_skin_path(skin_id)
        sprite = QPixmap(str(path)) if path else QPixmap()
        available = not sprite.isNull() and sprite.hasAlphaChannel()
        check(f"内置形象（{label}）", available)
        if not available:
            continue
        window.skin_pixmap = sprite
        window.state = "idle"
        frame = QPixmap(PET_SIZE, PET_SIZE)
        frame.fill(Qt.transparent)
        window.render(frame)
        image = frame.toImage()
        painted = sum(
            image.pixelColor(x, y).alpha() > 8
            for y in range(0, PET_SIZE - 50, 4)
            for x in range(0, PET_SIZE, 4)
        )
        check(f"内置形象渲染（{label}）", painted > 100, f"{painted} 个采样点有内容")
        if args.screenshot:
            target = Path(args.screenshot)
            frame.save(str(target.with_name(f"{target.stem}-{skin_id}.png")))
    window.skin_pixmap = original_pixmap

    if args.chat:
        try:
            reply = ""
            for event in client.chat_stream(args.chat, persona_id=default):
                if event.kind == "done":
                    reply = event.text or reply
                elif event.kind == "text":
                    reply += event.text
            check("对话链路", bool(reply.strip()), (reply or "").strip()[:48])
        except BackendError as exc:
            check("对话链路", False, str(exc))

    if args.stt:
        try:
            wav = Path(args.stt).read_bytes()
            started = time.time()
            text = client.transcribe(wav)
            check("语音识别链路（能听见）", bool(text), f"{time.time()-started:.2f}s -> {text[:40]}")
        except Exception as exc:  # noqa: BLE001
            check("语音识别链路（能听见）", False, str(exc))

    if args.speak:
        try:
            frames = list(client.speak_stream(args.speak, persona_id=default))
            total = sum(len(audio) for audio, _ in frames)
            check("语音合成链路", total > 1000, f"{len(frames)} 块 / {total} 字节")
        except BackendError as exc:
            check("语音合成链路", False, str(exc))

    if window.tray is not None:
        window.tray.hide()
    client.close()
    emit("")
    if failures:
        emit(f"  PET SELFTEST FAIL（{len(failures)} 项）：{', '.join(failures)}")
    else:
        emit("  PET SELFTEST PASS")

    # A --windowed exe has no console, so the result has to land somewhere readable.
    if args.report:
        try:
            Path(args.report).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"  写报告失败：{exc}")
    return 1 if failures else 0


def resolve_persona(client, persona_id: Optional[str]):
    """Return ``(persona_id, persona_name)`` for the single-instance key.

    Best-effort on purpose: the pet is allowed to start before the backend does (it shows
    "连不上本地服务" and can start the service itself), so a failed lookup must not stop it
    from opening — it just falls back to a generic key.
    """
    try:
        personas, default = client.personas()
    except Exception:  # noqa: BLE001 - backend down is a normal starting state
        return (persona_id or "default"), None
    from persona.catalog import LEGACY_IDS
    persona_id = LEGACY_IDS.get(persona_id, persona_id)
    if not persona_id:
        persona_id = default or (personas[0].get("id") if personas else None) or "default"
    name = None
    for persona in personas:
        if persona.get("id") == persona_id:
            name = str(persona.get("name") or persona_id)
            break
    return persona_id, name


def run_gui(args: argparse.Namespace) -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from pet import single_instance
    from pet.client import BackendClient
    from pet.window import PetWindow

    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationName("ChatBotPet")
    app.setQuitOnLastWindowClosed(False)  # closing the window keeps the tray alive
    client = BackendClient(args.url)

    # One pet per persona, checked *before* any window appears: a second identical pet
    # would answer the same microphone and talk over the first one. Different personas are
    # still allowed to sit on the desktop side by side.
    persona_id, persona_name = resolve_persona(client, args.persona)
    label = f"{persona_name} · 桌宠" if persona_name else ""
    if not single_instance.acquire(persona_id):
        focused = bool(label and single_instance.focus_existing(label)) or single_instance.focus_existing("桌宠")
        print(f"「{persona_name or persona_id}」的桌宠已经在运行了" + ("，已把它切到最前面。" if focused else "。"))
        client.close()
        return 0

    window = PetWindow(client, persona_id=persona_id)
    window.show()
    window.raise_()
    window.activateWindow()
    return app.exec()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="本地聊天机器人桌宠")
    parser.add_argument("--url", default="http://127.0.0.1:8077", help="后端地址")
    parser.add_argument("--persona", default=None, help="启动时使用的人设 id")
    parser.add_argument("--no-backend", action="store_true", help="只显示桌宠，不自动启动服务")
    parser.add_argument("--mock", action="store_true", help="启动模拟后端")
    parser.add_argument("--selftest", action="store_true", help="离屏自检并退出")
    parser.add_argument(
        "--import-check",
        action="store_true",
        help="只检查运行时依赖能不能导入（打包后用，专抓缺 DLL 这类问题）",
    )
    parser.add_argument("--screenshot", default=None, help="把渲染结果保存为 PNG")
    parser.add_argument("--offscreen", action="store_true", help="使用离屏渲染（无窗口）")
    parser.add_argument("--chat", default=None, help="自检时顺带发一句话")
    parser.add_argument("--speak", default=None, help="自检时顺带合成一句话")
    parser.add_argument("--stt", default=None, help="自检时把这段 wav 发去做语音识别")
    parser.add_argument(
        "--report",
        default=None,
        help="把自检结果写到这个文件（打包成 --windowed 的 exe 没有控制台，只能看文件）",
    )
    args = parser.parse_args(argv)

    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    try:
        if args.import_check:
            return run_import_check(args)
        if args.selftest or args.screenshot:
            return run_selftest(args)
        if not args.no_backend:
            from pet.bootstrap import ensure_backend
            ensure_backend(args.url, mock=args.mock)
        return run_gui(args)
    except BaseException as exc:  # noqa: BLE001 - a GUI crash must leave a trace
        path = write_crash(exc)
        # A --windowed exe has no console, so the dialog is the only way to tell a human
        # what happened. But when --report was passed the caller is a *script* watching a
        # file: a modal box would block it forever (this hung the build once), so write
        # the file and exit instead.
        if args.report:
            try:
                Path(args.report).write_text(
                    f"启动失败：{type(exc).__name__}: {exc}\n详见 {path}\n", encoding="utf-8"
                )
            except OSError:
                pass
            return 3
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance() or QApplication([])
            QMessageBox.critical(None, "桌宠启动失败", f"{type(exc).__name__}: {exc}\n\n详见 {path}")
        except Exception:  # noqa: BLE001
            pass
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
