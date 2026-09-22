"""Single-instance guard: one desktop pet per persona.

Why a named mutex instead of a lock file: the kernel owns the object, so it disappears
automatically when the process dies — no stale lock to detect, no cleanup path to get
wrong, and it works even if two copies are started in the same instant (Windows
serialises ``CreateMutexW`` on the name).

The name is per persona, so 「温柔姐姐」 and 「阿澈」 may sit on the desktop together,
but double-clicking the same persona twice does not produce two identical pets fighting
over the microphone. Switching persona inside a running pet hands the old name back and
claims the new one, so the "one pet per persona" rule holds however the pet was started.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Optional

#: ERROR_ALREADY_EXISTS — CreateMutex succeeded, but the name was taken.
_ERROR_ALREADY_EXISTS = 183
_MUTEX_PREFIX = "Local\\ChatBotPet_"

#: key -> mutex handle. Handles are kept for the process lifetime (or until released) so
#: the kernel keeps the name owned; dropping the handle would silently free the guard.
_handles: dict = {}


def acquire(key: str) -> bool:
    """Try to become *the* pet instance for ``key``. Returns False if one already runs."""
    if sys.platform != "win32":  # pragma: no cover - the pet is Windows-only in practice
        return True
    if key in _handles:
        return False  # we already hold it ourselves (e.g. re-selecting the same persona)
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.SetLastError(0)
    handle = kernel32.CreateMutexW(None, False, _MUTEX_PREFIX + key)
    if not handle:
        return True  # cannot tell → allow (never block the user behind our own guard)
    if kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return False
    _handles[key] = handle
    return True


def release(key: str) -> None:
    """Give up the guard for one key — used when the pet switches persona."""
    if sys.platform != "win32":  # pragma: no cover
        return
    handle = _handles.pop(key, None)
    if not handle:
        return
    try:
        ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:  # noqa: BLE001
        pass


def held_keys() -> list:
    """Diagnostic/testing helper: which keys this process is guarding."""
    return list(_handles)


def release_all() -> None:
    """Drop every guard (tests, or an explicit hand-over)."""
    for key in list(_handles):
        release(key)


def _windows(title_contains: str = "桌宠", *, visible_only: bool = True) -> list:
    """``[(hwnd, title, (x, y, w, h)), ...]`` for windows whose title matches."""
    if sys.platform != "win32":  # pragma: no cover
        return []
    user32 = ctypes.windll.user32
    found: list = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):  # noqa: ANN001
        if visible_only and not user32.IsWindowVisible(hwnd):
            return True
        buffer = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buffer, 256)
        if title_contains in buffer.value:
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            found.append(
                (
                    hwnd,
                    buffer.value,
                    (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top),
                )
            )
        return True

    user32.EnumWindows(callback, 0)
    return found


#: ShowWindow 命令；None = 别碰窗口，直接置前就行。
SW_RESTORE = 9
SW_SHOW = 5


def focus_action(*, iconic: bool, visible: bool) -> Optional[int]:
    """Which ``ShowWindow`` command to send before focusing, if any.

    Calling ``SW_RESTORE`` on a window that is *not* minimised makes Windows move it to its
    stored "restore" position — the pet jumped from ``(1943, 915)`` to ``(1921, 774)`` every
    time a second double-click focused it. So: only restore when actually minimised, only
    show when actually hidden (the pet can be parked in the tray).
    """
    if iconic:
        return SW_RESTORE
    if not visible:
        return SW_SHOW
    return None


def focus_existing(title_contains: str = "桌宠") -> bool:
    """Bring the already-running pet to the front. Returns True if a window matched."""
    if sys.platform != "win32":  # pragma: no cover
        return False
    # Hidden windows count here: a pet parked in the tray must come back when the user
    # double-clicks the shortcut again, instead of the click appearing to do nothing.
    found = _windows(title_contains, visible_only=False)
    if not found:
        return False
    # Prefer an exact title: several pets can run (one per persona), and the caller asking
    # for 「温柔姐姐 · 桌宠」 must not be handed 「阿澈 · 桌宠」 just because it is on top.
    exact = [item for item in found if item[1] == title_contains]
    hwnd = (exact or found)[0][0]
    user32 = ctypes.windll.user32
    action = focus_action(iconic=bool(user32.IsIconic(hwnd)), visible=bool(user32.IsWindowVisible(hwnd)))
    if action is not None:
        user32.ShowWindow(hwnd, action)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    return True


def pet_window_rects() -> list:
    """Screen rects of the pets already on the desktop, so a new one can dodge them.

    Two personas are allowed to run together, and without this they both landed on the
    exact same bottom-right pixel — the second pet sat invisibly on top of the first.
    Only *visible* pets reserve a slot: one hidden in the tray is not in the way.
    """
    return [rect for _hwnd, title, rect in _windows("· 桌宠")]


def existing_window_titles(title_contains: str = "桌宠") -> list:
    """Visible window titles matching the pet, for diagnostics and tests."""
    return [title for _hwnd, title, _rect in _windows(title_contains)]


def holder_pid() -> Optional[int]:
    """Diagnostic: which process owns a mutex is not queryable, so this returns None.

    The pet reports a conflict to the user by focusing the existing window instead, which
    answers the question they actually have ("where did my pet go?").
    """
    return None
