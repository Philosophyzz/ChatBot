"""Pet-local settings.

The pet is a thin client: it talks to the server over HTTP and deliberately does not read
``config/config.yaml`` (that file configures the *engine*). Anything the pet itself owns —
hands-free sensitivity, custom character images, mute state — is stored here, as one small
JSON file, so it works the same whether the pet runs from source or from the packaged exe.

Location: ``<project>\\data\\pet_settings.json``. Both the source run and ``启动聊天机器人.exe``
resolve to the same project root (see :func:`project_root`), so a skin chosen in one is visible
in the other — and everything stays on D: as the user requires.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from core.logging import get_logger

log = get_logger(__name__)

SETTINGS_NAME = "pet_settings.json"

DEFAULTS: Dict[str, Any] = {
    #: "quiet" | "normal" | "noisy" — see pet.hands_free.SENSITIVITY_PRESETS.
    "hands_free_sensitivity": "normal",
    #: Whether the pet starts with 「一直听着」 already on.
    "hands_free_default": False,
    "muted": False,
    #: persona id -> image path (relative to the project root, so it survives a move).
    "skins": {},
    #: Used for every persona that has no skin of its own.
    "default_skin": None,
    #: Last window position, so the pet comes back where the user left it.
    "position": None,
    "scale": 1.0,
    "live2d_models": {},
    "awareness_enabled": True,
    "screen_awareness": True,
    "companion_scene": "normal",
}


def project_root() -> Optional[Path]:
    """Find the project root from either a source run or the frozen exe."""
    if getattr(sys, "frozen", False):
        start = Path(sys.executable).resolve().parent
    else:
        start = Path(__file__).resolve().parents[2]
    for candidate in (start, *start.parents):
        if (candidate / "config" / "config.yaml").exists() or (candidate / "src" / "core").is_dir():
            return candidate
    return start if start.exists() else None


def settings_path() -> Path:
    root = project_root()
    if root is None:
        return Path(tempfile.gettempdir()) / SETTINGS_NAME
    return root / "data" / SETTINGS_NAME


class PetSettings:
    """Dict-like settings with defaults, loaded once and saved atomically."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else settings_path()
        self.values: Dict[str, Any] = dict(DEFAULTS)
        self.load()

    def load(self) -> Dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.values.update(raw)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:  # corrupt file: keep defaults, do not crash the pet
            log.warning("could not read pet settings, using defaults", extra={"error": str(exc)})
        return self.values

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-replace: a crash mid-write must not leave an unparsable settings file.
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.values, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("could not save pet settings", extra={"error": str(exc), "path": str(self.path)})

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.values:
            return self.values[key]
        return DEFAULTS.get(key, default)

    def set(self, key: str, value: Any, *, save: bool = True) -> None:
        self.values[key] = value
        if save:
            self.save()

    # -- skins -------------------------------------------------------------------------
    def skin_for(self, persona_id: Optional[str]) -> Optional[str]:
        """Absolute path of the image to use for this persona, if any."""
        skins = self.values.get("skins") or {}
        from persona.catalog import CATALOG
        key = persona_id
        # Built-in companions always recover their matching outfit. Global legacy skins
        # must not turn every new character into the same animal.
        candidate = skins.get(key)
        if not candidate and key in CATALOG:
            candidate = f"builtin:{key}"
        candidate = candidate or self.values.get("default_skin")
        if not candidate:
            return None
        if str(candidate).startswith("builtin:"):
            from pet.skin import builtin_skin_path

            bundled = builtin_skin_path(str(candidate).removeprefix("builtin:"))
            return str(bundled) if bundled else None
        path = Path(candidate)
        if not path.is_absolute():
            root = project_root()
            path = (root / candidate) if root else path
        return str(path) if path.exists() else None

    def set_skin(self, persona_id: Optional[str], stored: Optional[str]) -> None:
        """Record ``stored`` (a path relative to the project root, or None to clear)."""
        skins = dict(self.values.get("skins") or {})
        if persona_id:
            if stored:
                skins[persona_id] = stored
            else:
                skins.pop(persona_id, None)
            self.values["skins"] = skins
        else:
            self.values["default_skin"] = stored
        self.save()


def get_settings(**kwargs) -> PetSettings:
    return PetSettings(**kwargs)
