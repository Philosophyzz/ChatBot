"""Stop our own GPU models from oversubscribing a 16 GB card.

Three models want this GPU: the chat model (llama.cpp, ~5.3 GB for 9B), whisper (~1.8 GB) and
IndexTTS2 (~4 GB), plus whatever the desktop is using (3–4 GB with a browser open). **Any two
fit; all three do not.**

When it does not fit, the failure is not a Python exception — nothing to catch, nothing to log.
Windows recorded exactly this on 2026-09-22 21:16:48::

    出错应用程序名称: python.exe
    出错模块名称:   ucrtbase.dll
    异常代码:       0xc0000409        (STATUS_STACK_BUFFER_OVERRUN → fail-fast abort)

four seconds after a voice turn, while whisper and IndexTTS2 were both resident. The service
simply vanishes; the browser says "请求失败：Failed to fetch" and the pet says it cannot connect.

So: before one of the big optional models loads, if free VRAM is not enough, unload the other.
Reloading costs 2–4 seconds; dying costs the whole conversation.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from core.logging import get_logger

log = get_logger(__name__)

#: Keep this much VRAM spare for the CUDA context / fragmentation.
DEFAULT_MARGIN_GB = 0.8


def nvidia_free_gb() -> Optional[float]:
    """Free VRAM in GB, or None when it cannot be determined (no GPU/nvidia-smi)."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed command
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        first = (completed.stdout or "").strip().splitlines()[0]
        return float(first.strip()) / 1024.0
    except Exception:  # noqa: BLE001 - no GPU is a supported configuration
        return None


@dataclass
class Resident:
    """A GPU model we can unload on demand."""

    name: str
    needs_gb: float
    unload: Optional[Callable[[], None]] = None
    note: str = ""
    unloads: int = field(default=0)


class VramCoordinator:
    """Knows which of our models are big, and evicts one before another loads."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        margin_gb: float = DEFAULT_MARGIN_GB,
        probe: Optional[Callable[[], Optional[float]]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.margin_gb = float(margin_gb)
        self._probe = probe or nvidia_free_gb
        self._models: Dict[str, Resident] = {}
        self._lock_free = None  # kept simple: callers serialise their own loads

    # -- registry ----------------------------------------------------------------------
    def register(
        self,
        name: str,
        *,
        needs_gb: float,
        unload: Optional[Callable[[], None]] = None,
        note: str = "",
    ) -> None:
        self._models[name] = Resident(name=name, needs_gb=float(needs_gb), unload=unload, note=note)

    def registered(self) -> List[str]:
        return sorted(self._models)

    def free_gb(self) -> Optional[float]:
        return self._probe()

    # -- the actual guard --------------------------------------------------------------
    def before_load(self, name: str) -> List[str]:
        """Make room for ``name``; returns the models that were unloaded.

        Called right before a big model is loaded. Never raises: a broken probe or a failing
        unload must not turn "we might run out of VRAM" into "voice input is broken".
        """
        if not self.enabled or name not in self._models:
            return []
        wanted = self._models[name]
        try:
            free = self.free_gb()
        except Exception as exc:  # noqa: BLE001
            log.warning("vram probe failed", extra={"error": str(exc)})
            return []
        if free is None:
            # Unknown (no nvidia-smi): do nothing rather than thrash models on every call.
            return []
        needed = wanted.needs_gb + self.margin_gb
        if free >= needed:
            return []

        unloaded: List[str] = []
        # Biggest first: evicting IndexTTS2 usually frees enough for whisper in one go.
        others = sorted(
            (item for key, item in self._models.items() if key != name and item.unload),
            key=lambda item: item.needs_gb,
            reverse=True,
        )
        for other in others:
            if free >= needed:
                break
            try:
                log.info(
                    "freeing vram for %s by unloading %s",
                    name,
                    other.name,
                    extra={"free_gb": round(free, 2), "needed_gb": round(needed, 2)},
                )
                other.unload()  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001
                log.warning("unloading %s failed", other.name, extra={"error": str(exc)})
                continue
            other.unloads += 1
            unloaded.append(other.name)
            try:
                free = self.free_gb() or free + other.needs_gb
            except Exception:  # noqa: BLE001
                free += other.needs_gb
        if not unloaded and free < needed:
            log.warning(
                "not enough vram for %s and nothing left to unload",
                name,
                extra={"free_gb": round(free, 2), "needed_gb": round(needed, 2)},
            )
        return unloaded

    def describe(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "margin_gb": self.margin_gb,
            "free_gb": self.free_gb() if self.enabled else None,
            "models": {
                name: {"needs_gb": item.needs_gb, "unloadable": item.unload is not None, "unloads": item.unloads}
                for name, item in self._models.items()
            },
        }


#: Process-wide instance; the speech factory configures it at startup.
coordinator = VramCoordinator()


def configure(*, enabled: bool, margin_gb: float = DEFAULT_MARGIN_GB) -> VramCoordinator:
    """Apply configuration (called once by ``build_speech``-style factories)."""
    coordinator.enabled = bool(enabled)
    coordinator.margin_gb = float(margin_gb)
    return coordinator
