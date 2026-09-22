"""VRAM guard tests — the fix for the silently dying service.

On 2026-09-22 the API process was killed by its own C++ runtime (Windows Application Error,
`python.exe`, faulting module `ucrtbase.dll`, exception 0xC0000409 = fail-fast abort) four
seconds after a voice turn. What was resident: the 9B chat model, whisper, and IndexTTS2 —
three GPU models on a 16 GB card that also feeds a desktop. Nothing catchable happens; the
service is simply gone, so the browser says "请求失败：Failed to fetch".

The rule under test: before loading one of the big optional models, evict the other if free
VRAM is not enough.
"""

from __future__ import annotations

import pytest

from core.vram import VramCoordinator


def _coordinator(free_gb):
    """Coordinator with a scripted VRAM probe and recording unloads."""
    unloaded = []
    state = {"free": free_gb}

    def probe():
        return state["free"]

    coordinator = VramCoordinator(probe=probe, margin_gb=0.8)
    coordinator.register("stt", needs_gb=1.8, unload=lambda: (unloaded.append("stt"), state.update(free=state["free"] + 1.8)))
    coordinator.register(
        "indextts",
        needs_gb=4.0,
        unload=lambda: (unloaded.append("indextts"), state.update(free=state["free"] + 4.0)),
    )
    return coordinator, unloaded, state


def test_whisper_is_evicted_before_indextts_loads_when_vram_is_short() -> None:
    """1 GB free + a 4 GB model → the other model must go first."""
    coordinator, unloaded, _state = _coordinator(1.0)
    freed = coordinator.before_load("indextts")
    assert freed == ["stt"], f"应该先卸掉 whisper，实际 {freed}"
    assert unloaded == ["stt"]


def test_nothing_is_evicted_when_there_is_room() -> None:
    """Churning models on every call would make voice input needlessly slow."""
    coordinator, unloaded, _state = _coordinator(9.0)
    assert coordinator.before_load("indextts") == []
    assert coordinator.before_load("stt") == []
    assert unloaded == []


def test_biggest_consumer_is_evicted_first() -> None:
    """With two candidates, freeing the largest is the one move that usually suffices."""
    coordinator, unloaded, _state = _coordinator(0.5)
    coordinator.register("other-big", needs_gb=3.0, unload=lambda: unloaded.append("other-big"))
    freed = coordinator.before_load("stt")
    assert freed[0] == "indextts", f"应该先卸最大的，实际 {freed}"


def test_unknown_free_vram_means_no_churn() -> None:
    """No nvidia-smi (or a broken one) must not make every load unload everything."""
    coordinator, unloaded, _state = _coordinator(None)
    assert coordinator.before_load("stt") == []
    assert unloaded == []


def test_a_failing_unload_does_not_break_the_load() -> None:
    """The guard is best-effort: it must never turn "low VRAM" into "voice is broken"."""
    def boom():
        raise RuntimeError("cannot unload")

    coordinator, _unloaded, _state = _coordinator(0.5)
    coordinator.register("stt", needs_gb=1.8, unload=boom)
    assert coordinator.before_load("indextts") == []  # 卸载失败 → 不报告成功，也不抛异常


def test_unknown_model_name_is_ignored() -> None:
    coordinator, unloaded, _state = _coordinator(0.1)
    assert coordinator.before_load("not-registered") == []
    assert unloaded == []


def test_disabled_guard_does_nothing() -> None:
    coordinator, unloaded, _state = _coordinator(0.1)
    coordinator.enabled = False
    assert coordinator.before_load("indextts") == []
    assert unloaded == []


def test_registered_models_are_reported_for_diagnostics() -> None:
    coordinator, _unloaded, _state = _coordinator(5.0)
    described = coordinator.describe()
    assert set(described["models"]) == {"stt", "indextts"}
    assert described["models"]["indextts"]["unloadable"] is True
    assert described["free_gb"] == 5.0


def test_speech_factory_registers_both_backends() -> None:
    """The wiring matters as much as the logic: a forgotten register() means no guard.

    This bit me for real: ``build_tts`` passed the backend dict as ``{name: backend}`` while
    the helper expected ``{"tts": {name: backend}}``, so IndexTTS2 was never registered and the
    guard logged "nothing left to unload" instead of evicting whisper.
    """
    from core.config import SpeechConfig
    from core.vram import coordinator
    from speech import _register_vram_users
    from speech.tts import EdgeTTSBackend, IndexTTSBackend, SapiTTSBackend

    class FakeSTT:
        vram_gb = 1.8

        def __init__(self) -> None:
            self.unloaded = 0

        def unload(self) -> None:
            self.unloaded += 1

    class FakeConfig:
        speech = SpeechConfig()
        extra: dict = {}

    stt = FakeSTT()
    coordinator._models.clear()
    try:
        _register_vram_users(
            FakeConfig(),
            {"stt": stt, "tts": {"indextts": IndexTTSBackend(model_dir=None), "edge": EdgeTTSBackend(), "sapi": SapiTTSBackend()}},
        )
        # edge/sapi hold no GPU model and must not be registered as evictable.
        assert set(coordinator.registered()) == {"stt", "indextts"}
        assert coordinator._models["indextts"].needs_gb == pytest.approx(4.0)

        # 真的会调用到后端自己的 unload
        coordinator._probe = lambda: 0.1
        coordinator.before_load("indextts")
        assert stt.unloaded == 1, "加载 IndexTTS2 之前必须先把 whisper 卸掉"
    finally:
        coordinator._models.clear()
