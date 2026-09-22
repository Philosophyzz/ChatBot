"""Hands-free listening: decide when an utterance starts and when it is over.

Why this is a module of its own with a state machine instead of six lines in a timer
callback — the previous version was wrong in two ways that are invisible without a
microphone, and the user hit both:

* it asked ``Recorder.level()`` for loudness, but that returned the peak **since recording
  started**. Once the user said anything, the value stayed above the threshold forever, so
  the "quiet for a while" branch could never run: "一直听着" just sat there listening and
  never sent anything. (Fixed on the recorder side: see :class:`pet.audio.LevelMeter`.)
* after a turn it re-armed with ``QTimer.singleShot(1200, start_recording)``, but
  ``start_recording`` returns early while the reply is still being generated — so hands-free
  died after a single exchange.

The rules, stated once and testable without hardware::

    off ──enable──▶ armed ──speech for min_speech_s──▶ speech ──quiet for silence_s──▶ send
                     ▲                                   │
                     └────── resume when the pet is idle ─┘  (paused while it thinks/speaks)

While the pet is thinking or speaking the listener is *paused*: it would otherwise record the
pet's own voice through the speakers and answer itself.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, Optional

PHASE_OFF = "off"
PHASE_ARMED = "armed"
PHASE_SPEECH = "speech"
PHASE_PAUSED = "paused"

#: Tick results.
EVENT_NONE = ""
EVENT_SPEECH_STARTED = "speech_started"
EVENT_SEND = "send"


@dataclass
class HandsFreeConfig:
    """Thresholds. Defaults are for a headset mic in an ordinary room.

    The fixed 0.02 threshold the first version used is hopeless on a real machine: measured
    on this desktop, the *idle* microphone sits at ~0.012 with 100 ms transients up to 0.041
    (fan, desk bumps, leakage from the speakers), so a fixed low threshold fires on nothing
    and a fixed high one misses a quiet talker. Hence: a base level, a margin above the
    measured noise floor, and a ceiling.
    """

    #: Speech must be at least this loud (fraction of full scale)…
    speech_level: float = 0.05
    #: …and at least this many times the measured noise floor.
    noise_margin: float = 4.0
    #: Never demand more than this, however noisy the room looks.
    max_speech_level: float = 0.25
    #: Below this counts as quiet — derived from the live trigger (hysteresis, so a voice
    #: hovering at the threshold does not flap between the two phases).
    release_ratio: float = 0.6
    #: Noise floor = this percentile of the recent "quiet" samples.
    floor_percentile: float = 0.2
    #: Length of the window the noise floor is estimated over.
    floor_window_s: float = 4.0
    #: How long the level must stay above the trigger before it is really speech. Rejects the
    #: 100 ms clicks that the level meter would otherwise stretch to ~0.35 s.
    min_speech_s: float = 0.35
    #: How long it must stay quiet before the utterance is considered finished.
    silence_s: float = 0.9
    #: An utterance shorter than this is noise (a cough, a bumped desk) and is dropped.
    min_utterance_s: float = 0.4
    #: Send anyway after this long, so a monologue still gets an answer.
    max_utterance_s: float = 20.0
    #: How often the window should call :meth:`HandsFreeListener.tick`.
    poll_s: float = 0.25

    def __post_init__(self) -> None:
        if not 0 < self.release_ratio < 1:
            raise ValueError("release_ratio 必须在 (0, 1) 之间")
        if self.speech_level <= 0 or self.max_speech_level < self.speech_level:
            raise ValueError("阈值设置不合法")


#: Tray-menu presets. ``quiet`` for a headset in a silent room, ``noisy`` when the speakers
#: are loud enough to leak into the microphone.
SENSITIVITY_PRESETS = {
    "quiet": {"speech_level": 0.03, "noise_margin": 3.0, "max_speech_level": 0.2},
    "normal": {"speech_level": 0.05, "noise_margin": 4.0, "max_speech_level": 0.25},
    "noisy": {"speech_level": 0.09, "noise_margin": 5.0, "max_speech_level": 0.35},
}
SENSITIVITY_LABELS = {
    "quiet": "灵敏（安静房间 / 耳机麦）",
    "normal": "普通（默认）",
    "noisy": "抗噪（外放、风扇、键盘声大）",
}
DEFAULT_SENSITIVITY = "normal"


def config_for(sensitivity: str, **overrides) -> HandsFreeConfig:
    """Build a config from a preset name (unknown names fall back to the default)."""
    preset = SENSITIVITY_PRESETS.get(sensitivity, SENSITIVITY_PRESETS[DEFAULT_SENSITIVITY])
    return HandsFreeConfig(**{**preset, **overrides})


class HandsFreeListener:
    """Turns a stream of loudness samples into "start talking" / "send it" events."""

    def __init__(self, config: Optional[HandsFreeConfig] = None) -> None:
        self.config = config or HandsFreeConfig()
        self.enabled = False
        self.phase = PHASE_OFF
        self._above_since: Optional[float] = None
        self._quiet_since: Optional[float] = None
        self._speech_started: Optional[float] = None
        window = max(4, int(self.config.floor_window_s / max(0.05, self.config.poll_s)))
        #: Only "quiet" samples land here, so a long sentence cannot raise the floor.
        self._quiet_levels: Deque[float] = deque(maxlen=window)
        #: When the current "waiting for speech" phase began, and the loudest thing heard since.
        self.armed_at: Optional[float] = None
        self.peak_since_arm: float = 0.0

    # -- lifecycle ---------------------------------------------------------------------
    @property
    def paused(self) -> bool:
        return self.phase == PHASE_PAUSED

    @property
    def listening(self) -> bool:
        return self.phase in {PHASE_ARMED, PHASE_SPEECH}

    def enable(self, now: Optional[float] = None) -> None:
        self.enabled = True
        self._quiet_levels.clear()
        self.arm(now)

    def disable(self) -> None:
        self.enabled = False
        self.phase = PHASE_OFF
        self._reset_timers()

    def arm(self, now: Optional[float] = None) -> None:
        """Wait for speech to begin."""
        self._reset_timers()
        if self.enabled:
            self.phase = PHASE_ARMED
            self.armed_at = time.monotonic() if now is None else now
            self.peak_since_arm = 0.0

    def seems_inaudible(self, now: float, *, min_wait_s: float = 20.0, ratio: float = 0.8) -> bool:
        """Waiting a long time and nothing ever came near the trigger.

        Either the microphone is too far/too quiet, or the preset is too high for this room —
        the two situations the user cannot tell apart from the outside. The pet uses this to
        say something useful instead of silently listening forever.
        """
        if not self.enabled or self.phase != PHASE_ARMED or self.armed_at is None:
            return False
        if now - self.armed_at < min_wait_s:
            return False
        return self.peak_since_arm < self.trigger_level() * ratio

    def pause(self) -> None:
        """Stop reacting (the pet is thinking or speaking). No-op when disabled."""
        if not self.enabled:
            return
        self.phase = PHASE_PAUSED
        self._reset_timers()

    def resume(self, now: Optional[float] = None) -> None:
        """Start reacting again after a pause."""
        if self.enabled and self.phase == PHASE_PAUSED:
            self.arm(now)

    def set_sensitivity(self, sensitivity: str) -> HandsFreeConfig:
        """Switch preset while running (tray menu). Keeps the learned noise floor.

        Must not disturb an utterance that is already in progress. Resetting the timers
        blindly used to leave ``phase == "speech"`` with ``_speech_started is None``, and the
        next tick then tripped an assertion **inside a Qt timer slot** — which PySide turns
        into a fatal error, so the whole pet vanished. (Reported as "切换灵敏度之后语音识别
        失败".) Only the thresholds change here; the sentence keeps its start time.
        """
        started = self._speech_started
        self.config = config_for(sensitivity, poll_s=self.config.poll_s)
        window = max(4, int(self.config.floor_window_s / max(0.05, self.config.poll_s)))
        if self._quiet_levels.maxlen != window:
            self._quiet_levels = deque(self._quiet_levels, maxlen=window)
        self._reset_timers()
        if self.phase == PHASE_SPEECH:
            self._speech_started = started if started is not None else time.monotonic()
        return self.config

    # -- noise floor -------------------------------------------------------------------
    def noise_floor(self) -> float:
        """Estimated room/microphone noise level (0.0 until a few samples exist)."""
        samples = sorted(self._quiet_levels)
        if len(samples) < 3:
            return 0.0
        index = min(len(samples) - 1, int(len(samples) * self.config.floor_percentile))
        return samples[index]

    def trigger_level(self) -> float:
        """Level that currently counts as speech: base, raised to clear the measured noise."""
        floor = self.noise_floor()
        return min(
            self.config.max_speech_level,
            max(self.config.speech_level, floor * self.config.noise_margin),
        )

    def release_level(self) -> float:
        return self.trigger_level() * self.config.release_ratio

    # -- the machine -------------------------------------------------------------------
    def tick(self, level: float, now: Optional[float] = None) -> str:
        """Feed the current loudness; returns ``EVENT_NONE`` / ``EVENT_SPEECH_STARTED`` / ``EVENT_SEND``."""
        stamp = time.monotonic() if now is None else now
        if not self.enabled or self.phase in {PHASE_OFF, PHASE_PAUSED}:
            return EVENT_NONE

        trigger = self.trigger_level()
        self.peak_since_arm = max(self.peak_since_arm, level)
        if level < self.release_level():
            # Only *unambiguously* quiet samples feed the floor estimate. Using "below trigger"
            # instead would let speech that is merely softer (the hysteresis band) inflate the
            # floor, which then raises the trigger above the speaker's own voice.
            self._quiet_levels.append(level)

        if self.phase == PHASE_ARMED:
            if level > trigger:
                if self._above_since is None:
                    self._above_since = stamp
                elif stamp - self._above_since >= self.config.min_speech_s:
                    self.phase = PHASE_SPEECH
                    self._speech_started = self._above_since
                    self._quiet_since = None
                    return EVENT_SPEECH_STARTED
            else:
                self._above_since = None
            return EVENT_NONE

        # PHASE_SPEECH: waiting for the user to stop talking.
        if self._speech_started is None:
            # Defensive: this runs from a Qt timer callback, where *any* exception is fatal to
            # the whole application. An inconsistent state must therefore degrade into "start
            # listening again", never into a crash.
            self.arm(stamp)
            return EVENT_NONE
        if stamp - self._speech_started >= self.config.max_utterance_s:
            self.arm(stamp)
            return EVENT_SEND
        if level > self.release_level():
            self._quiet_since = None
            return EVENT_NONE
        if self._quiet_since is None:
            self._quiet_since = stamp
            return EVENT_NONE
        if stamp - self._quiet_since < self.config.silence_s:
            return EVENT_NONE

        spoken_s = stamp - self._speech_started
        self.arm(stamp)
        if spoken_s < self.config.min_utterance_s:
            # Too short to be a sentence: treat it as noise and keep waiting.
            return EVENT_NONE
        return EVENT_SEND

    def _reset_timers(self) -> None:
        self._above_since = None
        self._quiet_since = None
        self._speech_started = None


def describe(config: Optional[HandsFreeConfig] = None, listener: Optional["HandsFreeListener"] = None) -> str:
    """One-line summary for logs, the bubble, and the docs."""
    cfg = config or (listener.config if listener else None) or HandsFreeConfig()
    text = (
        f"阈值 {cfg.speech_level:.3f}（自适应不低于噪声 {cfg.noise_margin:.0f} 倍）／"
        f"静音 {cfg.silence_s:.1f}s／最短 {cfg.min_utterance_s:.1f}s"
    )
    if listener is not None:
        text += f"｜当前触发 {listener.trigger_level():.3f}，环境噪声 {listener.noise_floor():.3f}"
    return text
