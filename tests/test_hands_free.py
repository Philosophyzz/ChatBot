"""Hands-free listening tests — the bug that made 「一直听着」 do nothing.

Two independent defects, both invisible without a microphone, both pinned here:

1. ``Recorder.level()`` returned the peak **since recording started**. Once the user said
   anything it stayed above the threshold forever, so the "quiet for a while" branch could
   never run and nothing was ever sent. Fixed by :class:`pet.audio.LevelMeter`, whose value
   decays with a short window.
2. Re-arming after a turn was a one-shot timer that silently no-op'd while the reply was being
   generated, so hands-free died after a single exchange. That logic now lives in
   :class:`pet.hands_free.HandsFreeListener`.

A third problem only showed up once a real microphone was measured (see
``tools/check_hands_free.py``): this desktop's idle mic reads ~0.012 with 100 ms transients up
to 0.041, so a fixed 0.02 threshold fires on nothing at all. The listener therefore raises its
threshold above the *measured* noise floor, and these tests cover that too.
"""

from __future__ import annotations

import struct
from collections import deque

import pytest

from pet.hands_free import (
    EVENT_NONE,
    EVENT_SEND,
    EVENT_SPEECH_STARTED,
    HandsFreeConfig,
    HandsFreeListener,
)


def _pcm(amplitude: int, samples: int = 1600) -> bytes:
    return struct.pack(f"<{samples}h", *([amplitude] * samples))


def _listener(**overrides) -> HandsFreeListener:
    listener = HandsFreeListener(HandsFreeConfig(**overrides))
    listener.enable(now=0.0)
    return listener


def _run(listener: HandsFreeListener, levels, start: float = 0.0, step: float = 0.25):
    """Feed a level trace; returns (events, end_time)."""
    now = start
    events = []
    for level in levels:
        now += step
        events.append(listener.tick(level, now))
    return events, now


# -- level meter -------------------------------------------------------------------------
def test_level_meter_decays_instead_of_remembering_the_loudest_moment() -> None:
    """The core regression: loud once must NOT mean loud forever."""
    from pet.audio import LevelMeter, block_peak

    meter = LevelMeter(window_s=0.35)
    assert meter.level(now=0.0) == 0.0, "还没喂数据时应该是 0"

    meter.feed(_pcm(6000), now=0.0)  # 6000/32768 ≈ 0.18，说话级别
    assert meter.level(now=0.0) == pytest.approx(0.183, abs=0.01)
    assert meter.level(now=0.3) == pytest.approx(0.183, abs=0.01), "窗口内仍算有声"

    meter.feed(_pcm(0), now=0.4)
    assert meter.level(now=0.6) == 0.0, "超过窗口后必须归零，否则永远等不到静音"
    assert block_peak(_pcm(32767)) == pytest.approx(1.0, abs=0.001)
    assert block_peak(b"") == 0.0

    meter.reset()
    assert meter.level(now=0.6) == 0.0


def test_meter_keeps_the_peak_of_the_window_not_the_last_block() -> None:
    """A single loud block inside the window still counts — speech is not a sine wave."""
    from pet.audio import LevelMeter

    meter = LevelMeter(window_s=0.5)
    meter.feed(_pcm(9000), now=0.0)
    meter.feed(_pcm(0), now=0.1)
    meter.feed(_pcm(0), now=0.2)
    assert meter.level(now=0.25) == pytest.approx(0.274, abs=0.01)


def test_recorder_keeps_only_the_most_recent_audio_while_waiting() -> None:
    """Hands-free records while it waits, so the buffer must not grow without bound.

    An hour of waiting at 16 kHz mono would be ~115 MB, and the clip handed to the recogniser
    would start with minutes of silence.
    """
    from pet.audio import Recorder

    recorder = Recorder()
    recorder._frames = deque((_pcm(0), 1600) for _ in range(50))  # 5 秒
    assert recorder.buffered_seconds() == pytest.approx(5.0, abs=0.01)

    recorder.retain_last(0.35)
    assert recorder.buffered_seconds() <= 0.45, "只该留下最后 0.35 秒（含一个整块）"
    assert recorder.buffered_seconds() > 0.0

    recorder.retain_last(0.0)  # 不许把缓冲清空成负数或报错
    assert recorder.buffered_seconds() > 0.0
    recorder._frames.clear()


# -- basic behaviour ---------------------------------------------------------------------
def test_silence_never_sends_anything() -> None:
    """Waiting quietly must not spam empty clips (and must not look like speech)."""
    listener = _listener()
    events, _now = _run(listener, [0.001] * 40)  # 10 秒安静
    assert events == [EVENT_NONE] * 40
    assert listener.phase == "armed"


def test_speech_then_silence_sends_exactly_once() -> None:
    listener = _listener()
    events, now = _run(listener, [0.2] * 8)  # 2 秒说话
    assert EVENT_SPEECH_STARTED in events, "应该识别出用户开始说话"
    assert EVENT_SEND not in events, "还在说话，不该发出去"

    silence, _now = _run(listener, [0.0] * 8, start=now)  # 2 秒安静（> 0.9s 门槛）
    assert silence.count(EVENT_SEND) == 1, f"应该恰好发一次，实际 {silence}"
    assert listener.phase == "armed", "发完之后要回到等待状态"


def test_second_utterance_is_heard_too() -> None:
    """The other half of the bug: it must keep working after the first exchange."""
    listener = _listener()
    now = 0.0
    sends = 0
    for _round in range(3):
        events, now = _run(listener, [0.25] * 4, start=now)
        events, now = _run(listener, [0.0] * 6, start=now)
        sends += events.count(EVENT_SEND)
    assert sends == 3, f"三轮对话应该发三次，实际 {sends}"


def test_short_noise_is_ignored() -> None:
    """A cough or a bumped desk must not be transcribed."""
    listener = _listener(min_speech_s=0.1, min_utterance_s=0.6)
    events, _now = _run(listener, [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert EVENT_SEND not in events, "太短的声音应该被丢掉"
    assert listener.phase == "armed", "丢掉之后要继续等"


def test_hysteresis_keeps_a_sentence_alive_between_the_two_thresholds() -> None:
    """Between release and trigger is still "talking" — otherwise soft endings get cut off."""
    listener = _listener()  # trigger 0.05, release 0.03
    now = 0.0
    events, now = _run(listener, [0.3] * 4, start=now)
    assert EVENT_SPEECH_STARTED in events

    # 说话音量降到 0.04：高于 release(0.03)、低于 trigger(0.05) → 还没说完
    events, now = _run(listener, [0.04] * 10, start=now)
    assert EVENT_SEND not in events, "没到静音门槛就不该断句"

    events, now = _run(listener, [0.0] * 6, start=now)
    assert EVENT_SEND in events, "真的安静下来之后就该发出去"


def test_a_monologue_is_sent_eventually() -> None:
    """Talking for a minute must not mean never getting an answer."""
    listener = _listener(max_utterance_s=3.0)
    events, now = _run(listener, [0.4] * 20)  # 5 秒不停顿
    assert EVENT_SEND in events, "说太久也要发出去"

    # 停止说话后，它应该回到等待状态继续听下一句（而不是卡在 speech 里）
    events, now = _run(listener, [0.002] * 8, start=now)
    assert listener.phase == "armed"
    events, now = _run(listener, [0.3] * 4, start=now)
    assert EVENT_SPEECH_STARTED in events, "长句之后还能继续听下一句"


def test_switching_sensitivity_mid_sentence_keeps_listening() -> None:
    """Regression (user-reported): 切换灵敏度之后语音识别失败.

    ``set_sensitivity`` reset the timers while ``phase`` stayed ``"speech"``, leaving no start
    time. The next tick then tripped an assertion — and because that runs inside a Qt timer
    slot, PySide escalated it to a fatal error and the pet disappeared. Changing thresholds
    must not disturb the sentence in progress.
    """
    listener = _listener()
    events, now = _run(listener, [0.3] * 4)
    assert EVENT_SPEECH_STARTED in events
    assert listener.phase == "speech"

    listener.set_sensitivity("noisy")
    assert listener.phase == "speech", "切档不该丢掉正在说的这句话"

    # 继续安静下去 → 应该正常断句发出，而不是抛异常
    events, now = _run(listener, [0.0] * 10, start=now)
    assert EVENT_SEND in events, f"切档之后这句也应该能发出去：{events}"
    assert listener.phase == "armed"


def test_switching_sensitivity_between_sentences_is_seamless() -> None:
    listener = _listener()
    _run(listener, [0.002] * 8)
    before = listener.trigger_level()
    listener.set_sensitivity("noisy")
    assert listener.trigger_level() > before
    events, now = _run(listener, [0.4] * 6, start=8 * 0.25)
    assert EVENT_SPEECH_STARTED in events, "切档后仍然要能识别说话"
    events, _now = _run(listener, [0.002] * 10, start=now)
    assert EVENT_SEND in events


def test_tick_recovers_from_an_inconsistent_state_instead_of_raising() -> None:
    """Belt and braces: even if a future edit breaks the state, tick() must not raise.

    It is called from a Qt timer callback; an exception there kills the application.
    """
    listener = _listener()
    _run(listener, [0.3] * 4)
    assert listener.phase == "speech"
    listener._speech_started = None  # simulate the broken state the tray menu used to create

    assert listener.tick(0.0, 99.0) == EVENT_NONE, "不该抛异常"
    assert listener.phase == "armed", "应该退回等待状态继续听"

    """The pet must not answer its own voice, and must start listening again afterwards."""
    listener = _listener()
    now = 0.0
    listener.pause()
    assert listener.paused
    events, now = _run(listener, [0.9] * 12, start=now)
    assert events == [EVENT_NONE] * 12, "暂停期间再响也不该有反应（那是它自己在说话）"

    listener.resume(now)
    assert listener.phase == "armed"
    events, now = _run(listener, [0.3] * 3, start=now)
    assert EVENT_SPEECH_STARTED in events, "恢复后应该还能正常断句"


def test_long_silence_is_reported_as_inaudible() -> None:
    """The "switched sensitivity and now it never hears me" case, made visible.

    Nothing arrives near the trigger for 20 s → the pet can say "I can't hear you" instead of
    listening silently forever. A single loud burst must clear the flag again.
    """
    listener = _listener()
    assert not listener.seems_inaudible(5.0), "刚开始等待时不该报警"
    _run(listener, [0.002] * 40)  # 安静 ~10 秒
    assert listener.armed_at is not None
    assert not listener.seems_inaudible(listener.armed_at + 10), "不到 20 秒不该报警"
    assert listener.seems_inaudible(listener.armed_at + 25), "一直没声音就该提示"

    # 只要真的听到过一次接近阈值的音量，就不再是"听不见"
    listener.peak_since_arm = listener.trigger_level()
    assert not listener.seems_inaudible(listener.armed_at + 25)

    # 暂停之后也不再提示
    listener.pause()
    assert not listener.seems_inaudible(listener.armed_at + 25)


def test_disabled_listener_ignores_everything() -> None:
    listener = _listener()
    listener.disable()
    events, _now = _run(listener, [0.9] * 10)
    assert events == [EVENT_NONE] * 10
    assert listener.phase == "off"
    assert not listener.listening


# -- adaptive noise floor ----------------------------------------------------------------
def test_noise_floor_raises_the_threshold_on_a_noisy_desk() -> None:
    """Measured on this machine: idle mic ≈0.012 with 0.04 transients.

    A fixed 0.02 threshold sent a clip every few seconds in that room; the adaptive floor is
    what makes hands-free usable without a human tuning numbers.
    """
    listener = _listener()
    # 4 秒环境噪声：底噪 0.013，偶尔一个 0.03 的瞬态
    trace = []
    for index in range(16):
        trace.append(0.03 if index % 6 == 5 else 0.013)
    events, now = _run(listener, trace)
    assert EVENT_SEND not in events, f"环境噪声不该被断句：{events}"
    assert listener.noise_floor() == pytest.approx(0.013, abs=0.004)
    assert listener.trigger_level() > 0.05, "阈值应该被抬到基础值之上"

    # 现在真的说话（0.2）：比噪声高一个数量级，必须认出来
    events, now = _run(listener, [0.2] * 6, start=now)
    assert EVENT_SPEECH_STARTED in events
    events, now = _run(listener, [0.013] * 8, start=now)
    assert EVENT_SEND in events, "说话之后安静下来要发出去"


def test_quiet_talker_still_gets_through_in_a_quiet_room() -> None:
    """Adaptive must not mean deaf: with a near-silent floor the base threshold applies."""
    listener = _listener()
    events, now = _run(listener, [0.002] * 12)  # 很安静的房间
    assert listener.trigger_level() == pytest.approx(0.05, abs=0.005)
    events, now = _run(listener, [0.08] * 6, start=now)  # 小声说话
    assert EVENT_SPEECH_STARTED in events
    events, now = _run(listener, [0.002] * 8, start=now)
    assert EVENT_SEND in events


def test_threshold_is_capped_so_a_loud_room_is_not_deafening() -> None:
    listener = _listener(max_speech_level=0.25)
    _run(listener, [0.2] * 40)  # 持续大声的环境（外放音乐）
    assert listener.trigger_level() <= 0.25


def test_sensitivity_presets_change_the_threshold() -> None:
    from pet.hands_free import SENSITIVITY_PRESETS, config_for

    assert set(SENSITIVITY_PRESETS) == {"quiet", "normal", "noisy"}
    assert config_for("quiet").speech_level < config_for("normal").speech_level
    assert config_for("noisy").speech_level > config_for("normal").speech_level
    assert config_for("不存在").speech_level == config_for("normal").speech_level, "未知名字回落到默认"

    listener = _listener()
    before = listener.trigger_level()
    listener.set_sensitivity("noisy")
    assert listener.trigger_level() > before, "切到抗噪档阈值应该更高"


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        HandsFreeConfig(release_ratio=0.0)
    with pytest.raises(ValueError):
        HandsFreeConfig(release_ratio=1.2)
    with pytest.raises(ValueError):
        HandsFreeConfig(speech_level=0.3, max_speech_level=0.2)
