"""验证「一直听着」到底能不能工作 —— 用真麦克风量一遍电平，再用状态机复算一遍结论。

免提模式出过的两个 bug（电平只增不减、回合一结束就不再续听）都是"没有麦克风就看不出来"
的类型，所以这个工具做的事很直接：录一段真实音频，打印电平曲线，并把同一串电平喂给
`pet.hands_free.HandsFreeListener`，看它会不会断句、会不会发出语音。

    python tools\\check_hands_free.py                 # 录 3 秒，看环境噪声电平
    python tools\\check_hands_free.py --seconds 6
    python tools\\check_hands_free.py --play          # 顺便用扬声器放一段测试音，验证麦克风真能听到
    python tools\\check_hands_free.py --play --speech # 放一段"说话"（多段起伏，更像真人）

判定标准：安静时电平应远低于阈值（默认 0.02），说话/播放时应稳定高于它。
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import wave
from io import BytesIO
from typing import List, Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def make_tone(seconds: float = 2.0, frequency: float = 440.0, amplitude: float = 0.28, rate: int = 16000) -> bytes:
    """A plain tone as WAV bytes — enough to prove the mic hears the speakers."""
    import math

    frames = bytearray()
    for index in range(int(seconds * rate)):
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * index / rate))
        frames += struct.pack("<h", value)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return buffer.getvalue()


def make_speech_like(seconds: float = 3.0, rate: int = 16000) -> bytes:
    """A tone that pulses like syllables, so the state machine sees speech then pauses."""
    import math

    frames = bytearray()
    total = int(seconds * rate)
    for index in range(total):
        position = index / rate
        syllable = 0.5 + 0.5 * math.sin(2 * math.pi * 3.5 * position)  # 3.5 次/秒的起伏
        quiet = 0.12 if (position % 3.0) > 2.4 else 1.0  # 每 3 秒留 0.6 秒停顿
        value = int(0.30 * quiet * syllable * 32767 * math.sin(2 * math.pi * 220 * position))
        frames += struct.pack("<h", value)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return buffer.getvalue()


def calibrate(config, seconds: float = 5.0) -> int:
    """Measure the room *and* the user's voice, then recommend a sensitivity preset.

    The presets are only useful if you know which one fits your microphone: a quiet headset
    reads 0.03 when you talk, a loud one 0.4, and that is a 13× difference. This walks the
    user through one sentence and says which preset will actually trigger.
    """
    import time as _time

    from pet.audio import AudioUnavailable, LevelMeter, Recorder
    from pet.hands_free import SENSITIVITY_PRESETS, config_for

    print("校准：先安静 2 秒（测环境噪声），然后正常说一句话（测你的音量）。")
    recorder = Recorder()
    try:
        recorder.start()
    except AudioUnavailable as exc:
        print(f"打不开麦克风：{exc}")
        return 1

    print("  [1/2] 请保持安静…")
    quiet_levels = []
    started = _time.monotonic()
    while _time.monotonic() - started < 2.0:
        _time.sleep(0.1)
        quiet_levels.append(recorder.level())
    floor = sorted(quiet_levels)[len(quiet_levels) // 5] if quiet_levels else 0.0

    print("  [2/2] 现在请正常说一句话（随便说什么）…")
    voice_levels = []
    started = _time.monotonic()
    while _time.monotonic() - started < seconds:
        _time.sleep(0.1)
        voice_levels.append(recorder.level())
    recorder.stop()

    peak = max(voice_levels) if voice_levels else 0.0
    # 说话时的峰值里，取第 80 百分位当作"稳定说话音量"，避开偶发爆音
    ordered = sorted(voice_levels)
    typical = ordered[int(len(ordered) * 0.8)] if ordered else 0.0

    print(f"\n  环境噪声底 ≈ {floor:.4f}")
    print(f"  你的说话音量：峰值 {peak:.4f}　常态 ≈ {typical:.4f}")

    if peak < 0.02:
        print("\n  [!] 几乎没测到声音 —— 麦克风可能没在工作，或说话时离麦太远。")
        print("      先在 Windows 声音设置里确认输入设备与电平条有反应。")
        return 1
    if typical < floor * 2.0:
        print("\n  [!] 第二段里没有明显高于环境噪声的声音 —— 可能你刚才没说话，")
        print("      或者麦克风根本没收到你的人声。请重跑一次并在提示后正常说一句话。")
        return 1

    best = None
    for name in ("quiet", "normal", "noisy"):
        cfg = config_for(name)
        trigger = min(cfg.max_speech_level, max(cfg.speech_level, floor * cfg.noise_margin))
        margin = typical / trigger if trigger else 0
        verdict = "✅ 会触发" if margin >= 1.5 else ("⚠ 勉强" if margin >= 1.0 else "❌ 不触发")
        print(f"  {name:<7} 触发值 {trigger:.3f}　你的音量是它的 {margin:.1f} 倍　{verdict}")
        if margin >= 1.5 and best is None:
            best = name

    print()
    if best is None:
        print("  三个档位都不够灵敏（你的音量贴着阈值）。可以：")
        print("    · 麦克风靠近一点 / 在系统里把输入音量调大")
        print("    · 或者用下面这行临时把阈值降到 0.02（不改配置文件）：")
        print("        python tools\\check_hands_free.py --seconds 8 --threshold 0.02")
        return 1
    print(f"  建议：桌宠右键 →「免提灵敏度」→ 选「{best}」")
    print("  （档位会记住，下次打开桌宠继续生效）")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    from pet.audio import AudioUnavailable, LevelMeter, Player, Recorder, list_devices
    from pet.hands_free import (
        SENSITIVITY_LABELS,
        SENSITIVITY_PRESETS,
        HandsFreeListener,
        config_for,
    )

    parser = argparse.ArgumentParser(description="用真麦克风验证免提监听")
    parser.add_argument("--seconds", type=float, default=3.0, help="录音时长")
    parser.add_argument("--play", action="store_true", help="同时用扬声器放测试音")
    parser.add_argument("--speech", action="store_true", help="放「像说话」的脉冲音而不是纯音")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖基础阈值")
    parser.add_argument(
        "--sensitivity",
        default="normal",
        choices=sorted(SENSITIVITY_PRESETS),
        help="免提灵敏度档位（与桌宠托盘菜单一致）",
    )
    parser.add_argument("--trace", action="store_true", help="连电平数值一起打印，便于判断噪声结构")
    parser.add_argument("--calibrate", action="store_true", help="校准模式：测环境噪声 + 你的说话音量，推荐档位")
    parser.add_argument("--device", default=None, help="输入设备名的一部分（默认系统的默认设备）")
    args = parser.parse_args(argv)

    if args.calibrate:
        from pet.hands_free import config_for

        return calibrate(config_for(args.sensitivity))

    devices = list_devices()
    if not devices:
        print("没有找到任何音频设备 —— sounddevice/PortAudio 不可用。")
        print("（桌宠会提示「麦克风不可用」，免提模式自然也不会工作。）")
        return 1

    config = config_for(args.sensitivity)
    if args.threshold is not None:
        config.speech_level = args.threshold
    print(f"灵敏度档位：{args.sensitivity} —— {SENSITIVITY_LABELS.get(args.sensitivity, '')}")
    print(
        f"基础阈值 {config.speech_level:.3f}，实际触发值会自动高于实测噪声 {config.noise_margin:.0f} 倍"
        f"（上限 {config.max_speech_level:.2f}）"
    )
    print(f"静音判定 {config.silence_s:.1f}s／最短有效 {config.min_utterance_s:.1f}s")
    print(f"音频设备 {len(devices)} 个；使用默认输入设备")

    recorder = Recorder()
    player = Player()
    try:
        recorder.start()
    except AudioUnavailable as exc:
        print(f"打不开麦克风：{exc}")
        return 1

    if args.play:
        payload = make_speech_like() if args.speech else make_tone()
        print(f"播放 {'脉冲音' if args.speech else '纯音'} {len(payload) / 32000:.1f}s …")
        player.enqueue(payload)

    listener = HandsFreeListener(config)
    listener.enable(now=time.monotonic())
    meter = LevelMeter()

    print("\n电平曲线（每 100ms 一格；# = 达到当前触发值，+ = 迟滞区间，. = 安静）：")
    started = time.monotonic()
    levels: List[float] = []
    events: List[str] = []
    line = ""
    while time.monotonic() - started < args.seconds:
        time.sleep(0.1)
        now = time.monotonic()
        level = recorder.level()
        levels.append(level)
        event = listener.tick(level, now)
        if event:
            events.append(f"{now - started:5.2f}s {event}")
        trigger = listener.trigger_level()
        line += "#" if level >= trigger else ("+" if level >= listener.release_level() else ".")
        if args.trace:
            print(f"    {now - started:5.2f}s  {level:.4f}  触发={trigger:.4f} 噪声={listener.noise_floor():.4f}")
        if len(line) == 60:
            print("  " + line)
            line = ""
    if line:
        print("  " + line)

    player.stop()
    recorder.stop()

    peak = max(levels) if levels else 0.0
    average = sum(levels) / len(levels) if levels else 0.0
    floor = listener.noise_floor()
    print(
        f"\n最高电平 {peak:.4f}／平均 {average:.4f}／实测噪声底 {floor:.4f}"
        f"／当前触发值 {listener.trigger_level():.4f}"
    )
    if events:
        print("状态机判定：")
        for event in events:
            print("  " + event)
    else:
        print("状态机判定：没有任何事件 —— 这段音频在它看来一直是安静。")

    sends = sum(1 for event in events if "send" in event)
    if args.play:
        if sends:
            print("\n[OK] 麦克风听得到声音、状态机能断句并发出语音 —— 免提链路是通的。")
            return 0
        if peak < listener.trigger_level():
            print("\n[!] 播放的声音没被麦克风听到（电平低于触发值）。")
            print("    麦克风离扬声器太远、被静音、或系统把播放设备与录音设备分开了。")
        else:
            print("\n[!] 听到了但没断句：把 --seconds 调长一点，或降低 --sensitivity 档位。")
        return 1

    # 没放声音：任何 send 都意味着"在没人刻意说话的时候也达到了说话级别"。
    # 注意这不能一概而论 —— 环境里真有人说话/有音乐时，它本来就该触发。
    if sends:
        print(
            f"\n[?] 有 {sends} 次达到说话级别（峰值 {peak:.4f}）—— 如果当时确实没人在说话，"
            f"说明这一档对当前环境太灵敏。"
        )
        print("    建议：桌宠右键 →「免提灵敏度」→ 抗噪；或换个更安静的麦克风/位置。")
        print("    想看得更细：加 --trace 打印每次采样的电平与触发值。")
        return 0
    print("\n[OK] 没有达到说话级别 —— 这一档在这台机器上不会自己触发。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
