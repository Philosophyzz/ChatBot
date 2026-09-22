"""Create a placeholder reference clip for IndexTTS2 voice cloning.

IndexTTS2 is a *zero-shot cloning* model: it has no built-in voices, so every call needs
5–15 seconds of reference speech. Without one the first spoken sentence would fail with
"缺少参考音频".

This helper produces a neutral Chinese paragraph as a WAV so the feature works out of
the box. Two things worth knowing:

* The timbre it clones is whatever voice was used to make this file.
* Replace the file with your own recording (same path, 5–15s, mono WAV, quiet room) to
  get *your* voice or any voice you have the right to use. Cloning a commercial voice
  and redistributing it is not okay — keep it for personal use, or use your own.

Offline-first: SAPI (Windows' built-in synthesis) is used when available so this works
with no network at all. ``--engine edge`` borrows a nicer online voice instead.

Usage::

    python tools/make_reference.py --out models/voices/default_sweet.wav
    python tools/make_reference.py --out models/voices/default_sweet.wav --engine edge
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEXT = (
    "你好呀，我是你的语音伙伴。今天过得怎么样呢？"
    "不管你想聊点什么，我都在这里听着。"
    "要是累了，就先歇一会儿，慢慢来也没关系。"
)


def via_sapi(text: str, out: Path) -> bool:
    """Windows SAPI through PowerShell — always available, no network, robotic."""
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.SelectVoice('Microsoft Huihui Desktop'); "
        f"$s.SetOutputToWaveFile('{out}'); "
        f"$s.Speak('{text}'); "
        "$s.Dispose()"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(f"  SAPI 失败：{completed.stderr.strip()[:200]}")
        return False
    return out.exists() and out.stat().st_size > 1000


async def via_edge(text: str, out: Path, voice: str) -> bool:
    try:
        import edge_tts  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"  未安装 edge-tts：{exc}")
        return False
    with tempfile.TemporaryDirectory(prefix="chatbot-ref-") as tmp:
        mp3 = Path(tmp) / "ref.mp3"
        communicate = edge_tts.Communicate(text, voice, rate="-8%")
        with open(mp3, "wb") as handle:
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    handle.write(chunk["data"])
        if not mp3.exists() or mp3.stat().st_size < 1000:
            print("  edge-tts 没有返回音频")
            return False
        # IndexTTS2 reads the reference with librosa/soundfile; WAV is the safe format.
        try:
            import soundfile as sf  # type: ignore

            data, rate = sf.read(str(mp3))
            sf.write(str(out), data, rate)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  soundfile 不可用（{type(exc).__name__}），尝试 ffmpeg")
        if not shutil.which("ffmpeg"):
            print("  也没有 ffmpeg：装一个（pip install soundfile 或 winget install Gyan.FFmpeg）")
            return False
        completed = subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3), "-ar", "22050", "-ac", "1", str(out)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            print(f"  ffmpeg 失败：{completed.stderr.strip()[:160]}")
            return False
    return out.exists() and out.stat().st_size > 1000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 IndexTTS2 的参考音色")
    parser.add_argument("--out", help="输出 WAV 路径（单文件模式）")
    parser.add_argument("--personas", help="按 config/personas.yaml 里每个 p 的音色各生成一个参考音频")
    parser.add_argument("--out-dir", default="models/voices", help="--personas 模式的输出目录")
    parser.add_argument("--engine", choices=("sapi", "edge"), default="sapi")
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural", help="单文件模式的 edge 音色")
    args = parser.parse_args(argv)

    if args.personas:
        return generate_for_personas(args)

    if not args.out:
        parser.error("需要 --out 或 --personas")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"生成参考音频 -> {out}（engine={args.engine}）")

    ok = asyncio.run(via_edge(TEXT, out, args.voice)) if args.engine == "edge" else via_sapi(TEXT, out)
    if not ok:
        print("生成失败。请手动放一段 5~15 秒的中文 WAV 到该路径。")
        return 1
    print(f"完成：{out.stat().st_size / 1024:.0f} KB")
    print("提示：换成你自己的录音会得到更像你的音色；请勿用他人享有权利的商业音色做传播用途。")
    return 0


def generate_for_personas(args: argparse.Namespace) -> int:
    """One reference clip per persona, using that persona's own online voice.

    IndexTTS2 clones a *timbre* from a single clip, so seven personas that share one
    reference file would all sound like the same person. Deriving each reference from
    the persona's existing voice keeps the characters audibly distinct after switching
    from the online engine to the local one.
    """
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
        print(f"需要 pyyaml：{exc}")
        return 2

    source = Path(args.personas)
    if not source.exists():
        print(f"找不到 {source}")
        return 2
    personas = (yaml.safe_load(source.read_text(encoding="utf-8")) or {}).get("personas") or {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made: list[tuple[str, str, str]] = []
    for persona_id, spec in personas.items():
        voice = (spec or {}).get("voice") or {}
        voice_id = str(voice.get("voice_id") or "").strip()
        engine = args.engine
        if engine == "edge" and not voice_id:
            print(f"  跳过 {persona_id}：没有 voice_id")
            continue
        target = out_dir / f"{persona_id}.wav"
        if target.exists() and target.stat().st_size > 1000:
            print(f"  已存在 {target.name}")
            made.append((persona_id, str(target).replace("\\", "/"), voice_id))
            continue
        print(f"  {persona_id} <- {voice_id or 'SAPI'}")
        ok = (
            asyncio.run(via_edge(TEXT, target, voice_id))
            if engine == "edge"
            else via_sapi(TEXT, target)
        )
        if ok:
            made.append((persona_id, str(target).replace("\\", "/"), voice_id))
        else:
            print(f"    {persona_id} 生成失败")

    if not made:
        print("一个都没生成。用 --engine edge（需要联网）或手动放录音。")
        return 1

    print("\n把下面的 voice.reference_audio 填进 config/personas.yaml：")
    for persona_id, path, voice_id in made:
        print(f"  {persona_id}: {path}    # 源音色 {voice_id or 'SAPI'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
