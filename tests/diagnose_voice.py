"""Isolate which TTS layer is broken when the voice sounds wrong or is silent.

The symptom this was written for: ``/api/voice/tts`` returned a 200 with a 46-byte
header-only WAV (zero audio frames) while the diagnostic endpoint synthesized fine in
the *same process*. The cause was the router's selection order — see
``test_tts_router_orders_by_quality_not_offline`` — but finding it required being able
to tell "the backend is broken" apart from "the router picked the wrong backend".

Run it whenever voice output is silent, robotic, or slower than expected::

    venvs\\main\\Scripts\\python.exe tests\\diagnose_voice.py

With the server already running, the same information (plus per-backend tracebacks) is
available over HTTP::

    curl "http://127.0.0.1:8077/api/voice/tts/diagnose"

The probe answers, in one pass:

  A  is the online backend reachable at all?
  B  does plugin registration change anything?
  C  does it behave differently inside a running event loop (as in the server)?
  D  what does the *router* actually pick, with the persona's real voice settings?
  E  is a specific voice parameter (pitch / emotion / speed) the trigger?
  F  is the provider rate-limiting us?

It is not a unit test — it talks to the network — so it lives outside ``test_*.py`` and
is never collected by pytest.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import load_config  # noqa: E402
from core.types import VoiceSpec  # noqa: E402
from speech.tts import EdgeTTSBackend  # noqa: E402

TEXT = "你好呀，我是小甜。"


def outcome(label: str, data: bytes, backend: str = "edge") -> None:
    print(f"  {label:52} OK  bytes={len(data):<7} via={backend}")


async def try_edge(label: str, backend: EdgeTTSBackend, voice: VoiceSpec) -> bool:
    try:
        chunks = [c async for c in backend.synthesize(TEXT, voice)]
        data = b"".join(c.data for c in chunks)
        if not data:
            print(f"  {label:52} FAIL empty audio")
            return False
        outcome(label, data)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:52} FAIL {type(exc).__name__}: {str(exc)[:70]}")
        return False


async def main() -> None:
    print("A) 裸后端（基线）")
    await try_edge("A: plain EdgeTTSBackend", EdgeTTSBackend(), VoiceSpec())

    print("\nB) 导入 plugs 之后（注册所有插件）")
    sys.path.insert(0, str(ROOT / "src"))
    import plugs  # noqa: F401,E402

    await try_edge("B: after importing plugs", EdgeTTSBackend(), VoiceSpec())

    print("\nC) 在运行中的事件循环里以任务形式调用")
    results: list[bool] = []

    async def task() -> None:
        results.append(await try_edge("C: inside asyncio task", EdgeTTSBackend(), VoiceSpec()))

    await asyncio.gather(task())

    print("\nD) 走应用真实的 TTSRouter + 人设 VoiceSpec")
    config = load_config(ROOT)
    from persona.manager import PersonaManager
    from speech import build_tts

    personas = PersonaManager(config)
    persona = personas.get("sweet_companion")
    tts = build_tts(config)
    print(f"  router preference = {getattr(tts, 'preference', None)}")
    print(f"  persona voice     = {persona.voice.to_public()}")
    try:
        chunks = [c async for c in tts.synthesize(TEXT, persona.voice, stream=False)]
        data = b"".join(c.data for c in chunks)
        outcome("D: via TTSRouter + persona voice", data, str(getattr(tts, "_last_used", "?")))
    except Exception as exc:  # noqa: BLE001
        print(f"  D: via TTSRouter + persona voice                    FAIL {type(exc).__name__}: {str(exc)[:70]}")

    print("\nE) 逐项复现 API 的 VoiceSpec 构造")
    for label, voice in (
        ("E1: backend='edge' + persona 音色", VoiceSpec(backend="edge", voice_id=persona.voice.voice_id,
                                                        speed=persona.voice.speed, emotion=persona.voice.emotion,
                                                        emotion_alpha=persona.voice.emotion_alpha,
                                                        pitch_shift=persona.voice.pitch_shift)),
        ("E2: 去掉 pitch_shift", VoiceSpec(backend="edge", voice_id=persona.voice.voice_id,
                                           speed=persona.voice.speed, emotion=persona.voice.emotion,
                                           emotion_alpha=persona.voice.emotion_alpha)),
        ("E3: 只留 backend", VoiceSpec(backend="edge")),
    ):
        await try_edge(label, EdgeTTSBackend(), voice)

    print("\nF) 连续 5 次直连，确认是否频率限制")
    good = 0
    for index in range(5):
        good += await try_edge(f"F{index + 1}: repeat", EdgeTTSBackend(), VoiceSpec())
        await asyncio.sleep(1.2)
    print(f"  -> {good}/5 成功")

    await tts.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
