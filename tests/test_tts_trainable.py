"""Trainable-TTS tests: the GPT-SoVITS backend and the dataset preparation tool.

These cover the two things that silently ruin a fine-tune:

* the HTTP contract with GPT-SoVITS' ``api_v2`` server — a wrong field name means the
  server returns a 500 and the user sees "TTS 挂了" with no clue why;
* dataset preparation — training on silent or mislabelled slices produces a voice that
  sounds broken, and the failure only shows up after an hour of GPU time.
"""

from __future__ import annotations

import json
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))


def _pcm(seconds: float, *, rate: int = 16000, silence: bool = False) -> bytes:
    """Raw 16-bit mono PCM: a loud tone, or silence."""
    import math

    frames = bytearray()
    for index in range(int(rate * seconds)):
        value = 0 if silence else int(9000 * math.sin(2 * math.pi * 220 * index / rate))
        frames += struct.pack("<h", value)
    return bytes(frames)


def _wav(seconds: float = 1.0, rate: int = 16000, silence: bool = False) -> bytes:
    """One complete WAV file."""
    from speech.audio import pcm_to_wav

    return pcm_to_wav(_pcm(seconds, rate=rate, silence=silence), sample_rate=rate, channels=1)


def _wav_from(segments: List[tuple]) -> bytes:
    """One WAV built from ``[(seconds, is_silence), ...]``.

    Concatenating WAV *files* would not work: a decoder reads the first RIFF chunk and
    ignores everything after it, so the test would silently analyse only the first
    segment (that mistake hid a slicing bug for a while).
    """
    from speech.audio import pcm_to_wav

    pcm = b"".join(_pcm(seconds, silence=silent) for seconds, silent in segments)
    return pcm_to_wav(pcm, sample_rate=16000, channels=1)


class StubGPTSoVITS(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    received: List[Dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:  # noqa: D102
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        StubGPTSoVITS.received.append(payload)
        body = _wav(0.5)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_gptsovits():
    StubGPTSoVITS.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubGPTSoVITS)
    # 处理器线程默认不是 daemon：httpx 保持长连接时，卡在 readline() 的线程会在退出时
    # 把解释器钉住（测试全过完了、进程却不返回）。
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", StubGPTSoVITS.received
    server.shutdown()
    server.server_close()


def test_gpt_sovits_posts_the_documented_payload(stub_gptsovits, tmp_path) -> None:
    """Field names matter: api_v2 rejects anything it does not recognise."""
    import asyncio

    from core.types import VoiceSpec
    from plugs.gpt_sovits_tts import GPTSoVITSTTSBackend
    from speech.tts import GPTSoVITSTTSBackend as ReExported

    assert ReExported is GPTSoVITSTTSBackend, "speech.tts 应再导出同一个类，供 build_tts 使用"

    base_url, received = stub_gptsovits
    reference = tmp_path / "ref.wav"
    reference.write_bytes(_wav(1.0))
    backend = GPTSoVITSTTSBackend(base_url=base_url, text_lang="zh", prompt_lang="zh")

    async def collect() -> bytes:
        chunks = [
            chunk
            async for chunk in backend.synthesize(
                "你好呀", VoiceSpec(reference_audio=str(reference), reference_text="你好呀", speed=1.05)
            )
        ]
        return b"".join(chunk.data for chunk in chunks)

    audio = asyncio.run(collect())
    assert audio[:4] == b"RIFF" and len(audio) > 1000
    assert received, "服务端没有收到请求"
    sent = received[0]
    for field in ("text", "text_lang", "ref_audio_path", "prompt_text", "prompt_lang", "media_type"):
        assert field in sent, f"请求缺少字段 {field}"
    assert sent["text"] == "你好呀"
    assert sent["ref_audio_path"] == str(reference)
    assert sent["prompt_text"] == "你好呀"
    assert sent["speed_factor"] == pytest.approx(1.05)


def test_gpt_sovits_requires_a_reference_clip(tmp_path) -> None:
    """Without a reference the server cannot clone at all — fail early and clearly."""
    import asyncio

    from core.errors import AudioError
    from core.types import VoiceSpec
    from plugs.gpt_sovits_tts import GPTSoVITSTTSBackend

    backend = GPTSoVITSTTSBackend(base_url="http://127.0.0.1:1")

    async def collect() -> bytes:
        return b"".join(
            [chunk.data async for chunk in backend.synthesize("你好", VoiceSpec(voice_id="x"))]
        )

    with pytest.raises(AudioError) as excinfo:
        asyncio.run(collect())
    assert "参考音频" in excinfo.value.message


def test_gpt_sovits_health_when_server_is_down() -> None:
    import asyncio

    from plugs.gpt_sovits_tts import GPTSoVITSTTSBackend

    backend = GPTSoVITSTTSBackend(base_url="http://127.0.0.1:1")
    info = asyncio.run(backend.health())
    assert info["ok"] is False
    assert "推理服务未运行" in info["error"]
    assert "train-tts.ps1" in info["hint"]


# ---------------------------------------------------------------------------------------
# dataset preparation
# ---------------------------------------------------------------------------------------


def test_dataset_tool_slices_on_silence_and_writes_a_list(tmp_path) -> None:
    """Four short bursts separated by silence must become separate training slices."""
    import tts_dataset

    silence = 0.6
    raw = tmp_path / "raw"
    raw.mkdir()
    # 4 utterances of 2.5s separated by 0.6s of silence (one file, ~11.8s)
    (raw / "take1.wav").write_bytes(
        _wav_from([(2.5, False), (silence, True), (2.5, False), (silence, True),
                   (2.5, False), (silence, True), (2.5, False)])
    )

    out = tmp_path / "dataset"
    report = tts_dataset.build(
        inputs=[raw / "take1.wav"],
        out_dir=out,
        speaker="tester",
        language="zh",
        min_s=1.5,
        max_s=6.0,
        normalize=True,
        labels=None,
        transcriber=None,  # 不跑 ASR：测试不去下载 1.6GB 模型
        dry_run=False,
    )
    assert len(report.slices) >= 3, f"应该在静音处切成多段，实际 {len(report.slices)}"
    assert report.duration() > 6.0
    for item in report.slices:
        assert item.path.exists() and item.path.stat().st_size > 1000
        assert 1.0 <= item.duration_s <= 7.0

    list_path, report_md = tts_dataset.write_outputs(
        report, out_dir=out, speaker="tester", language="zh"
    )
    lines = [line for line in list_path.read_text(encoding="utf-8").splitlines() if line]
    # 没有文本的切片不会写进 list（GPT-SoVITS 需要文本）
    assert lines == []
    assert "数据集报告" in report_md.read_text(encoding="utf-8")


def test_dataset_tool_uses_existing_labels_and_skips_asr(tmp_path) -> None:
    import tts_dataset

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "clip.wav").write_bytes(_wav_from([(1.2, False), (0.7, True), (1.2, False)]))
    labels = tmp_path / "labels.tsv"
    labels.write_text("clip\t你好，这是第一句。\n", encoding="utf-8")

    out = tmp_path / "dataset"
    report = tts_dataset.build(
        inputs=sorted(raw.glob("*.wav")),
        out_dir=out,
        speaker="tester",
        language="zh",
        min_s=1.0,
        max_s=5.0,
        normalize=True,
        labels=tts_dataset.load_text_file(labels),
        transcriber=None,
        dry_run=False,
    )
    assert report.slices, "应该至少切出一段"
    assert all(item.text for item in report.slices), "已有标注应该被套用到每个切片"
    list_path, _ = tts_dataset.write_outputs(report, out_dir=out, speaker="tester", language="zh")
    lines = [line for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "dataset.list 不应为空"
    for line in lines:
        parts = line.split("|")
        assert len(parts) == 4, f"每行必须是 路径|说话人|语言|文本，实际 {len(parts)} 段：{line}"
        assert parts[1] == "tester" and parts[2] == "zh"
        assert parts[3].strip(), "第 4 列必须是文本"
        assert Path(parts[0]).exists(), "第 1 列必须是切片文件路径"


def test_dataset_tool_drops_silence_and_reports_why(tmp_path) -> None:
    """A silent file must be dropped with a reason instead of becoming training data."""
    import tts_dataset

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "silent.wav").write_bytes(_wav(4.0, silence=True))
    (raw / "tiny.wav").write_bytes(_wav(0.4))
    (raw / "notes.txt").write_text("不是音频", encoding="utf-8")

    report = tts_dataset.build(
        inputs=tts_dataset.collect_inputs(raw),
        out_dir=tmp_path / "out",
        speaker="t",
        language="zh",
        min_s=1.0,
        max_s=6.0,
        normalize=True,
        labels=None,
        transcriber=None,
        dry_run=True,
    )
    reasons = " ".join(reason for _name, reason in report.dropped + report.skipped)
    assert "几乎无声" in reasons or "太短" in reasons
    assert not (tmp_path / "out" / "sliced").exists(), "dry-run 不应写文件"


def test_check_gptsovits_reports_missing_repo(tmp_path) -> None:
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_gptsovits.py"), "--repo", str(tmp_path / "nope")],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "install-gptsovits.ps1" in completed.stdout
