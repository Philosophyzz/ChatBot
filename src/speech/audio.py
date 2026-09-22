"""Audio decoding, encoding and probing.

The browser is the microphone, so the server receives whatever ``MediaRecorder``
produced — usually ``audio/webm;codecs=opus``, sometimes ``audio/ogg`` or
``audio/mp4``. Speech models want 16 kHz mono PCM, so something has to convert.

Resolution order, cheapest first:

1. **Already WAV PCM** → parsed in-process, zero dependencies.
2. **ffmpeg** (if present) → handles every browser codec, also used for
   normalisation to 16 kHz mono.
3. **PyAV / soundfile** if an optional install provided them.
4. **A precise error** naming the exact command to install, rather than a
   mysterious failure deep inside the speech model.

The frontend also implements a Web Audio path that uploads 16 kHz WAV directly,
so a machine without ffmpeg still gets full voice input for the common case; this
module is the safety net for browsers and formats that need transcoding.
"""

from __future__ import annotations

import array
import math
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.errors import AudioError
from core.logging import get_logger
from core.types import AudioClip

log = get_logger(__name__)

TARGET_SAMPLE_RATE = 16000

#: Container/codec hints we can identify from magic bytes.
_MAGIC = {
    b"RIFF": ("wav", "audio/wav"),
    b"\x1aE\xdf\xa3": ("webm", "audio/webm"),
    b"OggS": ("ogg", "audio/ogg"),
    b"fLaC": ("flac", "audio/flac"),
    b"ID3": ("mp3", "audio/mpeg"),
    b"\xff\xfb": ("mp3", "audio/mpeg"),
    b"\xff\xf3": ("mp3", "audio/mpeg"),
    b"#!AMR": ("amr", "audio/amr"),
}


def sniff_format(data: bytes) -> Tuple[str, str]:
    """Best-effort ``(extension, mime)`` from the leading bytes."""
    # ISO-BMFF (mp4/m4a/mov) puts the box type at offset 4, so it must be checked
    # before the prefix table — otherwise "....ftyp" falls through to "bin" and the
    # decoder reports a confusing "cannot decode" error for a valid recording.
    # Note ``>= 12``: a minimal ftyp box is exactly 12 bytes, so a strict ``>``
    # comparison would reject it.
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"M4A ", b"M4B ", b"mp42", b"mp41", b"isom", b"iso2", b"dash"):
            return "m4a", "audio/mp4"
        return "mp4", "audio/mp4"
    for magic, (ext, mime) in _MAGIC.items():
        if data.startswith(magic):
            return ext, mime
    return "bin", "application/octet-stream"


def ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


def has_ffmpeg() -> bool:
    return ffmpeg_path() is not None


def probe_wav(data: bytes) -> Optional[Dict[str, Any]]:
    """Read WAV header fields without decoding the samples."""
    if len(data) < 44 or not data.startswith(b"RIFF"):
        return None
    try:
        with wave.open(_BytesReader(data), "rb") as handle:  # type: ignore[arg-type]
            return {
                "channels": handle.getnchannels(),
                "sample_rate": handle.getframerate(),
                "sampwidth": handle.getsampwidth(),
                "frames": handle.getnframes(),
                "duration_s": handle.getnframes() / max(1, handle.getframerate()),
            }
    except Exception:  # noqa: BLE001
        return None


class _BytesReader:
    """Minimal seekable file-like object so ``wave`` can read from a bytes buffer."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk = self._data[self._pos :]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        else:
            self._pos = len(self._data) + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def close(self) -> None:  # pragma: no cover - symmetry
        return None


def decode_to_pcm(data: bytes, *, sample_rate: int = TARGET_SAMPLE_RATE) -> Tuple[bytes, int, int]:
    """Decode arbitrary audio bytes to mono 16-bit PCM.

    Returns ``(pcm_bytes, sample_rate, channels)``.
    """
    if not data:
        raise AudioError("音频内容为空")
    fmt, mime = sniff_format(data)

    if fmt == "wav":
        info = probe_wav(data)
        if info and info["sampwidth"] == 2:
            if info["channels"] == 1 and info["sample_rate"] == sample_rate:
                return _read_wav_frames(data), sample_rate, 1
            pcm = _read_wav_frames(data)
            pcm, channels = _downmix(pcm, int(info["channels"]))
            if info["sample_rate"] != sample_rate:
                pcm = resample_pcm(pcm, int(info["sample_rate"]), sample_rate)
            return pcm, sample_rate, channels

    executable = ffmpeg_path()
    if executable:
        return _decode_with_ffmpeg(executable, data, sample_rate)

    try:  # optional richer decoders
        return _decode_with_pyav(data, sample_rate)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        log.debug("pyav decode failed", extra={"error": str(exc)})

    raise AudioError(
        f"无法解码 {mime} 音频。请安装 ffmpeg（winget install Gyan.FFmpeg）后重试，"
        "或改用浏览器内置录音（前端会直接上传 16kHz WAV）。",
        detail={"detected_format": fmt, "mime": mime, "bytes": len(data)},
    )


def _decode_with_ffmpeg(executable: str, data: bytes, sample_rate: int) -> Tuple[bytes, int, int]:
    with tempfile.TemporaryDirectory(prefix="chatbot-audio-") as tmpdir:
        source = Path(tmpdir) / "input.bin"
        source.write_bytes(data)
        command = [
            executable,
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(source),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, check=False, timeout=120
            )
        except FileNotFoundError as exc:  # pragma: no cover
            raise AudioError("ffmpeg 不可执行", detail={"error": str(exc)}) from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioError("音频转码超时", detail={"timeout_s": 120}) from exc
        if completed.returncode != 0 or not completed.stdout:
            stderr = (completed.stderr or b"").decode("utf-8", "replace")[:400]
            raise AudioError("音频转码失败", detail={"ffmpeg": stderr, "code": completed.returncode})
        return completed.stdout, sample_rate, 1


def _decode_with_pyav(data: bytes, sample_rate: int) -> Tuple[bytes, int, int]:
    import av  # type: ignore

    with av.open(_BytesReader(data)) as container:  # type: ignore[arg-type]
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise AudioError("音频容器中没有音频轨道")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(bytes(resampled.planes[0]))
        if not chunks:
            raise AudioError("音频解码结果为空")
        return b"".join(chunks), sample_rate, 1


def _read_wav_frames(data: bytes) -> bytes:
    with wave.open(_BytesReader(data), "rb") as handle:  # type: ignore[arg-type]
        return handle.readframes(handle.getnframes())


def _downmix(pcm: bytes, channels: int) -> Tuple[bytes, int]:
    if channels <= 1:
        return pcm, 1
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % (2 * channels))])
    frames = len(samples) // channels
    mono = array.array("h", [0]) * frames
    for index in range(frames):
        total = 0
        base = index * channels
        for channel in range(channels):
            total += samples[base + channel]
        mono[index] = int(total / channels)
    return mono.tobytes(), 1


def resample_pcm(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linear-interpolation resampler.

    Speech models are robust to this quality level and it avoids pulling in
    scipy/librosa for what is a 16 kHz conversion of a few seconds of audio.
    """
    if source_rate == target_rate or not pcm:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    source_count = len(samples)
    if source_count == 0:
        return b""
    target_count = int(source_count * target_rate / source_rate)
    ratio = source_rate / target_rate
    out = array.array("h", [0]) * target_count
    for index in range(target_count):
        position = index * ratio
        low = int(position)
        high = min(low + 1, source_count - 1)
        weight = position - low
        out[index] = int(samples[low] * (1.0 - weight) + samples[high] * weight)
    return out.tobytes()


def trim_silence(pcm: bytes, *, threshold: int = 320, padding_ms: int = 120) -> bytes:
    """Trim leading/trailing near-silence.

    Cuts transcription time noticeably for push-to-talk recordings, where the user
    typically holds the button for a moment before speaking.
    """
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    count = len(samples)
    if count == 0:
        return pcm
    start = 0
    while start < count and abs(samples[start]) < threshold:
        start += 1
    end = count - 1
    while end > start and abs(samples[end]) < threshold:
        end -= 1
    if start >= end:
        return pcm  # all silence: let the caller decide what to do
    padding = int(TARGET_SAMPLE_RATE * padding_ms / 1000)
    start = max(0, start - padding)
    end = min(count - 1, end + padding)
    return samples[start : end + 1].tobytes()


def pcm_to_wav(pcm: bytes, *, sample_rate: int = TARGET_SAMPLE_RATE, channels: int = 1) -> bytes:
    """Wrap raw PCM in a WAV container."""
    buffer = _BytesWriter()
    with wave.open(buffer, "wb") as handle:  # type: ignore[arg-type]
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


class _BytesWriter:
    """Minimal writable file-like object for ``wave``.

    ``wave`` calls ``flush()`` while patching the header sizes, so it must exist —
    without it every WAV encode raises AttributeError.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def write(self, data: bytes) -> int:
        self._buffer.extend(data)
        return len(data)

    def tell(self) -> int:
        return len(self._buffer)

    def seek(self, offset: int, whence: int = 0) -> int:
        # ``wave`` only seeks back to patch sizes it already wrote; since we keep
        # the whole buffer in memory, reporting the current length is sufficient.
        if whence == 0:
            return offset
        if whence == 1:
            return len(self._buffer) + offset
        return max(0, len(self._buffer) + offset)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def getvalue(self) -> bytes:
        return bytes(self._buffer)


def pcm_duration_s(pcm: bytes, *, sample_rate: int = TARGET_SAMPLE_RATE, channels: int = 1) -> float:
    frame_bytes = 2 * max(1, channels)
    return round(len(pcm) / frame_bytes / max(1, sample_rate), 3)


def rms(pcm: bytes) -> float:
    """Root-mean-square level, used to reject near-silent recordings early."""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    total = sum(float(sample) * sample for sample in samples)
    return math.sqrt(total / len(samples))


def normalize_clip(data: bytes, *, sample_rate: int = TARGET_SAMPLE_RATE, vad_trim: bool = True) -> AudioClip:
    """Decode + clean a browser recording into a model-ready clip."""
    pcm, rate, channels = decode_to_pcm(data, sample_rate=sample_rate)
    if vad_trim:
        pcm = trim_silence(pcm)
    wav = pcm_to_wav(pcm, sample_rate=rate, channels=channels)
    return AudioClip(data=wav, mime="audio/wav", sample_rate=rate, channels=channels, duration_s=pcm_duration_s(pcm, sample_rate=rate, channels=channels))


__all__ = [
    "AudioClip",
    "TARGET_SAMPLE_RATE",
    "decode_to_pcm",
    "ffmpeg_path",
    "has_ffmpeg",
    "normalize_clip",
    "pcm_duration_s",
    "pcm_to_wav",
    "probe_wav",
    "resample_pcm",
    "rms",
    "sniff_format",
    "trim_silence",
]
