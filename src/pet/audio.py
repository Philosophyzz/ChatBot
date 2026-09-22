"""Microphone capture and speaker playback for the desktop pet.

Recording happens here, in Python, rather than in a browser/WebView:

* no microphone-permission dance (WebView2 needs a host-side permission callback that is
  awkward to reach from Python),
* the clip is already 16 kHz mono PCM, which is exactly what the ASR endpoint wants — no
  browser codec round-trip,
* push-to-talk becomes a plain key/mouse event.

Playback decodes with ``soundfile`` (libsndfile handles both WAV from a local model and
MP3 from edge-tts) and writes to the default output device through ``sounddevice``.
"""

from __future__ import annotations

import io
import threading
import time
from collections import deque
from typing import Callable, Deque, List, Optional, Tuple

from core.logging import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16000
#: How long a block stays in the level meter's window.
LEVEL_WINDOW_S = 0.35


class AudioUnavailable(RuntimeError):
    """Raised when PortAudio/sounddevice cannot be used, with a readable reason."""


def _import_sounddevice():
    try:
        import sounddevice as sd  # type: ignore

        return sd
    except Exception as exc:  # noqa: BLE001
        raise AudioUnavailable(f"没有可用的音频设备支持（sounddevice/PortAudio 不可用）：{exc}") from exc


def block_peak(pcm: bytes) -> float:
    """Peak amplitude of one int16 block, normalised to 0..1."""
    if not pcm:
        return 0.0
    samples = memoryview(pcm).cast("h")
    return max((abs(value) for value in samples), default=0) / 32768.0


class LevelMeter:
    """Recent input loudness: the peak over the last ``window_s`` seconds.

    ``Recorder.peak`` is a *session* maximum — it only ever grows, so it can answer "has the
    user said anything at all?" but never "have they stopped?". Hands-free mode asks the
    second question every 250 ms, so it needs the opposite behaviour: remember a short window
    and forget the rest.

    A stalled stream (mic unplugged, device sleeping) yields 0.0 after one window, which the
    caller treats as silence — the right failure mode for a listener that must not hang.
    """

    def __init__(self, window_s: float = LEVEL_WINDOW_S) -> None:
        self.window_s = window_s
        self._samples: Deque[Tuple[float, float]] = deque()
        self._lock = threading.Lock()

    def feed(self, pcm: bytes, now: Optional[float] = None) -> float:
        """Record one captured block; returns its peak (0..1)."""
        peak = block_peak(pcm)
        stamp = time.monotonic() if now is None else now
        with self._lock:
            self._samples.append((stamp, peak))
            self._trim(stamp)
        return peak

    def level(self, now: Optional[float] = None) -> float:
        """Loudest block inside the window right now."""
        stamp = time.monotonic() if now is None else now
        with self._lock:
            self._trim(stamp)
            return max((peak for _stamp, peak in self._samples), default=0.0)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()


class Recorder:
    """Push-to-talk recorder: keeps the stream open, returns a WAV on stop."""

    def __init__(self, *, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._frames: Deque[Tuple[bytes, int]] = deque()
        self._stream = None
        self._lock = threading.Lock()
        self.recording = False
        self.peak = 0.0
        self.meter = LevelMeter()

    def start(self) -> None:
        sd = _import_sounddevice()
        with self._lock:
            if self.recording:
                return
            self._frames = deque()
            self.peak = 0.0
            self.meter.reset()

            def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
                if status:  # overflow etc. — keep going, the clip is still usable
                    log.debug("audio input status", extra={"status": str(status)})
                pcm = bytes(indata)
                with self._lock:
                    self._frames.append((pcm, len(pcm) // 2))
                peak = self.meter.feed(pcm)
                self.peak = max(self.peak, peak)

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=1600,  # 100 ms
                callback=callback,
            )
            self._stream.start()
            self.recording = True

    def stop(self) -> bytes:
        """Stop recording and return WAV bytes (possibly empty if nothing was captured)."""
        from speech.audio import pcm_to_wav

        with self._lock:
            stream, self._stream = self._stream, None
            self.recording = False
            frames, self._frames = self._frames, deque()
            self.meter.reset()
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("closing input stream failed", extra={"error": str(exc)})
        pcm = b"".join(block for block, _samples in frames)
        if not pcm:
            return b""
        return pcm_to_wav(pcm, sample_rate=self.sample_rate, channels=1)

    def level(self) -> float:
        """Loudness right now (see :class:`LevelMeter`), not the session maximum."""
        return self.meter.level()

    def buffered_seconds(self) -> float:
        """How much audio is currently held (diagnostics, and the trim test)."""
        with self._lock:
            return sum(samples for _block, samples in self._frames) / float(self.sample_rate)

    def retain_last(self, seconds: float) -> None:
        """Drop everything but the last ``seconds`` of audio.

        Hands-free mode records continuously while it waits for the user to start talking, so
        without this the buffer grows at 32 KB/s for as long as the pet sits idle (an hour of
        waiting would be ~115 MB), and the clip handed to the recogniser would begin with
        minutes of silence. Called every tick while the listener is merely *armed*: what gets
        sent is then the utterance plus a short pre-roll.
        """
        if seconds <= 0:
            return
        budget = int(seconds * self.sample_rate)
        with self._lock:
            total = sum(samples for _block, samples in self._frames)
            while self._frames and total - self._frames[0][1] >= budget:
                total -= self._frames.popleft()[1]


class Player:
    """Sequential playback of decoded audio chunks, interruptible."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._queue: List[bytes] = []
        self.speaking = False
        self.on_state: Optional[Callable[[bool], None]] = None

    def enqueue(self, audio: bytes) -> None:
        """Queue one audio chunk (WAV/MP3/FLAC bytes) and start playing if idle."""
        if not audio:
            return
        with self._lock:
            self._queue.append(audio)
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="pet-player", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._queue.clear()
        try:
            _import_sounddevice().stop()
        except Exception:  # noqa: BLE001
            pass

    def wait(self, timeout_s: float = 30.0) -> None:
        thread = self._thread
        if thread:
            thread.join(timeout=timeout_s)

    def _decode(self, audio: bytes):
        import numpy as np
        import soundfile as sf

        data, rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.mean(axis=1)), int(rate)

    def _run(self) -> None:
        sd = _import_sounddevice()
        self._set_speaking(True)
        try:
            while not self._stop.is_set():
                with self._lock:
                    if not self._queue:
                        break
                    chunk = self._queue.pop(0)
                try:
                    samples, rate = self._decode(chunk)
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not decode audio chunk", extra={"error": str(exc)})
                    continue
                if self._stop.is_set():
                    break
                try:
                    sd.play(samples, rate)
                    sd.wait()
                except Exception as exc:  # noqa: BLE001
                    log.warning("playback failed", extra={"error": str(exc)})
                    break
        finally:
            self._set_speaking(False)

    def _set_speaking(self, value: bool) -> None:
        self.speaking = value
        callback = self.on_state
        if callback:
            try:
                callback(value)
            except Exception:  # noqa: BLE001
                pass


def list_devices() -> List[str]:
    """Human-readable audio device list, for troubleshooting."""
    try:
        sd = _import_sounddevice()
        out = []
        for index, device in enumerate(sd.query_devices()):
            out.append(
                f"[{index}] {device['name']} in={device['max_input_channels']} out={device['max_output_channels']}"
            )
        return out
    except AudioUnavailable:
        return []
