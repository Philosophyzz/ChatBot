"""Bounded, stereo ASMR composition. No TTS, microphone, network, or GPU use."""
from pathlib import Path
import json
import re
import numpy as np

RATE = 44100
PRESETS = {'paper': '纸张沙沙', 'brush': '柔软刷拭', 'tapping': '木质轻敲', 'mixed': '随机组合'}


def parse_command(text):
    value = re.sub(r'\s+', '', text.lower()).rstrip('。！!')
    control = re.fullmatch(r'(?:请|帮我|给我)?(停止|关闭|结束|暂停|继续)asmr', value)
    if control:
        return {'action': {'暂停':'pause','继续':'resume'}.get(control[1], 'stop')}
    match = re.fullmatch(r'(?:请|帮我|给我)?(?:播放|开始|来点|听)(纸张|刷拭|柔软刷拭|轻敲|木质轻敲|随机组合)?asmr(?:(5|15|30)分钟)?', value)
    if match:
        return {'action':'start', 'preset': {'纸张':'paper','刷拭':'brush','柔软刷拭':'brush','轻敲':'tapping','木质轻敲':'tapping'}.get(match[1], 'mixed'),
                'seconds': int(match[2])*60 if match[2] else None}
    return None


def finish_audio(audio, rate=RATE, rms_db=-25, peak_db=-8, fade_s=.3):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.repeat(audio[:, None], 2, axis=1)
    if audio.ndim != 2 or audio.shape[1] != 2 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Audio must be finite non-empty stereo PCM')
    audio = audio - audio.mean(axis=0)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        raise ValueError('Silent audio')
    # One gain for both channels preserves stereo balance. No hard clipping or boost of silence.
    gain = min(10 ** (rms_db / 20) / max(rms, 1e-8), 10 ** (peak_db / 20) / peak, 4.0)
    audio = audio * gain
    fade = min(int(fade_s * rate), len(audio) // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32) ** 2
        audio[:fade] *= ramp[:, None]
        audio[-fade:] *= ramp[::-1, None]
    return np.ascontiguousarray(audio, dtype=np.float32)


class TextureLibrary:
    def __init__(self, root):
        import soundfile as sf
        self.root = Path(root).resolve()
        spec = json.loads((self.root / 'library.json').read_text(encoding='utf-8'))
        self.clips = {}
        self.sources = {}
        for row in spec['clips']:
            if not row.get('approved', False):
                continue
            path = (self.root / row['file']).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Clip outside ASMR library')
            audio, sr = sf.read(path, dtype='float32', always_2d=True)
            if sr != RATE or audio.shape[1] != 2 or len(audio) < RATE:
                raise ValueError('ASMR library requires 44.1 kHz stereo clips of at least one second')
            self.clips.setdefault(row['category'], []).append(finish_audio(audio))
            self.sources.setdefault(row['category'], set()).add(row.get('engine', 'recording'))
        if not self.clips:
            raise ValueError('No approved ASMR clips; run tools/asmr_dataset.py first')

    def blocks(self, preset, seconds, *, seed=0, block_size=4410):
        if preset not in PRESETS or not 1 <= seconds <= 3600:
            raise ValueError('Unsupported ASMR preset or duration')
        categories = list(self.clips) if preset == 'mixed' else [preset]
        if any(k not in self.clips for k in categories):
            raise ValueError('This preset has no approved clips')
        rng = np.random.default_rng(seed)
        total = int(seconds * RATE)
        overlap = RATE // 2
        pending = np.zeros((0, 2), dtype=np.float32)
        emitted = 0
        previous = None
        while emitted < total:
            category = categories[int(rng.integers(len(categories)))]
            choices = self.clips[category]
            index = int(rng.integers(len(choices)))
            if len(choices) > 1 and (category, index) == previous:
                index = (index + 1) % len(choices)
            previous = category, index
            clip = choices[index].copy()
            # Gentle stereo balance movement; this is panning, not a binaural HRTF model.
            phase = rng.uniform(0, 2 * np.pi)
            pan = .4 * np.sin(np.arange(len(clip)) / RATE * .22 + phase)
            clip[:, 0] *= np.sqrt((1 - pan) / 2)
            clip[:, 1] *= np.sqrt((1 + pan) / 2)
            n = min(overlap, len(pending), len(clip))
            if n:
                ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
                pending[-n:] = pending[-n:] * (1 - ramp) + clip[:n] * ramp
            pending = np.concatenate([pending, clip[n:]])
            while len(pending) >= block_size + overlap:
                count = min(block_size, total - emitted)
                if not count:
                    return
                out = pending[:count].copy()
                pending = pending[count:]
                position = np.arange(emitted, emitted + count)
                envelope = np.minimum(np.clip(position / (2 * RATE), 0, 1), np.clip((total - 1 - position) / (3 * RATE), 0, 1))
                out *= envelope[:, None]
                yield out
                emitted += count


def render_session(library, output, preset='mixed', seconds=45, seed=0):
    import soundfile as sf
    with sf.SoundFile(output, 'w', samplerate=RATE, channels=2, subtype='PCM_24') as f:
        for block in library.blocks(preset, seconds, seed=seed):
            f.write(block)
