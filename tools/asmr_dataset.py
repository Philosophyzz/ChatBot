"""CC0 pilot preparation; source-separated splits, unchanged stereo, no VAD trimming."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from speech.asmr import finish_audio, TextureLibrary, render_session, RATE

CAPTIONS = {
    'paper': 'Close-up dry paper gently crinkling and rustling, isolated delicate ASMR texture, quiet background, no speech, no music.',
    'brush': 'A soft paint brush gently brushing a surface in slow strokes, close microphone, isolated dry ASMR sound, no speech, no music.',
    'tapping': 'Fingertips softly tapping a small wooden surface at a slow irregular pace, isolated close-up ASMR, quiet background, no speech, no music.',
}


def prepare(source, output, library):
    records = json.loads((source / 'sources.json').read_text(encoding='utf-8'))['records']
    output.mkdir(parents=True, exist_ok=True)
    library.mkdir(parents=True, exist_ok=True)
    rows, clips = [], []
    for item in records:
        if item['license'] != 'CC0-1.0':
            raise ValueError('Pilot only accepts explicit CC0 sources')
        path = Path(item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Source checksum changed')
        data, sr = sf.read(path, dtype='float32', always_2d=True)
        if not np.isfinite(data).all() or data.shape[1] > 2:
            raise ValueError('Invalid source PCM')
        if sr != RATE:
            factor = gcd(sr, RATE)
            data = resample_poly(data, RATE // factor, sr // factor).astype('float32')
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        # Disjoint 4-second crops; no adjacent crop crosses a source-level train/val split.
        for i, start in enumerate(range(0, len(data) - 2 * RATE, 4 * RATE)):
            raw = data[start:start + 4 * RATE]
            if len(raw) < 2 * RATE or np.sqrt(np.mean(raw ** 2)) < 1e-5:
                continue
            audio = finish_audio(raw, rms_db=-20, peak_db=-3, fade_s=.02)
            name = f"{item['category']}-{item['id']}-{i:03d}.wav"
            target = output / name
            sf.write(target, audio, RATE, subtype='PCM_24')
            rows.append({'audio': str(target.resolve()), 'text': CAPTIONS[item['category']], 'category': item['category'],
                         'source_id': item['id'], 'split': item['split'], 'duration': len(audio) / RATE, 'license': item['license'], 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        # Runtime baseline uses original CC0 recordings, distinctly labelled from neural outputs.
        for i, start in enumerate(range(0, len(data) - RATE, 8 * RATE)):
            raw = data[start:start + 10 * RATE]
            if len(raw) < RATE or np.sqrt(np.mean(raw ** 2)) < 1e-5:
                continue
            name = f"cc0-{item['category']}-{item['id']}-{i:03d}.wav"
            sf.write(library / name, finish_audio(raw), RATE, subtype='PCM_24')
            clips.append({'file': name, 'category': item['category'], 'approved': True,
                          'engine': 'CC0 recording', 'source': item['source'], 'author': item['author'], 'license': item['license']})
    for split in ('train', 'validation'):
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows if r['split'] == split), encoding='utf-8')
    train_ids = {r['source_id'] for r in rows if r['split'] == 'train'}
    val_ids = {r['source_id'] for r in rows if r['split'] == 'validation'}
    assert not train_ids & val_ids
    report = {'source_count': len(records), 'clips': len(rows), 'seconds': sum(r['duration'] for r in rows),
              'train_clips': sum(r['split'] == 'train' for r in rows), 'validation_clips': sum(r['split'] == 'validation' for r in rows),
              'train_source_ids': sorted(train_ids), 'validation_source_ids': sorted(val_ids),
              'quality': 'small MP3-preview pilot; not production ASMR training data'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (library / 'library.json').write_text(json.dumps({'version': 1, 'clips': clips}, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('D:/Models/asmr/datasets/cc0-pilot'))
    p.add_argument('--output', type=Path, default=Path('D:/Models/asmr/datasets/cc0-pilot/prepared'))
    p.add_argument('--library', type=Path, default=ROOT / 'data/asmr/library')
    args = p.parse_args()
    prepare(args.source, args.output, args.library)
    demo = ROOT / 'data/asmr/ASMR-45s-CC0-preview.wav'
    render_session(TextureLibrary(args.library), demo, seed=2026)
    print('Preview:', demo)
