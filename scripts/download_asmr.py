"""Pinned AudioX weights and a small, explicitly CC0 trigger-sound pilot set."""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from download_vllm_model import download_file
from download_ranges import download_ranges

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path('D:/Models/asmr')
REPOS = {
    'AudioX': ('HKUSTAudio/AudioX', '3d49eff6430b739ba5a28357b1a0eedd843e6711'),
    't5-base': ('google-t5/t5-base', 'a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'),
}
SOURCES = [
    ('paper', 'keweldog', 181774, 'train'),
    ('paper', 'NinjaSharkStudios', 592607, 'train'),
    ('paper', 'HarpyHarpHarp', 449127, 'validation'),
    ('tapping', 'mealwyrm', 495814, 'train'),
    ('tapping', 'alienistcog', 125790, 'train'),
    ('tapping', 'Topschool', 360462, 'validation'),
    ('brush', 'Vrymaa', 753283, 'train'),
    ('brush', 'saturdaysoundguy', 391442, 'validation'),
]


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'ChatBot-ASMR-research/1.0'}), timeout=60) as r:
        return r.read()


def models():
    for name, (repo, revision) in REPOS.items():
        dest = MODELS / name
        dest.mkdir(parents=True, exist_ok=True)
        manifest = dest / 'source.json'
        if revision is None and manifest.exists():
            revision = json.loads(manifest.read_text())['revision']
        meta = json.loads(get(f'https://huggingface.co/api/models/{repo}' + (f'/revision/{revision}' if revision else '') + '?blobs=true'))
        revision = meta['sha']
        entries = [x for x in meta['siblings'] if x['rfilename'] in {
            'model.ckpt', 'model.safetensors', 'config.json', 'README.md', 'LICENSE', 'LICENSE.md',
            'tokenizer.json', 'tokenizer_config.json', 'spiece.model', 'special_tokens_map.json'}]
        manifest.write_text(json.dumps({'repo': repo, 'revision': revision, 'files': entries}, indent=2), encoding='utf-8')
        for f in entries:
            download = download_ranges if f['size'] > 100_000_000 else download_file
            download(f"https://huggingface.co/{repo}/resolve/{revision}/{f['rfilename']}", dest / f['rfilename'], f['size'], (f.get('lfs') or {}).get('sha256'))
            print('verified', name, f['rfilename'], flush=True)
        (dest / '.complete.json').write_text(manifest.read_text(), encoding='utf-8')


def sounds():
    dest = MODELS / 'datasets/cc0-pilot'
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for category, author, ident, split in SOURCES:
        url = f'https://freesound.org/people/{author}/sounds/{ident}/'
        page = get(url).decode('utf-8')
        if not re.search(r'creativecommons\.org/publicdomain/zero/1\.0', page):
            raise RuntimeError(f'CC0 license not verified: {url}')
        previews = re.findall(r'https://cdn\.freesound\.org/previews/[^\s\"\x27<>]+-hq\.mp3', page)
        if not previews:
            raise RuntimeError(f'No public high quality preview: {url}')
        audio = get(previews[0])
        path = dest / f'{ident}.mp3'
        path.write_bytes(audio)
        (dest / f'{ident}.source.html').write_text(page, encoding='utf-8')
        records.append({'id': str(ident), 'category': category, 'author': author, 'source': url,
                        'download': previews[0], 'license': 'CC0-1.0', 'split': split,
                        'path': str(path), 'sha256': hashlib.sha256(audio).hexdigest(),
                        'quality': 'public MP3 preview; pilot only; replace with lossless original for production'})
        print('CC0', category, ident, len(audio), flush=True)
    (dest / 'sources.json').write_text(json.dumps({'created': time.time(), 'records': records}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=['models', 'sounds', 'all'])
    args = p.parse_args()
    if args.kind in ('models', 'all'):
        models()
    if args.kind in ('sounds', 'all'):
        sounds()
