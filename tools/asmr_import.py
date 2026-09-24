"""Import an explicitly listened-to non-speech neural candidate into the pet library."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import soundfile as sf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from speech.asmr import finish_audio, PRESETS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('audio', type=Path)
    p.add_argument('--category', choices=[x for x in PRESETS if x != 'mixed'], required=True)
    p.add_argument('--reviewed-no-speech', action='store_true', help='Confirm you listened and found no speech/music/harsh artifacts')
    args = p.parse_args()
    if not args.reviewed_no_speech:
        p.error('Listen to the candidate first, then explicitly pass --reviewed-no-speech')
    data, rate = sf.read(args.audio, always_2d=True, dtype='float32')
    if rate != 44100 or data.shape[1] != 2 or not 1 <= len(data)/rate <= 30:
        p.error('Expected 1-30 seconds of 44.1 kHz stereo audio')
    metadata = json.loads(args.audio.with_suffix('.json').read_text(encoding='utf-8'))
    if metadata.get('model') != 'HKUSTAudio/AudioX':
        p.error('Missing supported model provenance')
    library = ROOT / 'data/asmr/library'
    manifest = json.loads((library / 'library.json').read_text())
    name = 'audiox-' + hashlib.sha256(args.audio.read_bytes()).hexdigest()[:16] + '.wav'
    sf.write(library / name, finish_audio(data), rate, subtype='PCM_24')
    row = {'file': name, 'category': args.category, 'approved': True, 'engine': 'AudioX + LoRA' if metadata.get('adapter') else 'AudioX',
           'license': 'CC-BY-NC-4.0', 'source': 'https://huggingface.co/HKUSTAudio/AudioX', 'generation': metadata}
    manifest['clips'] = [x for x in manifest['clips'] if x['file'] != name] + [row]
    temp = library / 'library.json.tmp'
    temp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temp.replace(library / 'library.json')
    print('Imported', name)


if __name__ == '__main__':
    main()
