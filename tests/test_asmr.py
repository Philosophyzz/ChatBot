import json
from pathlib import Path
import sys
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from speech.asmr import finish_audio, TextureLibrary, RATE, parse_command


def test_only_explicit_user_playback_commands_start_audio():
    assert parse_command('帮我播放纸张 ASMR 15分钟') == {'action':'start','preset':'paper','seconds':900}
    assert parse_command('暂停ASMR') == {'action':'pause'}
    assert parse_command('如何训练一个可以播放ASMR的模型？') is None
    assert parse_command('不要播放ASMR') is None
    assert parse_command('播放ASMR999分钟') is None


def test_pcm_preserves_stereo_and_bounds_peaks():
    t = np.arange(RATE) / RATE
    source = np.stack([np.sin(t * 1200), .1 * np.sin(t * 1300)], axis=1)
    out = finish_audio(source)
    assert out.shape == (RATE, 2)
    assert np.max(np.abs(out)) <= 10**(-8/20) + 1e-6
    assert np.sqrt(np.mean(out[:, 0] ** 2)) > 8 * np.sqrt(np.mean(out[:, 1] ** 2))
    assert np.all(out[0] == 0) and np.all(out[-1] == 0)
    with pytest.raises(ValueError):
        finish_audio(np.full((20, 2), np.nan))


def test_stream_has_exact_length_fades_and_no_path_escape(tmp_path):
    audio = np.random.default_rng(1).normal(size=(2 * RATE, 2)).astype('float32') * .05
    sf.write(tmp_path / 'test.wav', audio, RATE)
    manifest = {'clips': [{'file': 'test.wav', 'category': 'paper', 'approved': True}]}
    (tmp_path / 'library.json').write_text(json.dumps(manifest))
    library = TextureLibrary(tmp_path)
    result = np.concatenate(list(library.blocks('paper', 5.3, seed=4)))
    assert result.shape == (int(5.3 * RATE), 2)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) < .4
    assert np.all(result[0] == 0) and np.all(result[-1] == 0)
    with pytest.raises(ValueError):
        list(library.blocks('unknown', 5))
    manifest['clips'][0]['file'] = '../private.wav'
    (tmp_path / 'library.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='outside'):
        TextureLibrary(tmp_path)


def test_unapproved_neural_candidates_never_autoplay(tmp_path):
    (tmp_path / 'library.json').write_text(json.dumps({'clips': [{'file': 'missing.wav', 'category': 'brush', 'approved': False}]}))
    with pytest.raises(ValueError, match='No approved'):
        TextureLibrary(tmp_path)


def test_pilot_split_is_disjoint_by_source():
    root = Path('D:/Models/asmr/datasets/cc0-pilot/prepared')
    if not (root / 'report.json').exists():
        pytest.skip('local pilot assets not installed')
    report = json.loads((root / 'report.json').read_text())
    assert not set(report['train_source_ids']) & set(report['validation_source_ids'])


def test_model_switch_respects_live_audio_job_and_ignores_stale_owner(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import psutil
    from llm.supervisor import ModelSupervisor, ModelSwitchError
    state = tmp_path / 'data/asmr/lab-state.json'
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'state':'running','pid':1234}))
    manager = ModelSupervisor(tmp_path)
    monkeypatch.setattr(psutil, 'Process', lambda pid: SimpleNamespace(cmdline=lambda: ['python', 'D:/Harness/ChatBot/tools/asmr_lab.py']))
    with pytest.raises(ModelSwitchError, match='ASMR'):
        manager.check_audio_training()
    def expired(pid):
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(psutil, 'Process', expired)
    manager.check_audio_training()
