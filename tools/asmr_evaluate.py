"""Technical audio checks and a reproducible before/after research report."""
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from faster_whisper.vad import get_speech_timestamps, VadOptions

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('D:/Models/asmr/adapters/AudioX-ASMR-pilot')
CLIPS = ROOT / 'data/asmr/candidates'


def main():
    reports = []
    waves = {}
    for path in sorted(CLIPS.glob('*.wav')):
        if path.stem.endswith('-comparison'):
            continue
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
        finite = bool(np.isfinite(audio).all())
        if not finite:
            raise ValueError('Non-finite generated audio')
        mono = resample_poly(audio.mean(axis=1), 16000, sr).astype('float32')
        segments = get_speech_timestamps(mono, VadOptions(threshold=.8, min_speech_duration_ms=300))
        true_peak = float(np.max(np.abs(resample_poly(audio, 4, 1))))
        row = {'file': path.name, 'sample_rate': sr, 'channels': audio.shape[1], 'seconds':len(audio)/sr,
               'sample_peak_dbfs':20*math.log10(max(float(np.max(np.abs(audio))),1e-9)),
               'true_peak_4x_dbfs':20*math.log10(max(true_peak,1e-9)),
               'rms_dbfs':20*math.log10(max(float(np.sqrt(np.mean(audio**2))),1e-9)),
               'possible_speech_seconds':sum((s['end']-s['start'])/16000 for s in segments),
               'stereo_difference_rms':float(np.sqrt(np.mean((audio[:,0]-audio[:,1])**2))),
               'finite':finite, 'approved_for_pet':False}
        reports.append(row)
        waves[path.stem] = audio
        print(json.dumps(row), flush=True)
    for category in ('paper','brush','tapping'):
        a, b = waves[category+'-base'], waves[category+'-lora']
        ar, br = float(np.sqrt(np.mean(a*a))), float(np.sqrt(np.mean(b*b)))
        target = min(ar, br)
        pair = np.concatenate([a * (target/max(ar,1e-9)),np.zeros((44100*2,2),dtype='float32'),b * (target/max(br,1e-9))])
        sf.write(CLIPS / (category+'-comparison.wav'), pair, 44100, subtype='PCM_24')
    (RUN / 'audio-qa.json').write_text(json.dumps({'method':'Silero VAD 0.8 is only a screen, not proof of no voices; true peak is 4x estimate','clips':reports},indent=2),encoding='utf-8')
    result = json.loads((RUN / 'report.json').read_text())
    delta = (1-result['best_validation_mse']/result['initial_validation_mse'])*100
    text = f'''# ASMR 首轮实验结果

- 基础模型：AudioX，44.1kHz 双声道；纯文本条件，保留学习到的空视频/空音频特征。
- 数据：8 个 CC0 原始录音，53 个切片约 207 秒；45 个训练、8 个验证，来源不重叠。
- 训练：{result['steps']} 个优化步，rank 4，{result['trainable_parameters']:,} 个可训练参数。
- 验证 MSE：{result['initial_validation_mse']:.6f} → {result['best_validation_mse']:.6f}，下降约 {delta:.2f}%。
- 优化循环耗时：{result['elapsed_seconds']:.1f} 秒，不包含下载、加载模型、音频编码与生成。
- 本次进程 PyTorch 峰值分配显存：{result['peak_cuda_mib']:.0f} MiB；不含桌面和其他进程占用。
- 状态：真实训练和检查点重新加载生成已完成；尚未经过人工盲听，不作成品 ASMR 音质结论。

![验证曲线](../data/asmr/validation.png)

## 样本

每个对照音频先播放基础模型 8 秒，静音 2 秒，再播放 LoRA 8 秒。两者使用同描述、同种子 2026、上游 DPM++ 3M SDE 采样器和 50 次去噪；对照文件把较响的一段降低到另一段的 RMS，不放大较轻的一段。

- [纸张对照](../data/asmr/candidates/paper-comparison.wav)
- [刷拭对照](../data/asmr/candidates/brush-comparison.wav)
- [轻敲对照](../data/asmr/candidates/tapping-comparison.wav)
- [当前桌宠 CC0 编排试听（45秒）](../data/asmr/ASMR-45s-CC0-preview.wav)

## 自动音频检查

| 文件 | 时长 | 峰值 dBFS | 4倍过采样峰值 | RMS dBFS | VAD可能语音秒数 |
|---|---:|---:|---:|---:|---:|
'''
    for row in reports:
        text += f"| {row['file']} | {row['seconds']:.1f} | {row['sample_peak_dbfs']:.1f} | {row['true_peak_4x_dbfs']:.1f} | {row['rms_dbfs']:.1f} | {row['possible_speech_seconds']:.2f} |\n"
    text += '\n这批神经样本仍存在明显局限：纸张样本声音偏稀疏，多数样本左右声道差异很小，适配后未必更好听。验证 MSE 的改善不能作为直接上线依据。自动检查不能证明没有人声、音乐或刺耳伪影，也不能证明让人放松。所有神经样本保持未收录状态；当前桌宠播放 CC0 音库。试听后可用 `tools/asmr_import.py` 明确收录。\n\n权重与指标：`D:/Models/asmr/adapters/AudioX-ASMR-pilot/`；完整方案见 [ASMR生成与训练方案](ASMR生成与训练方案.md)。\n'
    (ROOT / 'docs/ASMR首轮实验结果.md').write_text(text,encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        m=json.loads((RUN/'metrics.json').read_text())
        rows=[{'step':0,'validation_mse':m['initial_validation_mse']}]+[r for r in m['trace'] if 'validation_mse' in r]
        fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
        ax.plot([r['step'] for r in rows],[r['validation_mse'] for r in rows],marker='o',color='#6460bf')
        ax.set(xlabel='Optimizer step',ylabel='Held-out v-prediction MSE',title='AudioX ASMR pilot: source-separated validation')
        ax.grid(alpha=.2)
        fig.savefig(ROOT/'data/asmr/validation.png',dpi=170)
        plt.close(fig)
    except ImportError:
        pass


if __name__=='__main__':
    main()
