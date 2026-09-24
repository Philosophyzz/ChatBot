"""Local AudioX text-only inference and genuine attention LoRA fine-tuning.

AudioX's learned empty-video/audio features are retained. Frozen CLIP and the
unused second audio encoder are not constructed for text-only generation.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path('D:/Models/asmr')
os.environ.setdefault('HF_HOME', str(MODEL_ROOT / 'hf-cache'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [str(ROOT / 'vendor/AudioX'), str(ROOT / 'src')]
import numpy as np
import soundfile as sf
import torch
from torch import nn
from safetensors.torch import save_file, load_file


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        delta = nn.functional.linear(nn.functional.linear(x.float(), self.lora_a), self.lora_b)
        return self.base(x) + delta.to(x.dtype)


def attach_lora(model, rank):
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit('.', 1)[-1] in ('to_q', 'to_kv', 'to_qkv', 'to_out'):
            parent_name, attr = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, attr, LoRALinear(module, rank))
            count += 1
    if not count:
        raise RuntimeError('No supported attention modules found')
    return count


class AudioXRuntime:
    def __init__(self, device='cuda', adapter=None):
        from audiox.models.factory import create_model_from_config
        from transformers import AutoTokenizer, T5EncoderModel
        torch.set_num_threads(6)
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info()
            if free < 5 * 1024**3:
                raise RuntimeError('AudioX requires at least 5 GiB free GPU memory. Pause the owned chat model with scripts/asmr-lab.ps1; do not run both training and the 9B LLM together.')
        for name in ('AudioX', 't5-base'):
            if not (MODEL_ROOT / name / '.complete.json').is_file():
                raise RuntimeError('Weights not verified; run scripts/download_asmr.py models')
        config = json.loads((MODEL_ROOT / 'AudioX/config.json').read_text())
        config['model']['conditioning']['configs'] = []
        self.model = create_model_from_config(copy.deepcopy(config)).eval().requires_grad_(False)
        state = torch.load(MODEL_ROOT / 'AudioX/model.ckpt', map_location='cpu', weights_only=True, mmap=True)
        state = state.get('state_dict', state)
        self.empty_video = state['conditioner.conditioners.video_prompt.empty_visual_feat'].detach().clone()
        self.empty_audio = state['conditioner.conditioners.audio_prompt.empty_audio_feat'].detach().clone()
        kept = {k: v for k, v in state.items() if not k.startswith('conditioner.')}
        missing, extra = self.model.load_state_dict(kept, strict=False)
        if missing or extra:
            raise RuntimeError(f'Unexpected AudioX state: missing={missing[:4]}, extra={extra[:4]}')
        del state, kept
        gc.collect()
        self.model.model.to(self.device, dtype=self.dtype)
        # The waveform codec remains fp32 and runs only during data encoding/decoding.
        self.model.pretransform.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / 't5-base', local_files_only=True)
        self.text_encoder = T5EncoderModel.from_pretrained(MODEL_ROOT / 't5-base', local_files_only=True).eval().requires_grad_(False).to(self.device, dtype=self.dtype)
        self.adapter_info = None
        if adapter:
            info = json.loads(Path(adapter).with_suffix('.json').read_text())
            attach_lora(self.model.model, info['rank'])
            values = load_file(str(adapter))
            actual = {k for k, _ in self.model.model.named_parameters() if 'lora_' in k}
            if set(values) != actual:
                raise RuntimeError('Adapter parameter set does not match base model')
            self.model.model.load_state_dict(values, strict=False)
            self.adapter_info = info
        print('AudioX loaded; learned null modalities retained; CUDA memory MiB', round(torch.cuda.memory_allocated()/1024**2) if self.device.type == 'cuda' else 0, flush=True)

    @torch.no_grad()
    def conditioning(self, prompt):
        tokens = self.tokenizer([prompt], truncation=True, max_length=128, padding='max_length', return_tensors='pt').to(self.device)
        text = self.text_encoder(**tokens).last_hidden_state * tokens.attention_mask.unsqueeze(-1)
        return torch.cat([self.empty_video.to(self.device, dtype=self.dtype), text,
                          self.empty_audio.to(self.device, dtype=self.dtype)], dim=1)

    @torch.no_grad()
    def generate(self, prompt, output, *, seconds=8, steps=50, seed=2026):
        from speech.asmr import finish_audio
        torch.manual_seed(seed)
        conditioning = self.conditioning(prompt)
        length = math.ceil(seconds * 44100 / 2048)
        noise = torch.randn(1, 64, length, device=self.device, dtype=torch.float32)
        # Use the upstream production sampler. Keep the iterative diffusion state
        # in fp32, even though matrix operations use mixed precision.
        from audiox.inference.sampling import sample_k
        self.model.model.eval()
        x = sample_k(self.model.model, noise, steps=steps, sampler_type='dpmpp-3m-sde',
                     sigma_min=.03, sigma_max=500, cfg_scale=7, batch_cfg=True,
                     device=str(self.device), cross_attn_cond=conditioning)
        raw = self.model.pretransform.decode(x.float())[0, :, :int(seconds * 44100)].T.float().cpu().numpy()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, finish_audio(raw), 44100, subtype='PCM_24')
        meta = {'prompt': prompt, 'seed': seed, 'steps': steps, 'seconds': seconds, 'sample_rate': 44100,
                'model': 'HKUSTAudio/AudioX', 'adapter': self.adapter_info, 'license': 'CC-BY-NC-4.0',
                'sampler': 'upstream dpmpp-3m-sde, fp32 state, sigma 0.03..500, cfg 7',
                'peak_raw': float(np.max(np.abs(raw))), 'approved_for_pet': False,
                'note': 'Research candidate: listen and reject any speech, music or harsh artifacts before pet-library import.'}
        output.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('GENERATED', output, flush=True)

    @torch.no_grad()
    def cache_data(self, manifest):
        records = [json.loads(s) for s in Path(manifest).read_text(encoding='utf-8').splitlines() if s.strip()]
        cached, condition_cache = [], {}
        for row in records:
            path = Path(row['audio'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Dataset changed after preparation')
            audio, sr = sf.read(path, dtype='float32', always_2d=True)
            if sr != 44100 or audio.shape[1] != 2:
                raise ValueError('Training requires 44.1 kHz stereo')
            samples = math.ceil(len(audio) / 2048) * 2048
            wav = torch.from_numpy(np.pad(audio, ((0, samples-len(audio)),(0,0)))).T[None].to(self.device)
            torch.manual_seed(1000 + len(cached))
            z = self.model.pretransform.encode(wav).detach().to(dtype=self.dtype, device='cpu')
            if row['text'] not in condition_cache:
                condition_cache[row['text']] = self.conditioning(row['text']).detach().cpu()
            cached.append((z, condition_cache[row['text']]))
        if not cached:
            raise ValueError('Empty training/validation split')
        return cached


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    runtime = AudioXRuntime(args.device)
    train_data = runtime.cache_data(args.dataset / 'train.jsonl')
    val_data = runtime.cache_data(args.dataset / 'validation.jsonl')
    # Once encoded, train only the diffusion attention adapters. Codec and T5 leave the GPU.
    runtime.model.pretransform.cpu()
    runtime.text_encoder.cpu()
    torch.cuda.empty_cache()
    net = runtime.model.model
    layer_count = attach_lora(net, args.rank)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    def loss_for(item, validation=False):
        z, cond = [value.to(runtime.device) for value in item]
        t = torch.rand(1, device=runtime.device) * .98 + .01
        noise = torch.randn_like(z)
        alpha = torch.cos(t * math.pi / 2)[:, None, None]
        sigma = torch.sin(t * math.pi / 2)[:, None, None]
        noisy = (z * alpha + noise * sigma).to(runtime.dtype)
        target = noise * alpha - z * sigma
        with torch.autocast(runtime.device.type, dtype=runtime.dtype, enabled=runtime.device.type == 'cuda'):
            pred = net(noisy, t, cross_attn_cond=cond, cfg_dropout_prob=0 if validation else .1)
        return nn.functional.mse_loss(pred.float(), target.float())

    @torch.no_grad()
    def evaluate():
        net.eval()
        # Common random numbers make before/after validation comparable.
        with torch.random.fork_rng(devices=[runtime.device.index or 0] if runtime.device.type == 'cuda' else []):
            torch.manual_seed(97531)
            values = [float(loss_for(row, True)) for row in val_data]
        return sum(values) / len(values)

    initial = evaluate()
    best = initial
    trace = []
    print('TRAIN', layer_count, 'LoRA layers', sum(p.numel() for p in params), 'parameters; validation before', initial, flush=True)
    started = time.time()
    best_path = out / 'best.safetensors'
    def save(path, step, validation):
        values = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items() if 'lora_' in k}
        save_file(values, str(path))
        info = {'base': 'HKUSTAudio/AudioX', 'base_revision': '3d49eff6430b739ba5a28357b1a0eedd843e6711',
                'code_revision': '3bdfb7081636b9e62224039e37dadaa264dc781f', 'rank': args.rank, 'step': step,
                'validation_mse': validation, 'seed': args.seed, 'lr': args.lr, 'license': 'CC-BY-NC-4.0',
                'dataset': str(args.dataset), 'pilot': True, 'approved_for_pet': False}
        path.with_suffix('.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    for step in range(1, args.steps + 1):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.accumulation):
            loss = loss_for(random.choice(train_data))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; checkpoint not published')
            (loss / args.accumulation).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        row = {'step': step, 'train_loss': sum(losses) / len(losses)}
        if step % args.eval_every == 0 or step == args.steps:
            validation = evaluate()
            row['validation_mse'] = validation
            save(out / f'step-{step}.safetensors', step, validation)
            if validation < best:
                best = validation
                save(best_path, step, validation)
        trace.append(row)
        (out / 'metrics.json').write_text(json.dumps({'initial_validation_mse': initial, 'best_validation_mse': best, 'trace': trace}, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    report = {'steps': args.steps, 'elapsed_seconds': time.time()-started, 'initial_validation_mse': initial,
              'best_validation_mse': best, 'best_adapter': str(best_path) if best_path.exists() else None,
              'trainable_parameters': sum(p.numel() for p in params), 'peak_cuda_mib': torch.cuda.max_memory_allocated()/1024**2 if runtime.device.type == 'cuda' else 0,
              'status': 'pilot_trained_not_listening_validated'}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('TRAINING_COMPLETE', json.dumps(report), flush=True)


def compare(args):
    from asmr_dataset import CAPTIONS
    candidates = ROOT / 'data/asmr/candidates'
    baseline = AudioXRuntime(args.device)
    for category, prompt in CAPTIONS.items():
        baseline.generate(prompt, candidates / f'{category}-base.wav', seconds=8, steps=50, seed=args.seed)
    del baseline
    gc.collect()
    torch.cuda.empty_cache()
    adapter = args.output / 'best.safetensors'
    if not adapter.exists():
        adapter = args.output / f'step-{args.steps}.safetensors'
    adapted = AudioXRuntime(args.device, adapter)
    for category, prompt in CAPTIONS.items():
        adapted.generate(prompt, candidates / f'{category}-lora.wav', seconds=8, steps=50, seed=args.seed)
    print('EXPERIMENT_COMPLETE: 3 baseline/adapted pairs; not auto-imported', flush=True)


def experiment(args):
    train(args)
    gc.collect()
    torch.cuda.empty_cache()
    compare(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['generate', 'train', 'experiment', 'compare'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--prompt', default='Soft slow dry paper rustling, close-up ASMR, no speech, no music, quiet background.')
    p.add_argument('--output', type=Path, default=MODEL_ROOT / 'adapters/AudioX-ASMR-pilot')
    p.add_argument('--dataset', type=Path, default=MODEL_ROOT / 'datasets/cc0-pilot/prepared')
    p.add_argument('--adapter', type=Path)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--rank', type=int, default=4)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--eval-every', type=int, default=20)
    p.add_argument('--seconds', type=float, default=8)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    if not 1 <= args.steps <= 100000 or not 1 <= args.seconds <= 10 or not 1 <= args.rank <= 64 or not 1 <= args.accumulation <= 32 or args.eval_every < 1 or not 0 < args.lr <= .01:
        p.error('steps, seconds or rank out of range')
    if args.action == 'experiment':
        experiment(args)
    elif args.action == 'compare':
        compare(args)
    elif args.action == 'train':
        train(args)
    else:
        AudioXRuntime(args.device, args.adapter).generate(args.prompt, args.output, seconds=args.seconds, steps=args.steps, seed=args.seed)
