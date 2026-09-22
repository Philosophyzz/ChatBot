"""Turn a folder of raw recordings into a GPT-SoVITS fine-tuning dataset.

Why this exists instead of "just use the WebUI": the WebUI does the same steps, but
only interactively, and it will happily train on 40-second clips with music in them.
This tool is scriptable (so ``scripts/train-tts.ps1`` can run unattended), reuses the
project's own audio utilities and Whisper install, and — most importantly — **refuses to
quietly produce a bad dataset**: it reports duration distribution, clips that were
dropped and why, duplicate transcripts, and near-silent slices.

Pipeline (all steps optional/configurable)::

    scan → decode to mono 32k → trim silence → slice on pauses → normalize loudness
         → transcribe (faster-whisper) → write sliced/*.wav + dataset.list + report.md

Output ``dataset.list`` uses the format GPT-SoVITS expects::

    <absolute wav path>|<speaker>|<language>|<text>

Usage::

    python tools/tts_dataset.py --input D:\\voice\\raw --out D:\\voice\\dataset --speaker my_voice
    python tools/tts_dataset.py --input raw --out ds --text-file labels.tsv   # 已有标注，跳过 ASR
    python tools/tts_dataset.py --input raw --out ds --dry-run                # 只看会做什么
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
# Windows 控制台默认是 GBK：脚本里的 ✅ / ⚠ / 中文一旦编码不了就 UnicodeEncodeError，
# 检查明明通过了却以非 0 退出 —— 用户会以为环境坏了。统一切到 UTF-8 输出。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 被重定向的流
        pass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv"}
TARGET_RATE = 32000  # GPT-SoVITS trains on 32 kHz audio


@dataclass
class Slice:
    path: Path
    duration_s: float
    text: str = ""
    source: str = ""

    def to_list_line(self, speaker: str, language: str) -> str:
        return f"{self.path}|{speaker}|{language}|{self.text}"


@dataclass
class Report:
    inputs: int = 0
    decoded: int = 0
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    slices: List[Slice] = field(default_factory=list)
    dropped: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def duration(self) -> float:
        return sum(item.duration_s for item in self.slices)


# ---------------------------------------------------------------------------------------
# audio helpers
# ---------------------------------------------------------------------------------------


def decode_audio(path: Path) -> Tuple[bytes, int]:
    """Decode any supported file to mono 16-bit PCM at :data:`TARGET_RATE`."""
    from speech.audio import decode_to_pcm

    data = path.read_bytes()
    pcm, rate, _channels = decode_to_pcm(data, sample_rate=TARGET_RATE)
    return pcm, rate


def pcm_duration(pcm: bytes, rate: int) -> float:
    return len(pcm) / 2 / rate


def split_on_silence(pcm: bytes, rate: int, *, min_s: float, max_s: float, threshold: int = 320) -> List[bytes]:
    """Cut PCM into ``min_s..max_s`` chunks, preferring silence boundaries.

    Long recordings are the normal case (a video, a podcast), and GPT-SoVITS wants short
    utterances: 3–10 seconds. Cutting on pauses keeps words intact; a hard cut in the
    middle of a word teaches the model a wrong pronunciation.
    """
    import array

    samples = array.array("h")
    samples.frombytes(pcm)
    window = int(rate * 0.02)  # 20 ms frames
    frames: List[Tuple[int, float]] = []
    for start in range(0, len(samples) - window, window):
        chunk = samples[start : start + window]
        peak = max((abs(value) for value in chunk), default=0)
        frames.append((start, peak))

    min_len = int(min_s * rate)
    max_len = int(max_s * rate)

    pieces: List[bytes] = []
    piece_start = 0
    index = 0
    while index < len(frames):
        start, _peak = frames[index]
        length = start - piece_start
        if length >= min_len:
            # Look for a quiet frame to cut on, but never exceed max_len.
            cut_at: Optional[int] = None
            look = index
            while look < len(frames) and frames[look][0] - piece_start <= max_len:
                if frames[look][1] < threshold and frames[look][0] - piece_start >= min_len:
                    cut_at = frames[look][0]
                look += 1
            if cut_at is None and length >= max_len:
                cut_at = piece_start + max_len
            if cut_at:
                pieces.append(pcm[piece_start * 2 : cut_at * 2])
                piece_start = cut_at
                index = max(index, next((i for i, f in enumerate(frames) if f[0] >= cut_at), index))
                continue
        index += 1
    if len(samples) - piece_start >= min_len:
        pieces.append(pcm[piece_start * 2 :])
    return [piece for piece in pieces if pcm_duration(piece, rate) >= min_s]


def normalize_pcm(pcm: bytes, *, target_peak: float = 0.89) -> bytes:
    """Scale PCM so its peak sits just below full scale (loudness consistency)."""
    import array

    samples = array.array("h")
    samples.frombytes(pcm)
    peak = max((abs(value) for value in samples), default=0)
    if peak == 0:
        return pcm
    scale = (target_peak * 32767) / peak
    if 0.95 <= scale <= 1.05:  # already fine, avoid pointless requantization
        return pcm
    for index, value in enumerate(samples):
        scaled = int(value * scale)
        samples[index] = 32767 if scaled > 32767 else (-32768 if scaled < -32768 else scaled)
    return samples.tobytes()


def peak_level(pcm: bytes) -> float:
    import array

    samples = array.array("h")
    samples.frombytes(pcm)
    peak = max((abs(value) for value in samples), default=0)
    return peak / 32768.0


# ---------------------------------------------------------------------------------------
# transcription
# ---------------------------------------------------------------------------------------


def resolve_local_whisper(model_name: str) -> Optional[Path]:
    """Find an already-downloaded CTranslate2 whisper model on this machine.

    ``speech.stt`` owns the naming rules (``large-v3-turbo`` →
    ``deepdml/faster-whisper-large-v3-turbo-ct2``, and the downloader drops it in
    ``models/whisper/large-v3-turbo``); reuse them so the two never disagree about where the
    model lives. Without this the tool quietly tried to re-download 1.6 GB.
    """
    root = ROOT / "models" / "whisper"
    if not root.exists():
        return None
    candidates = [root / model_name]
    bare = model_name.split("/")[-1]
    candidates += [
        root / bare,
        root / bare.replace("faster-whisper-", ""),
        root / "large-v3-turbo",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "model.bin").exists():
            return candidate
    try:  # 让 speech.stt 的别名表也参与判断（它才是权威）
        sys.path.insert(0, str(ROOT / "src"))
        from speech.stt import resolve_stt_model  # noqa: PLC0415

        resolved = Path(str(resolve_stt_model(model_name)))
        if resolved.is_dir() and (resolved / "model.bin").exists():
            return resolved
    except Exception:  # noqa: BLE001 - 只是尽力而为
        pass
    return None


class Transcriber:
    """Lazy faster-whisper wrapper. Missing model/disk turns into a readable error."""

    def __init__(
        self,
        *,
        model: str = "deepdml/faster-whisper-large-v3-turbo-ct2",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        mirror: str = "https://hf-mirror.com",
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.mirror = mirror
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        # CTranslate2 模型的仓库名在国内直连 HuggingFace 经常超时；镜像先顶上。
        # 注意：Systran/faster-whisper-large-v3-turbo 在镜像上不存在（401），
        # 所以默认用的是等价的 deepdml/...-ct2 仓库。
        if self.mirror and not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = self.mirror
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"没装 faster-whisper（{exc}）；也可以先用 --text-file 提供标注") from exc
        download_root = ROOT / "models" / "whisper"
        download_root.mkdir(parents=True, exist_ok=True)

        # 本机已经下好模型时，直接用本地目录 —— 否则 faster-whisper 会拿仓库名去
        # models/whisper/<repo 名> 找，找不到就联网重下（离线时直接失败）。这个项目
        # 的语音识别模型由 tools/download_whisper.py 落在 models\whisper\large-v3-turbo，
        # 与仓库名完全不同，所以必须显式认出来，并优先使用它。
        target = self.model_name
        local = resolve_local_whisper(self.model_name)
        if local is not None:
            target = str(local)

        # CT2 的 CUDA 后端要 cublas64_12.dll / cudnn 这一套。本机驱动是 CUDA 13，这些 DLL
        # 只在另一个 conda 环境的 torch\lib 里；引擎靠 speech.stt._prepare_cuda_dlls()
        # 把那些目录挂进 DLL 搜索路径，独立工具必须做同样的事，否则直接报
        # "Library cublas64_12.dll is not found"（实测就是这么挂的）。
        try:
            if str(ROOT / "src") not in sys.path:
                sys.path.insert(0, str(ROOT / "src"))
            from speech.stt import _prepare_cuda_dlls  # noqa: PLC0415

            added = _prepare_cuda_dlls()
            if added:
                print(f"  已挂载 {len(added)} 个 CUDA 运行库目录")
        except Exception as exc:  # noqa: BLE001 - 挂不上就退 CPU，不因此中断
            print(f"  [!] 准备 CUDA 运行库失败（{exc}），必要时会回退 CPU")

        try:
            self._model = WhisperModel(
                target,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(download_root),
            )
        except Exception as exc:  # noqa: BLE001
            # CUDA 起不来不算致命：切 CPU int8 慢一些，但标注照样能产出，
            # 比让用户卡在"cublas 找不到"上强。
            if str(self.device).startswith("cuda"):
                print(f"  [!] CUDA 加载失败（{type(exc).__name__}: {exc}），改用 CPU int8 重试")
                try:
                    self._model = WhisperModel(target, device="cpu", compute_type="int8")
                    self.device = "cpu"
                    return self._model
                except Exception as cpu_exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"CPU 上也无法加载转写模型（{type(cpu_exc).__name__}: {cpu_exc}）。"
                        f"可以用 --text-file 提供标注，或确认 {download_root} 下的模型完整"
                    ) from cpu_exc
            raise RuntimeError(
                f"加载转写模型失败（{type(exc).__name__}: {exc}）。"
                f"离线环境可以先用 --text-file 提供标注，或先手动把模型放到 {download_root}"
            ) from exc
        return self._model

    def transcribe(self, wav_path: Path, *, language: str = "zh") -> str:
        model = self._load()
        segments, _info = model.transcribe(str(wav_path), language=language, vad_filter=False)
        return "".join(segment.text for segment in segments).strip()


# ---------------------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------------------


def load_text_file(path: Path) -> Dict[str, str]:
    """Read existing labels: ``name<TAB>text`` or ``name|text`` per line."""
    labels: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for separator in ("\t", "|"):
            if separator in line:
                name, _, text = line.partition(separator)
                labels[Path(name.strip()).stem] = text.strip()
                break
    return labels


def collect_inputs(target: Path, *, recursive: bool = True) -> List[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in AUDIO_SUFFIXES else []
    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in target.glob(pattern) if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def build(
    *,
    inputs: Sequence[Path],
    out_dir: Path,
    speaker: str,
    language: str,
    min_s: float,
    max_s: float,
    normalize: bool,
    labels: Optional[Dict[str, str]],
    transcriber: Optional[Transcriber],
    dry_run: bool = False,
) -> Report:
    from speech.audio import pcm_to_wav, trim_silence

    report = Report(inputs=len(inputs))
    sliced_dir = out_dir / "sliced"
    if not dry_run:
        sliced_dir.mkdir(parents=True, exist_ok=True)
    keep = {name for name in (labels or {})}

    for source in inputs:
        try:
            pcm, rate = decode_audio(source)
        except Exception as exc:  # noqa: BLE001
            report.skipped.append((source.name, f"解码失败：{type(exc).__name__}: {exc}"))
            continue
        report.decoded += 1
        pcm = trim_silence(pcm, threshold=280, padding_ms=120)
        total = pcm_duration(pcm, rate)
        if total < min_s:
            report.dropped.append((source.name, f"太短（{total:.1f}s < {min_s}s）"))
            continue

        pieces = split_on_silence(pcm, rate, min_s=min_s, max_s=max_s)
        for index, piece in enumerate(pieces):
            duration = pcm_duration(piece, rate)
            if duration > max_s + 0.5:
                report.dropped.append((f"{source.name}#{index}", f"过长且无法在静音处切分（{duration:.1f}s）"))
                continue
            level = peak_level(piece)
            if level < 0.02:
                report.dropped.append((f"{source.name}#{index}", f"几乎无声（峰值 {level:.3f}）"))
                continue
            if normalize:
                piece = normalize_pcm(piece)
            name = f"{source.stem}_{index:03d}.wav"
            target = sliced_dir / name
            if not dry_run:
                target.write_bytes(pcm_to_wav(piece, sample_rate=rate, channels=1))
            text = ""
            if labels:
                text = labels.get(source.stem, "") or labels.get(f"{source.stem}_{index:03d}", "")
            elif transcriber and not dry_run:
                try:
                    text = transcriber.transcribe(target, language=language)
                except Exception as exc:  # noqa: BLE001
                    report.warnings.append(f"{name} 转写失败：{exc}")
            report.slices.append(Slice(path=target, duration_s=duration, text=text, source=source.name))

    if labels and report.slices and not any(item.text for item in report.slices):
        report.warnings.append("标注文件里没有匹配到任何文件名，请检查 --text-file 的键是否为音频文件名")

    # sanity checks that would otherwise become "why does my model sound bad" later
    durations = [item.duration_s for item in report.slices]
    if durations:
        short = sum(1 for value in durations if value < 2.0)
        if short:
            report.warnings.append(f"{short} 个切片短于 2 秒，建议提高 --min-s")
        if statistics.mean(durations) > 12:
            report.warnings.append("平均切片超过 12 秒，GPT-SoVITS 更适合 3~10 秒的句子")
    texts = [item.text for item in report.slices if item.text]
    duplicates = {text for text in texts if texts.count(text) > 1}
    if duplicates:
        report.warnings.append(f"{len(duplicates)} 条转写文本重复（同一句话出现多次会削弱泛化）")
    if texts and len(texts) < len(report.slices):
        report.warnings.append(f"{len(report.slices) - len(texts)} 个切片没有文本，训练前必须补齐")
    _ = keep
    return report


def write_outputs(report: Report, *, out_dir: Path, speaker: str, language: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "dataset.list"
    lines = [
        item.to_list_line(speaker, language)
        for item in report.slices
        if item.text
    ]
    list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    durations = [item.duration_s for item in report.slices]
    summary = {
        "speaker": speaker,
        "language": language,
        "inputs": report.inputs,
        "decoded": report.decoded,
        "slices": len(report.slices),
        "labelled": len(lines),
        "total_seconds": round(report.duration(), 1),
        "mean_seconds": round(statistics.mean(durations), 2) if durations else 0,
        "min_seconds": round(min(durations), 2) if durations else 0,
        "max_seconds": round(max(durations), 2) if durations else 0,
        "skipped": report.skipped,
        "dropped": report.dropped,
        "warnings": report.warnings,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = [
        f"# 数据集报告：{speaker}",
        "",
        f"* 输入文件：{report.inputs}（成功解码 {report.decoded}）",
        f"* 切片：{len(report.slices)} 条，合计 **{report.duration() / 60:.1f} 分钟**",
        f"* 有文本的切片：{len(lines)}",
    ]
    if durations:
        markdown.append(
            f"* 时长分布：最短 {min(durations):.1f}s / 平均 {statistics.mean(durations):.1f}s / 最长 {max(durations):.1f}s"
        )
    if report.dropped:
        markdown += ["", "## 被丢弃的片段"] + [f"* {name}：{reason}" for name, reason in report.dropped]
    if report.skipped:
        markdown += ["", "## 解码失败"] + [f"* {name}：{reason}" for name, reason in report.skipped]
    if report.warnings:
        markdown += ["", "## 提醒"] + [f"* {warning}" for warning in report.warnings]
    markdown += [
        "",
        "## 下一步",
        "",
        "```powershell",
        "powershell -ExecutionPolicy Bypass -File scripts\\train-tts.ps1 -DataDir <上面这个目录> -Speaker "
        + speaker,
        "```",
    ]
    report_md = out_dir / "report.md"
    report_md.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return list_path, report_md


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="把原始录音整理成 GPT-SoVITS 微调数据集")
    parser.add_argument("--input", required=True, help="音频文件或目录")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--speaker", default="my_voice", help="说话人名字（写进 dataset.list）")
    parser.add_argument("--language", default="zh", choices=["zh", "en", "ja", "ko", "yue"])
    parser.add_argument("--min-s", type=float, default=3.0, help="切片最短时长（秒）")
    parser.add_argument("--max-s", type=float, default=10.0, help="切片最长时长（秒）")
    parser.add_argument("--no-normalize", action="store_true", help="不做峰值归一化")
    parser.add_argument("--text-file", default=None, help="已有标注（每行：文件名<TAB>文本），给了就不跑 ASR")
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--asr-compute-type", default="int8_float16")
    parser.add_argument("--no-asr", action="store_true", help="只切片不转写（之后自己补文本）")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只统计与检查，不写任何文件")
    args = parser.parse_args(argv)

    target = Path(args.input)
    if not target.exists():
        print(f"找不到输入：{target}")
        return 2
    inputs = collect_inputs(target, recursive=not args.no_recursive)
    if not inputs:
        print(f"在 {target} 里没有找到音频文件（支持 {', '.join(sorted(AUDIO_SUFFIXES))}）")
        return 2

    out_dir = Path(args.out)
    labels = load_text_file(Path(args.text_file)) if args.text_file else None
    transcriber = None
    if labels is None and not args.no_asr:
        transcriber = Transcriber(
            model=args.asr_model, device=args.asr_device, compute_type=args.asr_compute_type
        )

    print(f"  输入 {len(inputs)} 个音频文件 -> {out_dir}")
    print(f"  说话人 {args.speaker} / 语言 {args.language} / 切片 {args.min_s}~{args.max_s}s")
    if labels:
        print(f"  使用已有标注 {args.text_file}（{len(labels)} 条）")
    elif args.no_asr:
        print("  跳过转写（--no-asr）：dataset.list 里文本会留空，需要你自己补")
    else:
        print(f"  转写模型 {args.asr_model}（{args.asr_device}）")

    report = build(
        inputs=inputs,
        out_dir=out_dir,
        speaker=args.speaker,
        language=args.language,
        min_s=args.min_s,
        max_s=args.max_s,
        normalize=not args.no_normalize,
        labels=labels,
        transcriber=transcriber,
        dry_run=args.dry_run,
    )

    print("")
    print(f"  切片 {len(report.slices)} 条，合计 {report.duration() / 60:.1f} 分钟")
    for name, reason in report.skipped[:10]:
        print(f"    [跳过] {name}：{reason}")
    for name, reason in report.dropped[:10]:
        print(f"    [丢弃] {name}：{reason}")
    for warning in report.warnings:
        print(f"    [!] {warning}")

    if args.dry_run:
        print("\n  这是预演，没有写任何文件。")
        return 0

    list_path, report_md = write_outputs(report, out_dir=out_dir, speaker=args.speaker, language=args.language)
    print("")
    print(f"  dataset.list -> {list_path}")
    print(f"  报告       -> {report_md}")
    print(f"  切片目录   -> {out_dir / 'sliced'}")
    if len(report.slices) < 8 or report.duration() < 120:
        print("  提醒：数据偏少。GPT-SoVITS 少样本 1 分钟能出声，但要像样建议 10 分钟以上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
