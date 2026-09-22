"""Check a GPT-SoVITS checkout for the files training needs (used by the installer).

Upstream renames and moves things between releases. Rather than hard-failing on one
expected path, this reports what it found and what is missing, so the training script can
decide — and so a user reading the output immediately knows whether they are blocked by a
missing pretrained file or by a renamed script.

Usage::

    python tools/check_gptsovits.py --repo vendor/GPT-SoVITS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

#: (relative path or glob, description, required?)
CHECKS: Tuple[Tuple[str, str, bool], ...] = (
    ("GPT_SoVITS/prepare_datasets/1-get-text.py", "数据集文本特征", False),
    ("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", "HuBERT 特征", False),
    ("GPT_SoVITS/prepare_datasets/3-get-semantic.py", "语义 token", False),
    ("GPT_SoVITS/s2_train.py", "SoVITS 训练入口", True),
    ("GPT_SoVITS/s1_train.py", "GPT 训练入口", True),
    ("GPT_SoVITS/configs/s2*.json", "SoVITS 配置模板", True),
    ("GPT_SoVITS/configs/s1*.yaml", "GPT 配置模板", True),
    ("api_v2.py", "推理 API（人设接进来要用）", True),
    ("tools/asr/fasterwhisper_asr.py", "ASR 标注（也可用本项目的 tts_dataset.py）", False),
    ("tools/slice_audio.py", "音频切片（也可用本项目的 tts_dataset.py）", False),
    ("ffmpeg.exe", "ffmpeg（切片/转码）", False),
    ("GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth", "v2ProPlus 生成器权重", True),
    ("GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth", "v2ProPlus 判别器权重", True),
    (
        "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "GPT(s1) 预训练权重",
        True,
    ),
    ("GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin", "HuBERT 权重", True),
    (
        "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin",
        "RoBERTa 权重",
        True,
    ),
    ("GPT_SoVITS/text/G2PWModel", "中文 G2PW 模型（中文合成必装）", True),
)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 GPT-SoVITS 安装完整性")
    parser.add_argument("--repo", required=True, help="GPT-SoVITS 仓库目录")
    args = parser.parse_args(argv)

    repo = Path(args.repo)
    if not repo.exists():
        print(f"  仓库不存在：{repo}（先跑 scripts\\install-gptsovits.ps1）")
        return 2

    missing_required: List[str] = []
    print(f"  检查 {repo}")
    for pattern, description, required in CHECKS:
        matches = list(repo.glob(pattern)) if "*" in pattern else ([repo / pattern] if (repo / pattern).exists() else [])
        if matches:
            detail = matches[0].name if "*" not in pattern else f"{len(matches)} 个：{', '.join(m.name for m in matches[:3])}"
            size = ""
            if matches[0].is_file():
                size = f"  {matches[0].stat().st_size / 1024 / 1024:.0f}MB"
            print(f"    [OK]   {description}{size}  ({detail})")
        else:
            mark = "缺失" if required else "可选"
            print(f"    [{'FAIL' if required else '跳过'}] {description} —— {mark}：{pattern}")
            if required:
                missing_required.append(pattern)

    print("")
    if missing_required:
        print(f"  结论：缺少 {len(missing_required)} 个必需项，先补齐再训练")
        return 1
    print("  结论：训练所需的文件齐全")
    return 0


if __name__ == "__main__":
    sys.exit(main())
