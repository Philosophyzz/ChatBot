"""Download the GPT-SoVITS pretrained weights (used by install-gptsovits.ps1).

Repository ``lj1995/GPT-SoVITS`` (27 files, ~5.3GB). The files that matter depend on the
model generation you train:

* ``v2Pro/s2Gv2ProPlus.pth`` + ``s2Dv2ProPlus.pth`` — v2ProPlus generator/discriminator.
  Recommended default: the project's README states v2Pro matches v4 quality "with v2's
  hardware cost and speed", and v1/v2/v2Pro tolerate average-quality training audio
  better than v3/v4 do.
* ``gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt`` — the GPT
  (s1) starting point used by v2/v2Pro fine-tunes.
* ``chinese-hubert-base`` + ``chinese-roberta-wwm-ext-large`` — feature extractors, needed
  by every generation.
* ``gsv-v4-pretrained/*`` — only if you deliberately train v4.

Downloads go through ``HF_ENDPOINT`` (mirror by default) with Xet disabled: the mirror
does not proxy HuggingFace's Xet storage backend and answers HTTP 401 without this.

Usage::

    python tools/download_gptsovits_weights.py --dest vendor/GPT-SoVITS/GPT_SoVITS/pretrained_models
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "lj1995/GPT-SoVITS"

#: Everything needed for v2 / v2Pro / v2ProPlus training, plus v4 as an option.
REQUIRED = (
    "chinese-hubert-base/config.json",
    "chinese-hubert-base/preprocessor_config.json",
    "chinese-hubert-base/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/config.json",
    "chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "chinese-roberta-wwm-ext-large/tokenizer.json",
    "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "gsv-v2final-pretrained/s2D2333k.pth",
    "gsv-v2final-pretrained/s2G2333k.pth",
    "v2Pro/s2Dv2Pro.pth",
    "v2Pro/s2Gv2Pro.pth",
    "v2Pro/s2Dv2ProPlus.pth",
    "v2Pro/s2Gv2ProPlus.pth",
    "sv/pretrained_eres2netv2w24s4ep4.ckpt",
)

OPTIONAL = (
    "s1v3.ckpt",
    "gsv-v4-pretrained/s2Gv4.pth",
    "gsv-v4-pretrained/vocoder.pth",
    "hifigan_do_03357000",
    "hifigan_config.json",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 GPT-SoVITS 预训练权重")
    parser.add_argument("--dest", required=True, help="目标目录（GPT_SoVITS/pretrained_models）")
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--with-v4", action="store_true", help="同时下载 v4 权重（多约 0.9GB）")
    parser.add_argument("--all", action="store_true", help="下载整个仓库")
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        print(f"缺少 huggingface_hub：{exc}")
        print("请先安装：pip install huggingface_hub")
        return 2

    if not args.no_mirror:
        os.environ.setdefault("HF_ENDPOINT", args.mirror)
        # 关键：镜像不代理 Xet，开启时会 401
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    patterns = None if args.all else list(REQUIRED) + (list(OPTIONAL) if args.with_v4 else [])
    print(f"仓库：{REPO_ID}")
    print(f"目标：{dest.resolve()}")
    print(f"端点：{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}（Xet 已关闭）")
    print("约 5.3GB，中断后重跑会断点续传。")

    snapshot_download(repo_id=REPO_ID, local_dir=str(dest), allow_patterns=patterns, max_workers=8)

    missing = [name for name in REQUIRED if not (dest / name).exists()]
    total = sum(path.stat().st_size for path in dest.rglob("*") if path.is_file())
    print(f"已下载 {total / 1024 / 1024 / 1024:.2f} GB")
    if missing:
        print("以下必需文件缺失：")
        for name in missing:
            print(f"  - {name}")
        return 1
    print("权重齐全（v2ProPlus / v2Pro / v2final / hubert / roberta）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
