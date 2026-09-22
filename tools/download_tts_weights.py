"""Download the IndexTTS2 weights (used by ``scripts/install-tts.ps1``).

The official repository is ``IndexTeam/IndexTTS-2`` (22 files, ~5.9GB; the bulk is
``gpt.pth`` 3.5GB, ``s2mel.pth`` 1.2GB and the emotion model ``qwen0.6bemo4-merge``
1.2GB). This helper exists so the installer works even when the ``hf`` CLI is missing:
it talks to ``huggingface_hub`` directly and honours ``HF_ENDPOINT`` (the mirror).

Usage::

    python tools/download_tts_weights.py --dest models/tts/IndexTTS2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "IndexTeam/IndexTTS-2"

#: The files the model actually loads. Fetching them explicitly keeps the download
#: resumable and gives a useful error instead of silently missing a checkpoint.
REQUIRED = (
    "config.yaml",
    "gpt.pth",
    "s2mel.pth",
    "bpe.model",
    "feat1.pt",
    "feat2.pt",
    "wav2vec2bert_stats.pt",
    "qwen0.6bemo4-merge/config.json",
    "qwen0.6bemo4-merge/model.safetensors",
    "qwen0.6bemo4-merge/tokenizer.json",
    "qwen0.6bemo4-merge/vocab.json",
    "qwen0.6bemo4-merge/merges.txt",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 IndexTTS2 权重")
    parser.add_argument("--dest", required=True, help="目标目录（models/tts/IndexTTS2）")
    parser.add_argument("--mirror", default="https://hf-mirror.com", help="HF 镜像地址")
    parser.add_argument("--no-mirror", action="store_true", help="使用官方 huggingface.co")
    parser.add_argument("--all", action="store_true", help="下载整个仓库（含说明文档）")
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        print(f"缺少 huggingface_hub：{exc}")
        print("请先安装：pip install huggingface_hub")
        return 2

    if not args.no_mirror:
        os.environ.setdefault("HF_ENDPOINT", args.mirror)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"仓库：{REPO_ID}")
    print(f"目标：{dest}")
    print(f"端点：{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print("约 5.9GB，中断后重跑会断点续传。")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(dest),
        allow_patterns=None if args.all else list(REQUIRED) + ["*.json", "README.md", "LICENSE*"],
        max_workers=8,
    )

    missing = [name for name in REQUIRED if not (dest / name).exists()]
    if missing:
        print("以下文件缺失：")
        for name in missing:
            print(f"  - {name}")
        return 1
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"完成：{total / 1024 / 1024 / 1024:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
