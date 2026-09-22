"""Download the Whisper STT model — ModelScope first, HuggingFace mirror as fallback.

Measured on this machine: the HF mirror trickles and eventually stalls on the 1.6GB
``model.bin`` (a 1.5MB/s range probe, but whole-file downloads die around 40–850MB),
while ModelScope served the same file at ~4.4MB/s with working range requests. China-hosted
ModelScope is also the more sensible default for a China-based install.

Both sources host the *same* CTranslate2 conversion of ``large-v3-turbo``. Files are
fetched with ``tools/download_file.py``'s resumable range downloader, so an interrupted
run continues where it stopped and a half-written file is never mistaken for a finished
model (it lives as ``.part`` until the length matches).

Usage::

    python tools/download_whisper.py                      # -> models/whisper/large-v3-turbo
    python tools/download_whisper.py --source hf          # force the HuggingFace mirror
    python tools/download_whisper.py --dest D:\\models\\whisper\\turbo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from download_file import download, head_size, human  # noqa: E402

#: (source name, repo id, url builder)
SOURCES: Dict[str, Tuple[str, str]] = {
    # ModelScope: fast in China, supports Range.
    "modelscope": ("pengzhendong/faster-whisper-large-v3-turbo", "https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={file}"),
    # HuggingFace mirror: same weights, much slower here.
    "hf": ("deepdml/faster-whisper-large-v3-turbo-ct2", "https://hf-mirror.com/{repo}/resolve/main/{file}"),
}

FILES: List[str] = [
    "model.bin",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
]

EXPECTED_MODEL_BIN = 1_617_884_929  # bytes, from the repo metadata


def modelscope_metadata(repo: str) -> Dict[str, Tuple[int, str]]:
    """``{filename: (size, sha256)}`` from the ModelScope API.

    Verifying sha256 is the only way to be sure a 1.5GB model file is *correct* rather
    than merely the right length — and the API hands the hash out for free.
    """
    import httpx

    try:
        with httpx.Client(timeout=30, trust_env=False, follow_redirects=True) as client:
            response = client.get(
                f"https://modelscope.cn/api/v1/models/{repo}/repo/files",
                params={"Revision": "master"},
            )
            files = response.json()["Data"]["Files"]
    except Exception as exc:  # noqa: BLE001
        print(f"  警告：拿不到 ModelScope 文件清单（{type(exc).__name__}: {exc}），跳过哈希校验")
        return {}
    return {
        entry["Path"]: (int(entry.get("Size") or 0), str(entry.get("Sha256") or ""))
        for entry in files
        if entry.get("Path")
    }


def fetch(source: str, dest: Path, *, chunk_mb: int = 24, quiet: bool = False) -> bool:
    repo, template = SOURCES[source]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  源: {source}（{repo}）")
    print(f"  目标目录: {dest}")
    metadata = modelscope_metadata(repo) if source == "modelscope" else {}
    failures = 0
    for name in FILES:
        url = template.format(repo=repo, file=name)
        target = dest / name
        expected_size, expected_hash = metadata.get(name, (0, ""))
        size = expected_size or head_size(url)
        if target.exists() and size and target.stat().st_size == size and not expected_hash:
            print(f"  [跳过] {name} 已完整（{human(size)}）")
            continue
        print(f"  [下载] {name}（{human(size) if size else '大小未知'}）")
        if not download(url, target, chunk_mb=chunk_mb, quiet=quiet, sha256=expected_hash):
            failures += 1
    if failures:
        print(f"  {failures} 个文件没下完，重跑本命令会继续")
        return False
    total = sum((dest / name).stat().st_size for name in FILES if (dest / name).exists())
    print(f"  完成：{len(FILES)} 个文件，合计 {human(total)}")
    local = dest / "model.bin"
    if local.exists() and local.stat().st_size != EXPECTED_MODEL_BIN:
        print(f"  注意：model.bin 是 {local.stat().st_size} 字节，官方是 {EXPECTED_MODEL_BIN} 字节")
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 Whisper 语音识别模型（默认走 ModelScope）")
    parser.add_argument("--dest", default=str(ROOT / "models" / "whisper" / "large-v3-turbo"))
    parser.add_argument("--source", choices=sorted(SOURCES), default="modelscope")
    parser.add_argument("--chunk-mb", type=int, default=24)
    args = parser.parse_args(argv)

    if fetch(args.source, Path(args.dest), chunk_mb=args.chunk_mb):
        print("")
        print("  程序会自动识别这个目录（config.yaml 的 stt_model 写 large-v3-turbo 即可）。")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
