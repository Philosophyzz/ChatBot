"""Resumable chunked downloader for large model files.

Why this exists: ``huggingface_hub``/``hf`` download a big file over a single connection
and, through the mirror used here, that connection reliably dies around 40MB — leaving a
``.incomplete`` file that never finishes (measured: 1.5MB/s on a 32MB range request, but a
1.6GB whole-file download stalls and dies). Fetching the file in explicit byte ranges and
appending each one to disk survives that: every range is an independent request, so a
dropped connection costs one chunk, not the whole download.

Usage::

    python tools/download_file.py --url <url> --out models/whisper/.../model.bin
    python tools/download_file.py --url <url> --out out.bin --chunk-mb 16 --retries 50

Re-running after an interruption continues from the current file size.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# The system proxy (Clash on 127.0.0.1:7890) must not be used for these hosts; going
# through it produced connection resets in testing.
NO_PROXY_ENV = {"NO_PROXY": "*", "no_proxy": "*"}


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def head_size(url: str, *, timeout: float = 30.0) -> Optional[int]:
    """Content-Length via a GET with Range 0-0 (HEAD is often rejected by mirrors)."""
    import httpx

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, trust_env=False) as client:
            response = client.get(url, headers={"Range": "bytes=0-0"})
            content_range = response.headers.get("content-range")
            if content_range and "/" in content_range:
                total = content_range.rsplit("/", 1)[1]
                if total.isdigit():
                    return int(total)
            if response.status_code == 200 and response.headers.get("content-length"):
                return int(response.headers["content-length"])
    except Exception as exc:  # noqa: BLE001
        print(f"  无法获取文件大小：{type(exc).__name__}: {exc}")
    return None


def sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    url: str,
    out: Path,
    *,
    chunk_mb: int = 32,
    retries: int = 40,
    timeout_s: float = 120.0,
    quiet: bool = False,
    sha256: str = "",
) -> bool:
    """Fetch ``url`` into ``out`` in byte ranges, resuming from whatever exists.

    Two details that were learned the hard way:

    * The download goes to ``<out>.part`` and is renamed only after the length (and, when
      known, the sha256) matches. With a half-written ``model.bin`` in the target
      directory another downloader (``hf download``) tries to delete it and hits
      ``PermissionError: [WinError 32]`` on Windows; a ``.part`` file is never mistaken
      for a finished model either.
    * An existing ``out`` is only adopted as a resume point when it is **shorter** than
      the expected total. A file that is *longer* came from a different source/revision
      (``tokenizer.json`` was 2,710,337 bytes from one repo and 2,480,645 from another);
      appending to it produces a corrupt model that fails much later with an unrelated
      looking parse error. Such a file is moved aside instead.
    """
    import httpx

    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    total = head_size(url)

    if out.exists():
        existing = out.stat().st_size
        if total and existing == total and not part.exists():
            # Already complete — verify by hash when we can, then short-circuit.
            if not sha256 or sha256_of(out).lower() == sha256.lower():
                if not quiet:
                    print(f"  已存在且大小正确，跳过：{out.name}（{human(existing)}）")
                return True
            print(f"  {out.name} 大小对但 sha256 不符，重新下载")
            part.unlink(missing_ok=True)
        elif not part.exists():
            if total and existing < total:
                out.rename(part)
                if not quiet:
                    print(f"  接管已有的 {human(existing)} 半成品，改名 {part.name}")
            else:
                aside = out.with_name(out.name + ".replaced")
                out.replace(aside)
                print(
                    f"  {out.name} 是 {human(existing)}，与本次来源的 "
                    f"{human(total) if total else '未知大小'} 不一致，已移到 {aside.name} 并重新下载"
                )

    if total:
        print(f"  目标 {out}")
        print(f"  大小 {human(total)}")
    chunk = max(1, chunk_mb) * 1024 * 1024
    started = time.time()

    with httpx.Client(timeout=timeout_s, follow_redirects=True, trust_env=False) as client:
        attempt = 0
        stalled = 0
        while True:
            have = part.stat().st_size if part.exists() else 0
            if total and have >= total:
                break
            end = min(have + chunk - 1, (total - 1) if total else have + chunk - 1)
            headers = {"Range": f"bytes={have}-{end}"}
            try:
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {response.status_code}")
                    before = have
                    with part.open("ab") as handle:
                        for block in response.iter_bytes(1024 * 256):
                            handle.write(block)
                            have += len(block)
                    if have == before:  # server returned nothing for this range
                        stalled += 1
                        if stalled > 5:
                            raise RuntimeError("连续多次没有收到数据")
                    else:
                        stalled = 0
                    attempt = 0
                    if not quiet and total:
                        elapsed = max(0.001, time.time() - started)
                        speed = have / elapsed
                        percent = have / total * 100
                        eta = (total - have) / speed if speed > 0 else 0
                        print(
                            f"\r  {percent:5.1f}%  {human(have)}/{human(total)}  "
                            f"{human(speed)}/s  剩余 {eta/60:.1f} 分钟",
                            end="",
                            flush=True,
                        )
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > retries:
                    print(f"\n  放弃：{type(exc).__name__}: {exc}")
                    print(f"  已下载的部分保留在 {part}，重跑本命令可续传")
                    return False
                wait = min(10.0, 1.0 + attempt * 0.5)
                if not quiet:
                    print(f"\n  第 {attempt} 次失败（{type(exc).__name__}），{wait:.0f}s 后续传…")
                time.sleep(wait)

    if not quiet:
        print("")
    size = part.stat().st_size if part.exists() else 0
    if total and size < total:
        print(f"  不完整：{human(size)} / {human(total)}（保留 {part.name}，可续传）")
        return False
    if sha256:
        actual = sha256_of(part)
        if actual.lower() != sha256.lower():
            print(f"  sha256 校验失败：本地 {actual[:16]}… / 期望 {sha256[:16]}…（保留 {part.name}，请重下）")
            return False
        if not quiet:
            print(f"  sha256 校验通过（{actual[:16]}…）")
    try:
        part.replace(out)
    except OSError as exc:
        print(f"  下载完成但改名失败（{exc}）：文件在 {part}")
        return False
    print(f"  完成：{out}（{human(out.stat().st_size)}，用时 {(time.time()-started)/60:.1f} 分钟）")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="断点续传的大文件下载器（分块 Range 请求）")
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk-mb", type=int, default=32, help="每块大小（越小越抗断线）")
    parser.add_argument("--retries", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    for key, value in NO_PROXY_ENV.items():
        os.environ.setdefault(key, value)

    out = Path(args.out)
    if out.exists() and out.stat().st_size:
        print(f"  已存在 {human(out.stat().st_size)}，将续传")
    ok = download(args.url, out, chunk_mb=args.chunk_mb, retries=args.retries, timeout_s=args.timeout)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
