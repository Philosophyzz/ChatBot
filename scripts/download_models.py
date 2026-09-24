#!/usr/bin/env python
"""Download GGUF model files and report exactly what happened.

Why a script instead of a documented ``huggingface-cli download`` line: model
repositories get renamed, quantisation filenames drift, and a 16 GB download that
fails at 95% is expensive. This tool therefore

* resolves the **real** filename through the HuggingFace API instead of trusting a
  hard-coded string (and falls back across repositories when one is gone),
* prints the remote file size *before* downloading so the disk-space decision is
  informed,
* resumes a partial file with an HTTP range request (``Ctrl+C`` is safe),
* verifies the final size and reports the sha256,
* and writes a small manifest so the start script knows what is actually present.

Usage::

    python scripts/download_models.py --tier Qwen3.6-14B-A3B-FableVibes-Q4_K_M
    python scripts/download_models.py --tier Qwen3.6-27B-Q4_K_M-mtp --mirror
    python scripts/download_models.py --support          # embedding + reranker
    python scripts/download_models.py --list             # show tiers and sizes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "models.json"
MANIFEST = ROOT / "data" / "models.manifest.json"


def models_root() -> Path:
    sys.path.insert(0, str(ROOT / "src"))
    from core.config import load_config as load_app_config
    return load_app_config(ROOT).paths.models_dir

MIRROR = "https://hf-mirror.com"
CHUNK = 4 * 1024 * 1024

# Corporate/university networks occasionally MITM TLS; allow an explicit opt-out
# rather than making the user edit the script.
_INSECURE = os.environ.get("CHATBOT_INSECURE_TLS") == "1"
_CTX = ssl._create_unverified_context() if _INSECURE else None  # noqa: SLF001


def log(message: str) -> None:
    print(message, flush=True)


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _open(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60):
    request = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(request, timeout=timeout, context=_CTX)  # noqa: S310


def load_config() -> Dict[str, Any]:
    if not CONFIG.exists():
        raise SystemExit(f"找不到 {CONFIG}；请确认项目结构完整。")
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def api_base(mirror: bool) -> str:
    return MIRROR if mirror else "https://huggingface.co"


def resolve_file(repo: str, filename: str, *, mirror: bool, token: Optional[str]) -> Optional[Tuple[str, int]]:
    """Return ``(actual_filename, size_bytes)`` for a file inside a repo.

    Exact match first, then a case-insensitive match, then the closest
    quantisation variant — a repository that renamed ``Q4_K_M`` to ``q4_k_m``
    should not fail the install.
    """
    url = f"{api_base(mirror)}/api/models/{repo}?blobs=true"
    headers = {"User-Agent": "chatbot-setup"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with _open(url, headers) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            log(f"    [x] 仓库不存在: {repo}")
        else:
            log(f"    [x] 查询 {repo} 失败: HTTP {exc.code}")
        return None
    except Exception as exc:  # noqa: BLE001
        log(f"    [x] 查询 {repo} 失败: {exc}")
        return None

    siblings = payload.get("siblings") or []
    entries = [(item.get("rfilename"), item.get("size")) for item in siblings if item.get("rfilename")]

    for name, size in entries:
        if name == filename:
            return name, int(size or 0)
    lowered = filename.lower()
    for name, size in entries:
        if name.lower() == lowered:
            return name, int(size or 0)
    # Same quantisation, any naming convention.
    stem = lowered.replace(".gguf", "")
    quant = stem.split("-")[-1]
    for name, size in entries:
        if name.lower().endswith(".gguf") and quant in name.lower():
            return name, int(size or 0)
    return None


def download(url: str, target: Path, *, token: Optional[str], expected: int = 0) -> bool:
    """Chunked download with resume, progress and size verification."""
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.stat().st_size if target.exists() else 0
    if expected and existing == expected:
        log(f"    已完整存在，跳过（{human(existing)}）")
        return True
    if expected and existing > expected:
        log("    本地文件比远端大，删除后重新下载")
        target.unlink(missing_ok=True)
        existing = 0

    headers = {"User-Agent": "chatbot-setup"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        log(f"    断点续传：从 {human(existing)} 继续")

    mode = "ab" if existing else "wb"
    started = time.time()
    try:
        with _open(url, headers, timeout=120) as response:
            total = expected or int(response.headers.get("Content-Length") or 0) + existing
            done = existing
            if not existing and response.status != 200:
                log(f"    [x] 意外的 HTTP 状态: {response.status}")
                return False
            with open(target, mode) as handle:
                last_report = 0.0
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last_report > 1.0:
                        last_report = now
                        elapsed = max(0.01, now - started)
                        speed = (done - existing) / elapsed
                        percent = (done / total * 100) if total else 0
                        remaining = ((total - done) / speed) if speed > 0 and total else 0
                        sys.stdout.write(
                            f"\r    {percent:5.1f}%  {human(done)}/{human(total)}  "
                            f"{human(speed)}/s  剩余 {int(remaining)}s   "
                        )
                        sys.stdout.flush()
        sys.stdout.write("\r" + " " * 78 + "\r")
        actual = target.stat().st_size
        if expected and actual != expected:
            log(f"    [x] 大小不符：期望 {human(expected)}，实际 {human(actual)}")
            return False
        log(f"    完成：{human(actual)}，用时 {int(time.time() - started)}s")
        return True
    except KeyboardInterrupt:
        log("\n    已中断。重新运行本脚本会自动续传。")
        return False
    except Exception as exc:  # noqa: BLE001
        log(f"\n    [x] 下载失败：{exc}")
        return False


def sha256_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_url(repo: str, filename: str, *, mirror: bool) -> str:
    return f"{api_base(mirror)}/{repo}/resolve/main/{filename}?download=true"


def ensure_free_space(target_dir: Path, needed_bytes: int) -> bool:
    try:
        free = shutil.disk_usage(target_dir).free
    except OSError:
        return True
    if free < needed_bytes * 1.15:
        log(
            f"    [x] 磁盘空间不足：需要约 {human(needed_bytes * 1.15)}，"
            f"可用 {human(free)}"
        )
        return False
    log(f"    磁盘可用 {human(free)}，本次需要约 {human(needed_bytes)}")
    return True


def fetch_model(spec: Dict[str, Any], *, mirror: bool, token: Optional[str], tier: str) -> Optional[Dict[str, Any]]:
    """Try the primary repo then each fallback; return manifest entry on success."""
    candidates: List[Dict[str, str]] = [{"repo": spec["repo"], "file": spec["file"]}]
    candidates.extend(spec.get("fallback_repos") or [])

    for candidate in candidates:
        repo, filename = candidate["repo"], candidate["file"]
        log(f"  仓库 {repo}")
        resolved = resolve_file(repo, filename, mirror=mirror, token=token)
        if not resolved:
            continue
        actual, size = resolved
        if actual != filename:
            log(f"    文件名已变更为 {actual}")
        target = models_root() / "gguf" / spec["local_name"]
        expected = size or int(float(spec.get("size_gb") or 0) * 1024 ** 3)
        if not ensure_free_space(target.parent, expected):
            return None
        log(f"    目标 {target}（{human(expected)}）")
        url = file_url(repo, actual, mirror=mirror)
        if download(url, target, token=token, expected=size or 0):
            log("    计算 sha256…")
            digest = sha256_of(target)
            expected_sha = str(spec.get("sha256") or "").lower()
            if expected_sha and digest.lower() != expected_sha:
                log(f"    [x] sha256 校验失败：期望 {expected_sha}，实际 {digest}；删除损坏文件")
                target.unlink(missing_ok=True)
                return None
            companions = []
            companion_name = str(spec.get("mmproj_local_name") or "")
            companion_file = str(spec.get("mmproj_file") or "")
            if companion_name and companion_file:
                companion_repo = str(spec.get("mmproj_repo") or repo)
                log(f"  查询图像投影文件 {companion_repo}/{companion_file}")
                companion = resolve_file(companion_repo, companion_file, mirror=mirror, token=token)
                if companion is None:
                    log("    [x] 没找到与此量化匹配的图像投影文件")
                    return None
                companion_actual, companion_size = companion
                companion_target = target.parent / companion_name
                if not ensure_free_space(companion_target.parent, companion_size):
                    return None
                companion_url = file_url(companion_repo, companion_actual, mirror=mirror)
                if not download(companion_url, companion_target, token=token, expected=companion_size):
                    return None
                log("    校验图像投影 sha256…")
                companion_digest = sha256_of(companion_target)
                expected_companion_sha = str(spec.get("mmproj_sha256") or "").lower()
                if expected_companion_sha and companion_digest.lower() != expected_companion_sha:
                    log(f"    [x] 图像投影 sha256 校验失败；删除损坏文件")
                    companion_target.unlink(missing_ok=True)
                    return None
                companions.append({"repo": companion_repo, "file": companion_actual,
                                   "local_name": companion_name, "size_bytes": companion_target.stat().st_size,
                                   "sha256": companion_digest, "path": companion_target.as_posix()})
            return {
                "id": spec["id"],
                "tier": tier,
                "repo": repo,
                "file": actual,
                "local_name": spec["local_name"],
                "path": target.as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": digest,
                "companions": companions,
                "n_gpu_layers": spec.get("n_gpu_layers"),
                "ctx": spec.get("ctx"),
                "port": spec.get("port"),
                "server_args": spec.get("server_args") or [],
                "downloaded_at": time.time(),
            }
        log("    该仓库下载失败，尝试下一个候选")
    return None


def write_manifest(entry: Dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if MANIFEST.exists():
        try:
            data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    models = data.setdefault("models", {})
    models[entry["id"]] = entry
    data["updated_at"] = time.time()
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_tiers(config: Dict[str, Any]) -> None:
    log("对话模型档位：")
    for tier in config["chat_tiers"]:
        marker = "*" if tier["id"] == config.get("default_tier") else " "
        log(f" {marker} {tier['id']:<14} {tier['size_gb']:>5}GB  {tier['label']}")
        log(f"     {tier['notes']}")
    log("")
    log("辅助模型：")
    for spec in config["support_models"]:
        log(f"   {spec['id']:<12} {spec.get('size_gb', 0):>5}GB  {spec['label']}")
    log("")
    log("用法：python scripts/download_models.py --tier Qwen3.6-14B-A3B-FableVibes-Q4_K_M --support")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="下载 GGUF 模型到项目 models 目录")
    parser.add_argument("--tier", default=None, help="对话模型档位 id（见 --list）")
    parser.add_argument("--support", action="store_true", help="同时下载嵌入与重排模型")
    parser.add_argument("--all-tiers", action="store_true", help="下载全部对话档位（很占空间）")
    parser.add_argument("--mirror", action="store_true", help="使用 hf-mirror.com 镜像（国内推荐）")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HuggingFace token（可选）")
    parser.add_argument("--list", action="store_true", help="列出可选档位")
    args = parser.parse_args(argv)

    config = load_config()
    if args.list:
        list_tiers(config)
        return 0

    tier_ids = []
    if args.all_tiers:
        tier_ids = [tier["id"] for tier in config["chat_tiers"]]
    elif args.tier:
        tier_ids = [args.tier]
    if not tier_ids and not args.support:
        tier_ids = [config.get("default_tier") or config["chat_tiers"][0]["id"]]
        log(f"未指定档位，使用默认：{tier_ids[0]}")

    tiers = {tier["id"]: tier for tier in config["chat_tiers"]}
    for tier_id in tier_ids:
        if tier_id not in tiers:
            log(f"[x] 未知档位：{tier_id}（可选：{', '.join(tiers)}）")
            return 2

    log("")
    log(f"下载源：{'hf-mirror.com（镜像）' if args.mirror else 'huggingface.co'}")
    log(f"目标目录：{models_root() / 'gguf'}")
    if not args.mirror:
        log("提示：国内网络若很慢或超时，加上 --mirror 参数。")
    log("")

    failures: List[str] = []
    for tier_id in tier_ids:
        spec = tiers[tier_id]
        log(f"== 对话模型 [{tier_id}] {spec['label']}")
        entry = fetch_model(spec, mirror=args.mirror, token=args.token, tier=tier_id)
        if entry:
            entry["id"] = f"chat-{tier_id}"
            write_manifest(entry)
            log(f"   [OK] 已登记到 models.manifest.json")
        else:
            failures.append(f"chat:{tier_id}")
            log(f"   [x] {tier_id} 下载失败")
        log("")

    if args.support:
        for spec in config["support_models"]:
            if spec.get("kind") == "python":
                # Whisper / TTS weights are fetched by their own runtime on first use.
                log(f"== 跳过 {spec['id']}（由 Python/ModelScope 侧自动下载）")
                log(f"   {spec.get('notes', '')}")
                continue
            log(f"== 辅助模型 [{spec['id']}] {spec['label']}")
            entry = fetch_model(spec, mirror=args.mirror, token=args.token, tier="support")
            if entry:
                write_manifest(entry)
                log("   [OK] 已登记到 models.manifest.json")
            else:
                log(
                    f"   [!] {spec['id']} 下载失败。该项为可选："
                    f"{spec.get('notes', '')}"
                )
            log("")

    log("")
    if failures:
        log(f"以下模型未能下载：{', '.join(failures)}")
        log("可换镜像重试：--mirror；或手动下载后放入 配置的模型目录并保持文件名一致。")
        return 1
    log("全部完成。下一步：")
    log("  powershell -ExecutionPolicy Bypass -File scripts\\verify.ps1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
