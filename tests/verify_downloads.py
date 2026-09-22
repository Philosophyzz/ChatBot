"""Verify downloaded GGUF files are complete and structurally valid.

``download-models.ps1`` writes ``data/models.manifest.json`` only when a download
finishes cleanly, so a manifest entry is one signal — but it is missing whenever the
process was interrupted after the bytes landed (Ctrl+C during the tail, a closed
window, a re-run that resumed). The other signal is the bytes themselves, which is what
this script checks:

1. **Structure** — GGUF magic, version, tensor count, metadata count, and a
   plausibility check on those numbers. Catches a file that is not a model at all
   (e.g. an HTML error page saved under a .gguf name).
2. **Size** — compared against the authoritative ``Content-Length`` from the remote
   (through the mirror when ``--mirror`` is passed). A short file is a truncated
   download and must be resumed, not loaded: llama.cpp will fail mid-load, often
   minutes in, with a confusing error.
3. **Tail** — a truncated file almost always ends in zero padding. A tail of all zeros
   is reported as suspicious even when the size check is unavailable.

Usage::

    python tests/verify_downloads.py                # check models/gguf
    python tests/verify_downloads.py --mirror       # use hf-mirror.com for sizes
    python tests/verify_downloads.py --repair       # register intact files in the manifest
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
GGUF_DIR = ROOT / "models" / "gguf"
CONFIG = ROOT / "config" / "models.json"
MANIFEST = ROOT / "data" / "models.manifest.json"
MIRROR = "https://hf-mirror.com"


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"


def read_gguf_header(path: Path) -> Optional[Dict[str, Any]]:
    """Parse the fixed GGUF header. Returns None when the file is not GGUF."""
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
            if magic != b"GGUF":
                return None
            version, n_tensors, n_kv = struct.unpack("<IQQ", handle.read(20))
    except (OSError, struct.error):
        return None
    return {"version": version, "tensors": n_tensors, "metadata": n_kv}


def tail_is_zero_padding(path: Path, chunk: int = 4096) -> bool:
    """True when the last ``chunk`` bytes are entirely zero (truncation signature)."""
    size = path.stat().st_size
    if size < chunk:
        return False
    with path.open("rb") as handle:
        handle.seek(size - chunk)
        return not any(handle.read(chunk))


def remote_size(repo: str, filename: str, *, mirror: bool) -> Optional[int]:
    """Authoritative file size from the model repository listing."""
    base = MIRROR if mirror else "https://huggingface.co"
    url = f"{base}/api/models/{repo}?blobs=true"
    request = urllib.request.Request(url, headers={"User-Agent": "chatbot-verify"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"      [i] 无法查询远端大小（{type(exc).__name__}）：{exc}")
        return None
    siblings = payload.get("siblings") or []
    for item in siblings:
        if item.get("rfilename") == filename:
            return int(item.get("size") or 0) or None
    lowered = filename.lower()
    for item in siblings:
        if str(item.get("rfilename", "")).lower() == lowered:
            return int(item.get("size") or 0) or None
    return None


def load_specs() -> Dict[str, Dict[str, Any]]:
    """Map local file name -> {id, tier, repo, file, expected_bytes}."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    specs: Dict[str, Dict[str, Any]] = {}
    for tier in config.get("chat_tiers") or []:
        specs[tier["local_name"]] = {
            "id": f"chat-{tier['id']}",
            "tier": tier["id"],
            "repo": tier["repo"],
            "file": tier["file"],
            "fallback_repos": tier.get("fallback_repos") or [],
            "declared_gb": tier.get("size_gb"),
        }
    for spec in config.get("support_models") or []:
        if spec.get("kind") == "python":
            continue
        specs[spec["local_name"]] = {
            "id": spec["id"],
            "tier": "support",
            "repo": spec["repo"],
            "file": spec["file"],
            "fallback_repos": spec.get("fallback_repos") or [],
            "declared_gb": spec.get("size_gb"),
        }
    return specs


def check_one(path: Path, spec: Optional[Dict[str, Any]], *, mirror: bool) -> Tuple[str, List[str]]:
    """Return ``(status, notes)`` where status is ok | truncated | invalid."""
    notes: List[str] = []
    size = path.stat().st_size
    print(f"  {path.name}")
    print(f"    本地大小  : {size:,} 字节 ({human(size)})")

    header = read_gguf_header(path)
    if header is None:
        notes.append("不是合法的 GGUF 文件（魔数不对）")
        print("    [X] 魔数不是 GGUF —— 可能是错误信息被保存成了文件")
        return "invalid", notes
    print(
        f"    结构      : GGUF v{header['version']}  "
        f"{header['tensors']} 个张量  {header['metadata']} 项元数据"
    )
    if not (header["version"] in (2, 3) and 50 < header["tensors"] < 5000 and 3 < header["metadata"] < 500):
        notes.append("头部数值不合理")
        print("    [!] 头部数值异常，文件可能损坏")

    expected: Optional[int] = None
    if spec:
        for candidate in [{"repo": spec["repo"], "file": spec["file"]}, *spec["fallback_repos"]]:
            expected = remote_size(candidate["repo"], candidate["file"], mirror=mirror)
            if expected:
                print(f"    远端大小  : {expected:,} 字节 ({human(expected)})  [{candidate['repo']}]")
                break
    if expected:
        delta = expected - size
        if delta == 0:
            print("    [OK] 大小与远端完全一致")
        elif delta > 0:
            pct = delta / expected * 100
            notes.append(f"缺少 {human(delta)}（{pct:.1f}%），下载未完成")
            print(f"    [X] 缺少 {human(delta)}（{pct:.1f}%）—— 下载被截断，需续传")
            return "truncated", notes
        else:
            notes.append(f"比远端大 {human(-delta)}")
            print(f"    [!] 比远端大 {human(-delta)}，可能重复写入")
    else:
        declared = spec.get("declared_gb") if spec else None
        if declared:
            approx = float(declared) * 1024 ** 3
            ratio = size / approx if approx else 0
            print(f"    参考值    : 配置声明 {declared}GB，实际/声明 = {ratio:.2%}")
            if ratio < 0.97:
                notes.append(f"小于配置声明值（{ratio:.1%}），可能未下载完")
                print("    [!] 明显小于声明值，可能未完成")

    if tail_is_zero_padding(path):
        notes.append("末尾为全零填充，典型的截断特征")
        print("    [!] 末尾 4KB 全为零 —— 截断特征")
        if not any("截断" in note for note in notes):
            return "truncated", notes
    else:
        print("    [OK] 末尾有有效数据（非全零填充）")

    if notes and any("截断" in note for note in notes):
        return "truncated", notes
    print("    [OK] 结构与大小检查通过")
    return "ok", notes


def update_manifest(entries: List[Dict[str, Any]]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if MANIFEST.exists():
        try:
            data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    models = data.setdefault("models", {})
    for entry in entries:
        models[entry["id"]] = entry
    data["updated_at"] = time.time()
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="校验已下载的 GGUF 文件完整性")
    parser.add_argument("--mirror", action="store_true", help="通过 hf-mirror.com 查询远端大小")
    parser.add_argument("--repair", action="store_true", help="把完整的文件登记进 manifest")
    parser.add_argument("--dir", default=str(GGUF_DIR), help="要检查的目录")
    args = parser.parse_args(argv)

    directory = Path(args.dir)
    print("=" * 64)
    print(" GGUF 下载完整性校验")
    print(f" 目录: {directory}")
    print(f" 查询源: {'hf-mirror.com（镜像）' if args.mirror else 'huggingface.co'}")
    print("=" * 64)

    files = sorted(directory.glob("*.gguf")) if directory.exists() else []
    if not files:
        print("\n  目录下没有 .gguf 文件。先运行 scripts\\download-models.ps1")
        return 1

    specs = load_specs()
    results: List[Tuple[Path, str, List[str]]] = []
    for path in files:
        print("")
        status, notes = check_one(path, specs.get(path.name), mirror=args.mirror)
        results.append((path, status, notes))

    ok = [item for item in results if item[1] == "ok"]
    truncated = [item for item in results if item[1] == "truncated"]
    invalid = [item for item in results if item[1] == "invalid"]

    print("")
    print("=" * 64)
    print(f" 完整 {len(ok)} 个 / 截断 {len(truncated)} 个 / 无效 {len(invalid)} 个")
    for path, _status, notes in truncated + invalid:
        print(f"   - {path.name}: {'; '.join(notes)}")
    if truncated:
        print("")
        print("  修复：重新运行同一条下载命令即可断点续传，不会从头开始：")
        tier = specs.get(truncated[0][0].name, {}).get("tier")
        hint = f" -Tier {tier}" if tier and tier != "support" else ""
        print(f"      powershell -ExecutionPolicy Bypass -File scripts\\download-models.ps1 -Mirror{hint}")
    print("=" * 64)

    if args.repair and ok:
        entries = []
        for path, _status, _notes in ok:
            spec = specs.get(path.name)
            if not spec:
                continue
            entries.append(
                {
                    "id": spec["id"],
                    "tier": spec["tier"],
                    "repo": spec["repo"],
                    "file": spec["file"],
                    "local_name": path.name,
                    "path": f"models/gguf/{path.name}",
                    "size_bytes": path.stat().st_size,
                    "verified_at": time.time(),
                    "verified_by": "tests/verify_downloads.py",
                }
            )
        if entries:
            update_manifest(entries)
            print(f" 已把 {len(entries)} 个完整文件登记到 {MANIFEST.relative_to(ROOT)}")

    return 1 if (truncated or invalid) else 0


if __name__ == "__main__":
    sys.exit(main())
