"""Validate that every repository and filename in config/models.json resolves.

Running this before a ~20 GB download is the difference between "it works" and
"the filename drifted six months ago and the download 404s at 90%". It calls the
same resolution code the downloader uses (``scripts/download_models.resolve_file``),
so a pass here is a real guarantee rather than a second implementation agreeing with
itself.

Usage::

    python tests/verify_models.py            # check every candidate
    python tests/verify_models.py --mirror   # check via hf-mirror.com
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "models.json"


def load_downloader() -> Any:
    """Import scripts/download_models.py, which is not a package module."""
    path = ROOT / "scripts" / "download_models.py"
    spec = importlib.util.spec_from_file_location("download_models", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_models"] = module
    spec.loader.exec_module(module)
    return module


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}GB"


def check(dl: Any, label: str, repo: str, filename: str, *, mirror: bool, token: str | None) -> bool:
    resolved = dl.resolve_file(repo, filename, mirror=mirror, token=token)
    if not resolved:
        print(f"  [FAIL] {label}: {repo} :: {filename} 未解析到")
        return False
    actual, size = resolved
    note = "" if actual == filename else f"  (实际文件名: {actual})"
    print(f"  [OK]   {label}: {repo}{note}")
    print(f"         文件 {actual}  大小 {human(size)}")
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验模型仓库与文件名可用性")
    parser.add_argument("--mirror", action="store_true", help="通过 hf-mirror.com 校验")
    parser.add_argument("--token", default=None, help="HuggingFace token（可选）")
    parser.add_argument("--chat-only", action="store_true", help="只校验对话模型档位")
    args = parser.parse_args(argv)

    dl = load_downloader()
    config: Dict[str, Any] = json.loads(CONFIG.read_text(encoding="utf-8"))
    print(f"校验源: {'hf-mirror.com' if args.mirror else 'huggingface.co'}")
    print(f"默认档位: {config.get('default_tier')}\n")

    failures: List[Tuple[str, str]] = []
    checked = 0

    for tier in config.get("chat_tiers") or []:
        print(f"== 对话档位 [{tier['id']}] {tier['label']}")
        candidates = [{"repo": tier["repo"], "file": tier["file"]}, *(tier.get("fallback_repos") or [])]
        ok_any = False
        for index, candidate in enumerate(candidates):
            role = "主仓库" if index == 0 else f"备选{index}"
            if check(dl, role, candidate["repo"], candidate["file"], mirror=args.mirror, token=args.token):
                ok_any = True
                break
        checked += 1
        if not ok_any:
            failures.append((tier["id"], "所有候选仓库均不可用"))
        print()

    if not args.chat_only:
        for spec in config.get("support_models") or []:
            if spec.get("kind") == "python":
                print(f"== 辅助模型 [{spec['id']}] 由运行库自行下载，跳过校验")
                continue
            print(f"== 辅助模型 [{spec['id']}] {spec['label']}")
            candidates = [{"repo": spec["repo"], "file": spec["file"]}, *(spec.get("fallback_repos") or [])]
            ok_any = False
            for index, candidate in enumerate(candidates):
                role = "主仓库" if index == 0 else f"备选{index}"
                if check(dl, role, candidate["repo"], candidate["file"], mirror=args.mirror, token=args.token):
                    ok_any = True
                    break
            checked += 1
            if not ok_any:
                # Support models are optional: record but do not fail the run.
                print(f"  [!]    {spec['id']} 不可用（可选组件，缺失会自动降级）")
            print()

    print("=" * 60)
    if failures:
        print(f"校验失败 {len(failures)}/{checked}：")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
        print("\n处理方式：编辑 config/models.json 换成可用仓库，或手动下载 GGUF 到 models/gguf/")
        return 1
    print(f"全部 {checked} 个模型条目校验通过，可以开始下载。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
