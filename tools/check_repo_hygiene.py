"""检查"准备上传到 GitHub 的东西里有没有不该有的"。

为什么需要它：这个项目的模型有 33.6GB、vendor 5.8GB，而 data/ 里是**用户真实的对话与
记忆库**。推上去之后再撤回是撤不干净的（还会留在 GitHub 的缓存与 fork 里），所以在
commit 之前用脚本把边界钉死，而不是靠记性。

    python tools\\check_repo_hygiene.py            # 检查已暂存的内容（git add 之后）
    python tools\\check_repo_hygiene.py --list      # 顺便列出将上传的文件

退出码 0 = 可以放心提交；1 = 有不该上传的东西或体积异常。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

#: 绝不能出现在提交里的路径前缀（相对仓库根）。
FORBIDDEN_PREFIXES = (
    "data/",
    "logs/",
    "models/",
    "vendor/",
    "bin/",
    "venvs/",
    "build/",
    "dist/",
    ".agent-teams/",
    "__pycache__/",
)
#: 绝不能出现的具体文件（本机配置/密钥/数据库）。
FORBIDDEN_FILES = ("config/local.yaml", ".env")
#: 绝不能出现的后缀（模型权重、日志、对话记录、数据库）。
FORBIDDEN_SUFFIXES = (".gguf", ".safetensors", ".pth", ".ckpt", ".onnx", ".part", ".sqlite3", ".jsonl", ".log")

#: 单个文件上限：GitHub 在 100MB 处硬性拒绝，50MB 就会警告。仓库本身应远小于此。
MAX_FILE_MB = 50.0
#: 整个仓库（未压缩）的合理上限：代码 + 文档几 MB 足矣。
MAX_TOTAL_MB = 30.0

#: .gitignore 必须覆盖的关键规则（少了任何一条都要报错）。
REQUIRED_IGNORES = (
    "models/",
    "vendor/",
    "bin/",
    "data/",
    "logs/",
    "venvs/",
    "build/",
    "dist/",
    "config/local.yaml",
)


def git(*args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed command
        ["git", *args], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if completed.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败：{completed.stderr.strip()}")
    return completed.stdout


def staged_files() -> List[str]:
    output = git("diff", "--cached", "--name-only", "-z")
    return [name for name in output.split("\0") if name]


def check_gitignore() -> List[str]:
    path = ROOT / ".gitignore"
    if not path.exists():
        return [".gitignore 不见了 —— 没有它就会把 models/ 和 data/ 一起推上去"]
    text = path.read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines()}
    return [f".gitignore 缺少关键规则：{rule}" for rule in REQUIRED_IGNORES if rule not in lines]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="检查待上传内容是否干净")
    parser.add_argument("--list", action="store_true", help="列出将上传的文件")
    args = parser.parse_args(argv)

    problems: List[str] = list(check_gitignore())

    try:
        files = staged_files()
    except SystemExit as exc:
        print(f"[!] {exc}")
        print("    先在仓库根目录执行：git init && git add -A")
        return 1

    if not files:
        print("[!] 没有暂存任何文件 —— 先 git add -A")
        return 1

    forbidden: List[str] = []
    total = 0.0
    biggest: List[Tuple[float, str]] = []
    for name in files:
        posix = name.replace("\\", "/")
        full = ROOT / posix
        size_mb = (full.stat().st_size / 1024 / 1024) if full.exists() else 0.0
        total += size_mb
        biggest.append((size_mb, posix))
        if posix in FORBIDDEN_FILES or posix.startswith(FORBIDDEN_PREFIXES):
            forbidden.append(posix)
        elif posix.endswith(FORBIDDEN_SUFFIXES) and not posix.endswith(".example"):
            forbidden.append(posix)
        elif size_mb > MAX_FILE_MB:
            forbidden.append(f"{posix}（{size_mb:.1f}MB，超过 {MAX_FILE_MB}MB 单文件上限）")

    biggest.sort(reverse=True)
    print(f"将上传 {len(files)} 个文件，合计 {total:.2f} MB")
    if args.list or forbidden:
        for size_mb, name in biggest[:20]:
            print(f"  {size_mb:8.2f} MB  {name}")

    if forbidden:
        problems.append("以下内容不该上传：\n    " + "\n    ".join(forbidden[:20]))
    if total > MAX_TOTAL_MB:
        problems.append(f"总体积 {total:.1f}MB 超过 {MAX_TOTAL_MB}MB —— 是不是漏了某条 .gitignore？")

    print()
    if problems:
        print("[X] 先别提交：")
        for item in problems:
            print("  - " + item)
        print("\n  处理办法：确认 .gitignore 覆盖它们，然后 git rm -r --cached <路径> 重新暂存。")
        return 1
    print("[OK] 干净：没有模型、没有日志、没有 data/（你的对话与记忆）、没有本机配置。")
    print("     下一步：git commit -m \"初始提交\" 然后按 docs\\上传GitHub.md 推送到远端。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
