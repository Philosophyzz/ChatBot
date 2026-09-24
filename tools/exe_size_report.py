"""看看 PyInstaller 到底把什么塞进了 exe —— 体积失控时先跑这个。

出过一次 3.3GB 的桌宠 exe：`src\\plugs\\__init__.py` 为了注册内置插件会 import
`speech.stt` / `speech.tts`，PyInstaller 顺着这条静态边把 torch 整套 CUDA 运行库也打了进来
（光 `torch_cuda.dll` 就 981MB）。体积冒烟在 `scripts\\build-pet.ps1` 里已经卡了 400MB 上限，
真被卡住时就用这个工具看是哪几个包在膨胀：

    python tools\\exe_size_report.py                 # 读 build\\启动聊天机器人\\Analysis-00.toc
    python tools\\exe_size_report.py --top 40
    python tools\\exe_size_report.py --contains torch

输出：条目总大小、体积最大的若干条（二进制 / 数据 / 模块），以及几个常见"重家伙"是否被打进去。
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

#: PyInstaller 里最容易被误打包的推理依赖，逐一报个数。
SUSPECTS = (
    "torch", "transformers", "faster_whisper", "ctranslate2", "onnxruntime",
    "cv2", "llvmlite", "numba", "scipy", "modelscope", "librosa", "accelerate",
)


def _mentions(path: str, suspect: str) -> bool:
    """这个路径里是不是**真的**有这个包（按路径分段判断，不做子串匹配）。

    子串匹配会误报：`numpy.libs\\libscipy_openblas64_….dll` 里含 "scipy"，但它其实属于 numpy。
    """
    for segment in re.split(r"[\\/]", path.lower()):
        if segment == suspect or re.match(rf"^{re.escape(suspect)}[.\-]", segment):
            return True
    return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="分析 PyInstaller 打包产物里的大块头")
    parser.add_argument("--toc", default=r"build\启动聊天机器人\Analysis-00.toc", help="Analysis TOC 路径")
    parser.add_argument("--top", type=int, default=25, help="列出体积最大的多少条")
    parser.add_argument("--contains", default=None, help="只看路径里含这段文字的条目")
    args = parser.parse_args(argv)

    toc_path = Path(args.toc)
    if not toc_path.exists():
        print(f"找不到 {toc_path} —— 先跑一次 scripts\\build-pet.ps1（分析结果就在这里）")
        return 1

    toc = ast.literal_eval(toc_path.read_text(encoding="utf-8"))
    rows: List[tuple] = []
    for index, element in enumerate(toc):
        if not isinstance(element, (list, tuple)):
            continue
        for entry in element:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            path = str(entry[1])
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            rows.append((size, index, str(entry[0])[:70], path))

    total = sum(row[0] for row in rows)
    print(f"打包条目 {len(rows)} 个，合计 {total / 1048576:.0f} MB（压缩前）")
    print(f"成品种类：二进制 {len(toc[15]) if len(toc) > 15 else '?'} 条 / "
          f"数据 {len(toc[18]) if len(toc) > 18 else '?'} 条")

    if args.contains:
        rows = [row for row in rows if args.contains.lower() in row[3].lower()]
        print(f"（只看含「{args.contains}」的 {len(rows)} 条）")

    rows.sort(reverse=True)
    print(f"\n体积最大的 {args.top} 条：")
    for size, index, name, path in rows[: args.top]:
        print(f"{size / 1048576:9.1f}MB  toc[{index}]  {name:<70} {path[:80]}")

    print("\n可疑的推理依赖：")
    for suspect in SUSPECTS:
        hits = [row for row in rows if _mentions(row[3], suspect)]
        if hits:
            size = sum(row[0] for row in hits)
            print(f"  [打包进来了] {suspect:<16} {len(hits):>4} 条 / {size / 1048576:8.1f} MB")
    print("  （上面没列出的，就是被 --exclude-module 挡在外面了）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
