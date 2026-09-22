"""Verify that one conda environment can host both TTS stacks — run this after installing.

Why this exists: ``indextts`` pins ``numpy==2.2.6`` while GPT-SoVITS'
``requirements.txt`` asks for ``numpy<2.0``. Those cannot both be satisfied, so a single
environment is only viable if you install the GPT-SoVITS requirements first and then put
numpy back to 2.2.6 (what ``scripts\\install-gptsovits.ps1`` does). Whether that actually
works cannot be decided from version numbers alone — a compiled extension built against
the numpy 1.x C ABI only fails at *import* time. This script imports everything and says
which side broke, with the consequence spelled out.

Usage::

    conda activate chatbot
    python tools\\check_tts_stack.py

Exit code 0 = both stacks usable; 1 = something is broken (the failing row says what).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
# Windows 控制台默认是 GBK：脚本里的 ✅ / ⚠ / 中文一旦编码不了就 UnicodeEncodeError，
# 检查明明通过了却以非 0 退出 —— 用户会以为环境坏了。统一切到 UTF-8 输出。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 被重定向的流
        pass
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]

#: (import name, pip name, why it matters, required?)
CHECKS: Tuple[Tuple[str, str, str, bool], ...] = (
    ("numpy", "numpy", "两个 TTS 栈的共同基础（版本冲突点就在这）", True),
    ("torch", "torch", "IndexTTS2 与 GPT-SoVITS 都要", True),
    ("torchaudio", "torchaudio", "GPT-SoVITS 读取音频", True),
    ("librosa", "librosa==0.10.2", "GPT-SoVITS 特征提取", True),
    ("pytorch_lightning", "pytorch-lightning", "GPT-SoVITS 训练器", True),
    ("transformers", "transformers", "两个栈都要（版本需满足 4.51~4.99）", True),
    ("soundfile", "soundfile", "音频读写 / 参考音色生成", True),
    ("faster_whisper", "faster-whisper", "语音识别 + 数据集标注", True),
    ("indextts", "indextts（-e vendor\\index-tts）", "IndexTTS2 零样本克隆", True),
    ("funasr", "funasr", "GPT-SoVITS 的 ASR 备选（可用 faster-whisper 代替）", False),
    ("modelscope", "modelscope", "模型下载来源之一", False),
    ("sentencepiece", "sentencepiece", "文本前端", False),
    ("pyopenjtalk", "pyopenjtalk", "日文支持（中文不需要）", False),
    ("opencc", "opencc 或 opencc-python-reimplemented", "中英混排转换", False),
    ("jieba", "jieba", "中文分词（indextts 依赖）", False),
    ("cn2an", "cn2an", "数字转中文", False),
)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="验证单个环境能否同时支撑 IndexTTS2 与 GPT-SoVITS")
    parser.add_argument("--repo", default=str(ROOT / "vendor" / "GPT-SoVITS"))
    args = parser.parse_args(argv)

    print("  解释器:", sys.executable)
    print("  Python :", sys.version.split()[0])
    print("")

    failures: List[str] = []
    warnings: List[str] = []
    print("  === 依赖导入检查 ===")
    for module, pip_name, why, required in CHECKS:
        spec = importlib.util.find_spec(module) if module != "indextts" else None
        if module == "indextts":
            # indextts 是本地 -e 安装，用包内路径判断更可靠
            spec = importlib.util.find_spec("indextts")
        if spec is None:
            mark = "缺失" if required else "可选缺失"
            print(f"    [{mark}] {module:18} {pip_name:34} {why}")
            (failures if required else warnings).append(f"{module}（{pip_name}）")
            continue
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "")
            print(f"    [OK]   {module:18} {str(version):12} {why}")
        except Exception as exc:  # noqa: BLE001 - this is exactly what we are hunting
            print(f"    [导入失败] {module:18} {type(exc).__name__}: {str(exc)[:70]}")
            print(f"              -> {why}")
            (failures if required else warnings).append(f"{module} 导入失败")

    print("")
    print("  === 关键版本 ===")
    try:
        import numpy

        print(f"    numpy {numpy.__version__}（indextts 要求 2.2.6；GPT-SoVITS 写的是 <2.0 但未验证为硬性）")
    except Exception as exc:  # noqa: BLE001
        print(f"    numpy 无法导入: {exc}")
    try:
        import torch

        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "-"
        print(f"    torch {torch.__version__}  cuda={cuda}  {name}")
        if not cuda:
            warnings.append("torch 看不到 CUDA（训练会退到 CPU）")
    except Exception as exc:  # noqa: BLE001
        print(f"    torch 无法导入: {exc}")

    print("")
    print("  === 功能级验证 ===")
    try:
        from indextts.infer_v2 import IndexTTS2  # noqa: F401

        print("    [OK]   indextts.infer_v2.IndexTTS2 可导入（IndexTTS2 可用）")
    except Exception as exc:  # noqa: BLE001
        print(f"    [失败] indextts.infer_v2 导入失败：{type(exc).__name__}: {str(exc)[:70]}")
        failures.append("IndexTTS2 主类不可导入")

    repo = Path(args.repo)
    if repo.exists():
        import subprocess

        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "check_gptsovits.py"), "--repo", str(repo)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        print(f"    {'[OK]  ' if completed.returncode == 0 else '[失败]'} GPT-SoVITS 仓库文件检查"
              f"（exit {completed.returncode}）")
        if completed.returncode != 0:
            failures.append("GPT-SoVITS 仓库文件不齐")
    else:
        print(f"    [跳过] 找不到 GPT-SoVITS 仓库：{repo}")

    print("")
    if warnings:
        print(f"  提醒（{len(warnings)}）：")
        for item in warnings:
            print(f"    - {item}")
        print("")
    if failures:
        print(f"  结论：单环境方案有问题，{len(failures)} 项必需依赖不可用：")
        for item in failures:
            print(f"    - {item}")
        print("")
        print("  怎么办：")
        print("    1) 缺什么装什么（pip install <包名> -i https://pypi.tuna.tsinghua.edu.cn/simple）")
        print("    2) 如果是「导入失败」且提到 numpy：这是 numpy 1.x/2.x ABI 不兼容，")
        print("       同一个环境里无法同时满足两个 TTS 栈 —— 训练改用独立环境：")
        print("         conda create -n chatbot-tts-train python=3.10 -y")
        print("         conda activate chatbot-tts-train")
        print("         python -m pip install -r vendor\\GPT-SoVITS\\requirements.txt")
        return 1
    print("  结论：两个 TTS 栈在同一个环境里都能导入，单环境方案可行 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
