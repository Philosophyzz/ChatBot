"""实测 IndexTTS2：先 CPU 证明安装正确，再试 GPU 看和 9B 模型能否共存。"""

from __future__ import annotations

import os
import sys
# Windows 控制台默认是 GBK：脚本里的 ✅ / ⚠ / 中文一旦编码不了就 UnicodeEncodeError，
# 检查明明通过了却以非 0 退出 —— 用户会以为环境坏了。统一切到 UTF-8 输出。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 被重定向的流
        pass
import time
from pathlib import Path

ROOT = Path(r"D:\Harness\ChatBot")
MODEL_DIR = ROOT / "models" / "tts" / "IndexTTS2"
REFERENCE = ROOT / "models" / "voices" / "gentle_sister.wav"
TEXT = "你好，我是温柔姐姐，现在是用本地模型合成的声音。"


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    fp16 = device.startswith("cuda")
    print(f"  设备={device} fp16={fp16}")
    print(f"  模型目录={MODEL_DIR}")
    print(f"  参考音色={REFERENCE.name}")
    import torch

    if device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info()
        print(f"  当前显存: 空闲 {free/1024/1024:.0f}MB / 总 {total/1024/1024:.0f}MB")

    t0 = time.time()
    from indextts.infer_v2 import IndexTTS2

    print(f"  导入 indextts 用时 {time.time()-t0:.1f}s")
    t0 = time.time()
    tts = IndexTTS2(model_dir=str(MODEL_DIR), cfg_path=None, use_fp16=fp16, device=device)
    print(f"  模型加载完成，用时 {time.time()-t0:.1f}s")
    if device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info()
        print(f"  加载后显存: 空闲 {free/1024/1024:.0f}MB / 总 {total/1024/1024:.0f}MB "
              f"（本次占用约 {(16303 - free/1024/1024):.0f}MB 含桌面）")

    out = ROOT / "data" / f"_indextts_test_{device.replace(':', '')}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    t0 = time.time()
    tts.infer(
        spk_audio_prompt=str(REFERENCE),
        text=TEXT,
        output_path=str(out),
        use_emo_text=True,
        emo_text="用甜美、温柔的语气说",
        emo_alpha=0.9,
        verbose=False,
    )
    elapsed = time.time() - t0
    size = out.stat().st_size if out.exists() else 0
    print(f"  合成完成: {elapsed:.1f}s，输出 {size/1024:.0f} KB -> {out.name}")
    print(f"  RTF（相对实时）: {elapsed/ (size/32000):.2f}x  （<1 表示比实时快）")
    if size > 20000:
        print("  结论: IndexTTS2 可用 ✅")
        return 0
    print("  结论: 输出异常 ❌")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
