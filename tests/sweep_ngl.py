"""Sweep n-gpu-layers for a model that does not fit fully in VRAM.

On a 16 GB card the 27B Q4 weights (15.66 GB) leave only two options: put most layers
on the GPU and accept a tight fit, or put some on the CPU and pay for it on every
token. The default of 28 layers measured 3.6 tok/s — far worse than the estimate — so
this script measures the real curve instead of guessing again.

For each layer count it starts its own llama-server, waits for readiness, runs one
short completion, records tok/s and VRAM, then stops the server. Results are printed as
a table and written to ``logs/ngl-sweep.json`` so the choice can be re-checked later.

Usage::

    python tests/sweep_ngl.py --model models/gguf/qwen3.6-27b-q4_k_m.gguf --layers 28,34,40,46
    python tests/sweep_ngl.py --layers 40 --ctx 8192
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = ROOT / "bin" / "llama.cpp" / "llama-server.exe"
LOGS = ROOT / "logs"
PORT = 8091  # a dedicated port so a running chat server is not disturbed


def vram_used() -> Optional[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def wait_ready(process: subprocess.Popen, timeout_s: int = 300) -> Optional[float]:
    started = time.time()
    while time.time() - started < timeout_s:
        if process.poll() is not None:
            return None  # died (usually OOM)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=4) as response:
                if response.status == 200:
                    return time.time() - started
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return None


def measure(model: str, ctx: int, layers: int) -> Dict[str, Any]:
    before = vram_used()
    stdout = LOGS / f"ngl-{layers}.log"
    stderr = LOGS / f"ngl-{layers}.err.log"
    args = [
        str(SERVER),
        "--model", model,
        "--alias", f"ngl{layers}",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--ctx-size", str(ctx),
        "--n-gpu-layers", str(layers),
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--parallel", "1",
        "--cont-batching",
        "--no-reasoning-preserve",
        "--jinja",
    ]
    result: Dict[str, Any] = {"layers": layers, "ctx": ctx, "ok": False}
    with open(stdout, "wb") as out, open(stderr, "wb") as err:
        process = subprocess.Popen(args, stdout=out, stderr=err)
        try:
            load_s = wait_ready(process)
            if load_s is None:
                result["error"] = "启动失败或显存不足（OOM）"
                # Surface the reason from the server log.
                tail = ""
                if stderr.exists():
                    tail = stderr.read_text(encoding="utf-8", errors="replace")[-400:]
                if "out of memory" in tail.lower() or "cuda error" in tail.lower():
                    result["error"] = "显存不足（CUDA OOM）"
                return result

            result["load_s"] = round(load_s, 1)
            after_load = vram_used()
            result["vram_before_mib"] = before
            result["vram_loaded_mib"] = after_load
            if before is not None and after_load is not None:
                result["vram_delta_mib"] = after_load - before

            payload = json.dumps(
                {
                    "model": f"ngl{layers}",
                    "messages": [{"role": "user", "content": "你好"}],
                    "max_tokens": 32,
                    "stream": False,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.time()
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - started
            timings = body.get("timings") or {}
            result["ok"] = True
            result["tps"] = round(float(timings.get("predicted_per_second") or 0), 2)
            result["prompt_tps"] = round(float(timings.get("prompt_per_second") or 0), 1)
            result["tokens"] = int(timings.get("predicted_n") or 0)
            result["elapsed_s"] = round(elapsed, 1)
            result["vram_peak_mib"] = vram_used()
            return result
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
            time.sleep(3)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="扫描 n-gpu-layers 找最优速度")
    parser.add_argument("--model", default=str(ROOT / "models" / "gguf" / "qwen3.6-27b-q4_k_m.gguf"))
    parser.add_argument("--layers", default="28,34,40,44,48")
    parser.add_argument("--ctx", type=int, default=8192)
    args = parser.parse_args(argv)

    model = pathlib.Path(args.model)
    if not model.exists():
        print(f"找不到模型：{model}")
        return 2
    if not SERVER.exists():
        print(f"找不到 llama-server：{SERVER}")
        return 2

    layers = [int(item) for item in args.layers.split(",") if item.strip()]
    LOGS.mkdir(parents=True, exist_ok=True)

    print(f"模型: {model.name}  ({model.stat().st_size / 1024 ** 3:.2f} GB)")
    print(f"上下文: {args.ctx}   扫描 n-gpu-layers: {layers}")
    print(f"基线的显存占用: {vram_used()} MiB")
    print("")

    results: List[Dict[str, Any]] = []
    for value in layers:
        print(f"--- n-gpu-layers = {value} ---")
        outcome = measure(str(model), args.ctx, value)
        results.append(outcome)
        if outcome.get("ok"):
            print(
                f"    加载 {outcome.get('load_s')}s   显存 +{outcome.get('vram_delta_mib')} MiB   "
                f"生成 {outcome.get('tps')} tok/s   预填充 {outcome.get('prompt_tps')} tok/s"
            )
        else:
            print(f"    失败: {outcome.get('error')}")

    print("")
    print("=" * 72)
    print(f"{'n-gpu-layers':>13} {'显存增量':>10} {'加载s':>7} {'生成 tok/s':>11} {'预填充 tok/s':>13}")
    usable = [item for item in results if item.get("ok")]
    for item in results:
        if item.get("ok"):
            print(
                f"{item['layers']:>13} {item.get('vram_delta_mib', 0):>9} MiB "
                f"{item.get('load_s', 0):>7} {item.get('tps', 0):>11} {item.get('prompt_tps', 0):>13}"
            )
        else:
            print(f"{item['layers']:>13} {'-':>10} {'-':>7} {'-':>11}   {item.get('error')}")
    print("=" * 72)

    if usable:
        best = max(usable, key=lambda item: item.get("tps") or 0)
        print(f"最快: n-gpu-layers={best['layers']}  {best['tps']} tok/s  显存 +{best.get('vram_delta_mib')} MiB")
        print(f"把它写进 config/models.json 的 n_gpu_layers 即可生效。")

    (LOGS / "ngl-sweep.json").write_text(
        json.dumps({"model": model.name, "ctx": args.ctx, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已写入 logs/ngl-sweep.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
