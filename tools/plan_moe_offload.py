"""算出「这个 GGUF 要多少显存、MoE 专家要放几层到内存」。

为什么需要它：35B 这种 MoE 模型，"能不能塞进 10GB" 取决于**非专家权重**（注意力、词表、
norm）与**专家权重**各占多少。llama.cpp 的 ``--n-cpu-moe N`` 可以把前 N 层的专家权重留在
内存，只把注意力等留在显存 —— 但 N 该取多少，光看文件体积猜不出来。

这个工具不需要下完整个模型：GGUF 的张量表就在文件开头（tokenizer 之后），用 HTTP Range
取几十 MB 就能精确算出每种方案的显存占用。下 12GB 之前先把账算清楚。

    # 本地已有的模型
    python tools\\plan_moe_offload.py --local models\\gguf\\Qwen3.6-27B-Q4_K_M.gguf --ctx 8192 --budget 10

    # 还没下载的模型（走镜像，只取文件头）
    python tools\\plan_moe_offload.py --repo unsloth\\Qwen3.6-35B-A3B-GGUF \\
        --file Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --ctx 8192 --budget 10

输出：权重总量、专家/非专家各多少、每层专家的体积，以及一张
"``--n-cpu-moe N`` → 显存占用" 的表，并给出推荐值。
"""

from __future__ import annotations

import argparse
import importlib.util
import struct
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

#: GGUF metadata value type ids.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL = range(8)
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12
_SCALAR_SIZE = {
    _UINT8: 1, _INT8: 1, _UINT16: 2, _INT16: 2, _UINT32: 4, _INT32: 4,
    _FLOAT32: 4, _BOOL: 1, _UINT64: 8, _INT64: 8, _FLOAT64: 8,
}
_MIB = 1024 * 1024


class BufferReader:
    """Sequential reader over a byte buffer that reports how much more it needs."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def need(self, count: int) -> None:
        if self.pos + count > len(self.data):
            raise EOFError(f"需要至少 {self.pos + count} 字节，目前只有 {len(self.data)} 字节")

    def take(self, count: int) -> bytes:
        self.need(count)
        chunk = self.data[self.pos : self.pos + count]
        self.pos += count
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        length = self.u64()
        return self.take(length).decode("utf-8", errors="replace")

    def value(self, type_id: int):
        if type_id in _SCALAR_SIZE:
            raw = self.take(_SCALAR_SIZE[type_id])
            fmt = {
                _UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h", _UINT32: "<I",
                _INT32: "<i", _FLOAT32: "<f", _BOOL: "<?", _UINT64: "<Q", _INT64: "<q",
                _FLOAT64: "<d",
            }[type_id]
            return struct.unpack(fmt, raw)[0]
        if type_id == _STRING:
            return self.string()
        if type_id == _ARRAY:
            element_type = self.u32()
            count = self.u64()
            if element_type == _STRING:
                # 词表数组可能有几十万个字符串，只要数量，不保留内容
                for _ in range(count):
                    self.string()
                return f"<{count} 个字符串>"
            step = _SCALAR_SIZE.get(element_type)
            if step is None:
                raise ValueError(f"数组元素类型 {element_type} 不支持")
            self.take(step * count)
            return f"<{count} 个数字>"
        raise ValueError(f"未知的元数据类型 {type_id}")


def parse_header(data: bytes) -> Tuple[Dict[str, object], List[Dict[str, object]], int]:
    """Returns (metadata, tensors, data_offset)."""
    reader = BufferReader(data)
    magic = reader.take(4)
    if magic != b"GGUF":
        raise ValueError("这不是 GGUF 文件（magic 不匹配）")
    version = reader.u32()
    tensor_count = reader.u64()
    kv_count = reader.u64()

    metadata: Dict[str, object] = {"general.gguf_version": version}
    for _ in range(kv_count):
        key = reader.string()
        type_id = reader.u32()
        metadata[key] = reader.value(type_id)

    tensors: List[Dict[str, object]] = []
    for _ in range(tensor_count):
        name = reader.string()
        n_dims = reader.u32()
        dims = [reader.u64() for _ in range(n_dims)]
        type_id = reader.u32()
        offset = reader.u64()
        tensors.append({"name": name, "dims": dims, "type": type_id, "offset": offset})
    return metadata, tensors, reader.pos


def tensor_sizes(tensors: List[Dict[str, object]], data_offset: int, file_size: int) -> None:
    """Fill each tensor's byte size using offsets (no quantization table needed)."""
    alignment = 32
    base = data_offset + (-data_offset % alignment)
    ordered = sorted(tensors, key=lambda item: int(item["offset"]))
    for index, tensor in enumerate(ordered):
        start = base + int(tensor["offset"])
        if index + 1 < len(ordered):
            end = base + int(ordered[index + 1]["offset"])
        else:
            end = file_size
        tensor["bytes"] = max(0, end - start)


def is_expert(name: str) -> bool:
    lowered = name.lower()
    return "exps" in lowered and "shexp" not in lowered


def layer_of(name: str) -> int:
    for part in name.split("."):
        if part.startswith("blk"):
            continue
    parts = name.split(".")
    for index, part in enumerate(parts):
        if part == "blk" and index + 1 < len(parts) and parts[index + 1].isdigit():
            return int(parts[index + 1])
    return -1


def gb(value: float) -> float:
    return value / 1024**3


def kv_cache_gb(meta: Dict[str, object], ctx: int, *, bytes_per_element: float = 2.0) -> float:
    """KV cache size for a given context length (fp16 by default)."""
    arch = str(meta.get("general.architecture") or "")
    block = int(meta.get(f"{arch}.block_count") or 0)
    head_count = int(meta.get(f"{arch}.attention.head_count") or 0)
    head_kv = int(meta.get(f"{arch}.attention.head_count_kv") or head_count)
    key_len = int(meta.get(f"{arch}.attention.key_length") or 0)
    value_len = int(meta.get(f"{arch}.attention.value_length") or key_len)
    if not key_len:
        embedding = int(meta.get(f"{arch}.embedding_length") or 0)
        key_len = value_len = (embedding // head_count) if head_count else 0
    if not (block and head_kv and key_len):
        return 0.0
    return block * ctx * head_kv * (key_len + value_len) * bytes_per_element / 1024**3


def read_local(path: Path, want_bytes: int) -> Tuple[bytes, int]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        return handle.read(want_bytes), size


def read_remote(repo: str, filename: str, want_bytes: int) -> Tuple[bytes, int]:
    """Fetch just the start of a (possibly not-yet-downloaded) GGUF from the mirror."""
    spec = importlib.util.spec_from_file_location("download_models", ROOT / "scripts" / "download_models.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_models"] = module
    spec.loader.exec_module(module)

    resolved = module.resolve_file(repo, filename, mirror=True, token=None)
    if not resolved:
        raise RuntimeError(f"仓库里找不到 {repo} :: {filename}")
    actual, size = resolved
    url = module.file_url(repo, actual, mirror=True)
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{want_bytes - 1}"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        data = response.read()
    return data, size


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="估算 GGUF 的显存占用与 MoE 卸载方案")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local", help="本地 GGUF 路径")
    source.add_argument("--repo", help="HuggingFace 仓库（走 hf-mirror）")
    parser.add_argument("--file", help="--repo 时必填：GGUF 文件名")
    parser.add_argument("--ctx", type=int, default=8192, help="上下文长度（算 KV cache 用）")
    parser.add_argument("--budget", type=float, default=10.0, help="显存预算（GB，默认 10）")
    parser.add_argument("--overhead", type=float, default=1.2, help="运行时额外开销 GB（CUDA 上下文/计算缓冲）")
    parser.add_argument("--head-mb", type=int, default=48, help="取文件头多少 MB 用来解析张量表")
    parser.add_argument(
        "--kv-type",
        default="q8_0",
        choices=["q8_0", "fp16"],
        help="KV cache 精度；本项目部署脚本用的是 q8_0（显存约为 fp16 的一半）",
    )
    args = parser.parse_args(argv)

    want = args.head_mb * _MIB
    try:
        if args.local:
            path = Path(args.local)
            if not path.exists():
                print(f"找不到文件：{path}")
                return 1
            print(f"读取本地文件头：{path.name}")
            data, file_size = read_local(path, want)
        else:
            if not args.file:
                print("--repo 需要同时给 --file")
                return 1
            print(f"从 hf-mirror 取文件头（最多 {args.head_mb} MB）：{args.repo} :: {args.file}")
            data, file_size = read_remote(args.repo, args.file, want)
    except Exception as exc:  # noqa: BLE001
        print(f"取文件头失败：{exc}")
        return 1

    for attempt in range(3):
        try:
            meta, tensors, data_offset = parse_header(data)
            tensor_sizes(tensors, data_offset, file_size)
            break
        except EOFError as exc:
            bigger = len(data) * 3
            print(f"  文件头不够（{exc}），再取 {bigger // _MIB} MB …")
            try:
                if args.local:
                    data, file_size = read_local(Path(args.local), bigger)
                else:
                    data, file_size = read_remote(args.repo, args.file, bigger)
            except Exception as inner:  # noqa: BLE001
                print(f"重试失败：{inner}")
                return 1
    else:
        print("解析 GGUF 文件头失败（多次扩大读取仍不够）")
        return 1

    arch = str(meta.get("general.architecture") or "?")
    blocks = int(meta.get(f"{arch}.block_count") or 0)
    experts = meta.get(f"{arch}.expert_count")
    total = sum(int(item["bytes"]) for item in tensors)
    expert_tensors = [item for item in tensors if is_expert(str(item["name"]))]
    other_tensors = [item for item in tensors if not is_expert(str(item["name"]))]
    expert_total = sum(int(item["bytes"]) for item in expert_tensors)
    other_total = sum(int(item["bytes"]) for item in other_tensors)

    print(f"\n架构 {arch}｜层数 {blocks}｜专家数 {experts if experts is not None else '（非 MoE）'}")
    print(f"文件体积 {gb(file_size):.2f} GB｜张量合计 {gb(total):.2f} GB")
    print(f"  专家权重   {gb(expert_total):7.2f} GB（{len(expert_tensors)} 个张量）")
    print(f"  其余权重   {gb(other_total):7.2f} GB（注意力/词表/norm 等 {len(other_tensors)} 个）")

    kv = kv_cache_gb(meta, args.ctx, bytes_per_element=1.0 if args.kv_type == "q8_0" else 2.0)
    print(f"KV cache（ctx={args.ctx}{'，q8_0' if args.kv_type == 'q8_0' else '，fp16'}）≈ {kv:.2f} GB")
    print(f"运行时开销（估计）≈ {args.overhead:.2f} GB")
    print(f"显存预算 {args.budget:.1f} GB")

    if not expert_tensors:
        need = gb(other_total) + kv + args.overhead
        print(f"\n这是稠密模型：全部层进显存需要 ≈ {need:.2f} GB")
        print("  " + ("[OK] 在预算内" if need <= args.budget else "[!] 超出预算，需要分层卸载（-ngl 少几层）"))
        return 0

    per_layer: Dict[int, float] = {}
    for item in expert_tensors:
        per_layer[layer_of(str(item["name"]))] = per_layer.get(layer_of(str(item["name"])), 0.0) + int(item["bytes"])

    print(f"\n{'--n-cpu-moe':>12}  {'显存占用':>9}  {'内存占用':>9}   方案")
    best: Optional[int] = None
    for cpu_layers in range(0, blocks + 1, max(1, blocks // 24)):
        resident = sum(size for layer, size in per_layer.items() if layer >= cpu_layers)
        cpu_side = expert_total - resident
        need = gb(other_total) + gb(resident) + kv + args.overhead
        fits = need <= args.budget
        if fits and best is None:
            best = cpu_layers
        print(
            f"{cpu_layers:>12}  {need:>8.2f}G  {gb(cpu_side):>8.2f}G   "
            f"{'✅ 放得下' if fits else '❌ 超预算'}"
        )

    print()
    if best is None:
        print(f"[!] 即使把全部专家都放内存，显存仍需 ≈ {gb(other_total) + kv + args.overhead:.2f} GB —— 超出预算。")
        print("    可以再降 ctx、给 KV cache 用 q8_0（--quant-kv），或换更小的量化。")
    elif best == 0:
        print(f"[推荐] 全部层放显存（不需要 --n-cpu-moe），显存 ≈ {gb(other_total) + gb(expert_total) + kv + args.overhead:.2f} GB")
    else:
        print(f"[推荐] --n-cpu-moe {best}：显存 ≈ "
              f"{gb(other_total) + gb(sum(size for layer, size in per_layer.items() if layer >= best)) + kv + args.overhead:.2f} GB，"
              f"内存 ≈ {gb(expert_total - sum(size for layer, size in per_layer.items() if layer >= best)):.2f} GB")
        print("    含义：前 %d 层的专家权重留在内存，注意力等仍在显存 —— MoE 每 token 只激活少量专家，" % best)
        print("    换成 CPU 计算主要吃内存带宽，速度会降但通常仍可用。")
    print("\n把选定的参数写进 config\\models.json 对应档位的 server_args 里，例如：")
    print(f'  "server_args": ["--n-cpu-moe", "{best if best else 0}", "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0"]')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
