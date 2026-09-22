"""确认打好的 exe 里真的带上了桌宠需要的模块（尤其是运行中才 import 的那几个）。

PyInstaller 的产物可以当压缩包读：外层 CArchive 里有一份 PYZ，PYZ 里是纯 Python 模块清单。
比"跑起来点点看"更直接 —— 缺模块会在这里当场暴露。
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader  # noqa: E402

EXE = sys.argv[1] if len(sys.argv) > 1 else r"dist\ChatBotPet.exe"
#: 纯 Python 模块在 PYZ 里；Qt 这类编译扩展在 CArchive（外层）里，所以两边都要查。
WANTED = [
    "pet.window",
    "pet.client",
    "pet.audio",
    "pet.single_instance",
    "pet.hands_free",
    "pet.skin",
    "pet.settings",
    "core.logging",
    "httpx",
    "PySide6/QtWidgets.pyd",
    "PySide6/QtCore.pyd",
]

reader = CArchiveReader(EXE)
print(f"外层条目 {len(reader.toc)} 个，PYZ 在不在：{'PYZ.pyz' in reader.toc or 'PYZ-00.pyz' in reader.toc}")

pyz_name = next((name for name in ("PYZ.pyz", "PYZ-00.pyz") if name in reader.toc), None)
if pyz_name is None:
    print("[X] 找不到 PYZ")
    raise SystemExit(1)

pyz_data = reader.extract(pyz_name)
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

# ZlibArchiveReader 只接受文件路径（内部按 "?offset" 解析文件名），所以先落到临时文件。
with tempfile.NamedTemporaryFile(suffix=".pyz", delete=False) as handle:
    handle.write(pyz_data)
    temp_path = Path(handle.name)
try:
    archive = ZlibArchiveReader(str(temp_path))
    modules = set(archive.toc)
finally:
    temp_path.unlink(missing_ok=True)
print(f"PYZ 里共 {len(modules)} 个模块\n")

missing = []
# CArchive 里的路径用反斜杠（Windows 打包），两种写法都认。
outer_names = {name.replace("\\", "/") for name in reader.toc}
for name in WANTED:
    ok = name in modules or name in outer_names
    where = "PYZ" if name in modules else ("外层" if name in outer_names else "-")
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name:<26} {where}")
    if not ok:
        missing.append(name)

related = sorted(item for item in modules if item.startswith("pet.") or item.startswith("speech."))
print(f"\n桌宠相关模块：{', '.join(related)}")
print("\n" + (f"[X] 缺少 {missing}" if missing else "[OK] 需要的模块都在 exe 里"))
raise SystemExit(1 if missing else 0)
