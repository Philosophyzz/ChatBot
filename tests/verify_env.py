"""Verify that the installed environment can actually import what the app needs.

``install.ps1`` runs this right after installing dependencies. Reporting "install
succeeded" without checking imports pushes the failure to the user's first launch,
where it surfaces as an opaque ImportError — so the installer fails loudly here
instead.

The check is a real ``import``, not ``find_spec``: a package can be resolvable and
still fail to import (a missing DLL, a broken ABI). ``ctranslate2`` in particular
needs to load its native library, which is exactly the failure mode a spec lookup
would miss.

Exit code 0 = every required module imports. Exit code 1 = at least one required
module failed; optional modules are reported but never fail the run.
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Dict, List, Tuple

#: Modules the application cannot start without, mapped to the pip name to show in
#: any error message.
REQUIRED: Dict[str, str] = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "httpx": "httpx",
    "pydantic": "pydantic",
    "numpy": "numpy",
    "yaml": "PyYAML",
    "multipart": "python-multipart",
    "websockets": "websockets",
}

#: Modules that unlock a feature. Missing ones degrade gracefully by design.
OPTIONAL: Dict[str, Tuple[str, str]] = {
    "faster_whisper": ("faster-whisper", "语音输入"),
    "edge_tts": ("edge-tts", "在线语音合成"),
}


def try_import(module: str) -> Tuple[bool, str]:
    try:
        importlib.import_module(module)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - any failure is a failure to report
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print(f"  Python {platform.python_version()} ({sys.executable})")

    missing: List[Tuple[str, str, str]] = []
    for module, pip_name in REQUIRED.items():
        ok, error = try_import(module)
        if ok:
            print(f"    [OK]   {module}")
        else:
            print(f"    [FAIL] {module}  -> {error}")
            missing.append((module, pip_name, error))

    degraded: List[Tuple[str, str, str]] = []
    for module, (pip_name, feature) in OPTIONAL.items():
        ok, error = try_import(module)
        if ok:
            print(f"    [OK]   {module}（{feature}）")
        else:
            print(f"    [!]    {module} 缺失（{feature}不可用）-> {error}")
            degraded.append((module, pip_name, error))

    print("")
    if missing:
        print(f"  [X] {len(missing)} 个必需依赖无法导入：")
        for module, pip_name, error in missing:
            print(f"      {pip_name}  ({module})")
            print(f"        原因: {error}")
        print("")
        print("  修复命令（在项目根目录执行）：")
        print(f"      venvs\\main\\Scripts\\python.exe -m pip install {' '.join(p for _, p, _ in missing)}")
        print(f"  或：uv pip install --python venvs\\main\\Scripts\\python.exe {' '.join(p for _, p, _ in missing)}")
        if any(module == "ctranslate2" or module == "faster_whisper" for module, _, _ in missing):
            print("  提示：faster-whisper 依赖 ctranslate2 的原生库，若报 DLL 相关错误，")
            print("        先安装 VC++ 运行库：winget install Microsoft.VCRedist.2015+.x64")
        return 1

    print("  [OK] 全部必需依赖导入成功")
    if degraded:
        print(f"  [!]  {len(degraded)} 个可选依赖缺失，对应功能会自动降级：")
        for module, pip_name, _error in degraded:
            print(f"      {pip_name}  ->  pip install {pip_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
