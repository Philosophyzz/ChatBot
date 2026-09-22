#!/usr/bin/env python
"""Launcher for the web UI server, runnable from the project root.

Why this file exists: the application's top-level packages (``core``, ``memory``,
``api``…) live in ``src``, so ``python -m api.server`` only resolves when the working
directory *is* ``src``. Running it from the project root fails with a confusing
"No module named 'api'", and a bootstrap inside ``server.py`` cannot help because
``-m`` resolves the module name before executing any of its code.

This launcher puts ``src`` on ``sys.path`` first and then delegates, so all of these
work and behave identically::

    python run_server.py                       # project root, recommended
    python run_server.py --mock --port 8097
    python src\\api\\server.py --port 8077      # direct, also supported

``scripts/start-all.ps1`` uses this file as well.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if not SRC.is_dir():
    sys.stderr.write(
        f"[X] 找不到源码目录：{SRC}\n"
        "    请在项目根目录下运行本脚本（项目结构可能不完整）。\n"
    )
    raise SystemExit(2)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Default the project root so configuration resolves to this checkout even when the
# launcher is invoked from somewhere else (e.g. a shortcut or a scheduled task).
import os  # noqa: E402

os.environ.setdefault("CHATBOT_ROOT", str(ROOT))

try:
    from api.server import main
except ModuleNotFoundError as exc:  # the common first-run failure, made actionable
    missing = getattr(exc, "name", "") or str(exc)
    sys.stderr.write(
        f"[X] 缺少 Python 依赖：{missing}\n\n"
        "    请先运行安装脚本：\n"
        "        powershell -ExecutionPolicy Bypass -File scripts\\install.ps1\n\n"
        "    或手动安装：\n"
        "        venvs\\main\\Scripts\\python.exe -m pip install -r requirements.txt\n"
        "        venvs\\main\\Scripts\\python.exe -m pip install fastapi uvicorn httpx pydantic numpy PyYAML python-multipart websockets\n"
    )
    raise SystemExit(1) from exc

if __name__ == "__main__":
    raise SystemExit(main())
