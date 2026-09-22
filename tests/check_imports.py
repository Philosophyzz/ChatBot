"""Import every first-party module of the project, for use by the PowerShell scripts.

``verify.ps1`` wants to prove the source tree is coherent (no syntax errors, no broken
internal imports, no module that only fails at request time). Running that check through
``python -c`` with an inline snippet is unreliable on Windows for the same reason as
``check_module.py``: the command line is passed through the ANSI code page and any
non-ASCII character is corrupted before Python sees it.

The import list is deliberately explicit rather than discovered by walking the tree:
a module that is never imported here should be a conscious decision, and an explicit
list fails loudly when a file is renamed.

Usage::

    python tests/check_imports.py            # summary only
    python tests/check_imports.py --verbose  # list every module as it loads

Exit code 0 when everything imports, 1 with a per-module error report otherwise.
"""

from __future__ import annotations

import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Modules that must import successfully. Grouped by subsystem so a failure points at
#: a layer rather than a random file.
MODULES = (
    # contracts and infrastructure
    "core.types",
    "core.errors",
    "core.config",
    "core.registry",
    "core.logging",
    "core.utils",
    "core.session",
    # capability layers
    "llm.openai_compat",
    "llm.mock",
    "llm.embed",
    "memory.db",
    "memory.vector",
    "memory.store",
    "memory.extract",
    "memory.retrieve",
    "memory.consolidate",
    "memory.engine",
    "persona.manager",
    "speech.audio",
    "speech.stt",
    "speech.tts",
    # orchestration
    "core.chat",
    "core.app",
    # transport
    "api.models",
    "api.routes",
    "api.server",
    # plugin package runs every @register decorator
    "plugs",
)


def main() -> int:
    import importlib

    verbose = "--verbose" in sys.argv[1:] or "-v" in sys.argv[1:]
    failures: list[tuple[str, str]] = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            if verbose:
                print(f"ok      {name}")
        except Exception as exc:  # noqa: BLE001 - any failure must be reported
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"FAILED  {name}  -> {type(exc).__name__}: {exc}")

    print("")
    if failures:
        print(f"{len(failures)}/{len(MODULES)} 个模块导入失败：")
        for name, error in failures:
            print(f"  - {name}: {error}")
        print("")
        print("完整 traceback（最后一个失败模块）：")
        name = failures[-1][0]
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        return 1

    print(f"全部 {len(MODULES)} 个模块导入成功")
    return 0


if __name__ == "__main__":
    sys.exit(main())
