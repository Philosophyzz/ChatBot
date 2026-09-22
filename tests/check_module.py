"""Check whether a Python module can be imported, for use by the PowerShell scripts.

``verify.ps1`` needs to probe for optional dependencies. Doing that with
``python -c "..."`` is fragile on Windows: the command line is passed through the ANSI
code page, so any non-ASCII character in the inline snippet arrives corrupted and
Python raises a SyntaxError instead of reporting a missing module. Keeping the probe in
a real file avoids the whole class of problem.

Usage::

    python tests/check_module.py fastapi
    python tests/check_module.py faster_whisper yaml multipart

Exit code 0 when every named module imports, 1 otherwise. Requires no third-party
packages, so it works before the environment is fully installed.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import List


def main(argv: List[str]) -> int:
    names = [name for name in argv if name and not name.startswith("-")]
    if not names:
        sys.stderr.write("usage: check_module.py MODULE [MODULE ...]\n")
        return 2

    missing: List[str] = []
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if found:
            print(f"present {name}")
        else:
            print(f"missing {name}")
            missing.append(name)

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
