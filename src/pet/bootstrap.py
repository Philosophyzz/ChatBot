"""The one desktop executable starts its backend without opening a browser."""
import subprocess
from pathlib import Path

import httpx

from pet.settings import project_root


def ensure_backend(base_url: str, *, mock: bool = False) -> None:
    try:
        response = httpx.get(base_url.rstrip("/") + "/api/health", trust_env=False, timeout=4)
        if response.is_success and response.json().get("ok"):
            return
    except (httpx.HTTPError, ValueError):
        pass
    from urllib.parse import urlsplit
    parsed = urlsplit(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("远程后端没有响应，请先启动该服务")
    root = project_root()
    if root is None or not (root / "scripts/start-all.ps1").is_file():
        raise RuntimeError("找不到 scripts/start-all.ps1，请将启动程序放在 ChatBot 根目录")
    log_path = root / "logs/desktop-start.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
               str(root / "scripts/start-all.ps1"), "-Root", str(root), "-NoBrowser",
               "-Port", str(parsed.port or 8077)]
    if mock:
        command.append("-Mock")
    with log_path.open("ab") as log:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=300)
    if result.returncode:
        raise RuntimeError(f"服务启动失败，请查看 {log_path}")


def stop_services() -> None:
    root = project_root()
    if root is None:
        raise RuntimeError("找不到项目目录")
    subprocess.Popen(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                      str(root / "scripts/stop.ps1"), "-Root", str(root)], cwd=root,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
