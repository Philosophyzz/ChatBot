"""Runtime switching of the base chat model.

Until now "换底座模型" meant editing ``config/models.json`` and ``config/config.yaml`` by hand,
then restarting two scripts — so the honest answer to "can I try the 14B?" was a documentation
link. The web UI and the pet tray now offer a picker, and this module is what makes it real:
stop the llama-server that serves ``:8080``, start the chosen tier with the same flags the
start script uses, wait until ``/health`` answers, and repoint the live engine at it.

Deliberate choices:

* **Only :8080 is touched.** The embedding/reranker servers (8081/8082) keep running, so long
  term memory does not silently degrade into hash embeddings while the user tries a model.
* **The flags come from ``config/models.json``** (``default_server_args`` + the tier's
  ``server_args``) — the same file ``scripts/start-models.ps1`` reads, so the script and the
  engine cannot drift apart.
* **Switching is serialised.** It takes 30–120 s, and two concurrent switches would leave two
  servers fighting over the port; a lock plus ``/api/models/status`` makes progress visible.
* **A missing model file is a first-class answer**, not an exception trace: the tier list says
  which tiers are downloaded, and the error carries the exact download command.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging import get_logger

log = get_logger(__name__)


class ModelSwitchError(RuntimeError):
    """Readable failure for the UI (missing file, timeout, port busy …)."""


@dataclass
class Tier:
    """One entry of ``chat_tiers`` in config/models.json."""

    id: str
    label: str = ""
    local_name: str = ""
    size_gb: float = 0.0
    ctx: int = 8192
    n_gpu_layers: int = -1
    expected_tps: str = ""
    notes: str = ""
    repo: str = ""
    file: str = ""
    server_args: List[str] = field(default_factory=list)
    fallback_repos: List[Dict[str, str]] = field(default_factory=list)
    mmproj_local_name: str = ""
    mmproj_repo: str = ""
    mmproj_file: str = ""
    mmproj_sha256: str = ""
    sha256: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Tier":
        return cls(
            id=str(raw.get("id") or ""),
            label=str(raw.get("label") or ""),
            local_name=str(raw.get("local_name") or ""),
            size_gb=float(raw.get("size_gb") or 0.0),
            ctx=int(raw.get("ctx") or 8192),
            n_gpu_layers=int(raw.get("n_gpu_layers") if raw.get("n_gpu_layers") is not None else -1),
            expected_tps=str(raw.get("expected_tps") or ""),
            notes=str(raw.get("notes") or ""),
            repo=str(raw.get("repo") or ""),
            file=str(raw.get("file") or ""),
            server_args=[str(item) for item in (raw.get("server_args") or [])],
            fallback_repos=list(raw.get("fallback_repos") or []),
            mmproj_local_name=str(raw.get("mmproj_local_name") or ""),
            mmproj_repo=str(raw.get("mmproj_repo") or ""),
            mmproj_file=str(raw.get("mmproj_file") or ""),
            mmproj_sha256=str(raw.get("mmproj_sha256") or ""),
            sha256=str(raw.get("sha256") or ""),
        )

    def path(self, root: Path, models_dir: Optional[Path] = None) -> Path:
        return (Path(models_dir) if models_dir else root / "models") / "gguf" / self.local_name

    def mmproj_path(self, root: Path, models_dir: Optional[Path] = None) -> Optional[Path]:
        if not self.mmproj_local_name:
            return None
        return (Path(models_dir) if models_dir else root / "models") / "gguf" / self.mmproj_local_name


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    """The system proxy (Clash on 127.0.0.1:7890) turns localhost calls into 502s."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http_json(url: str, timeout: float = 5.0) -> Optional[Any]:
    try:
        with _no_proxy_opener().open(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - "not ready yet" is the normal case here
        return None


def port_listening(port: int, host: str = "127.0.0.1", timeout: float = 0.6) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((host, port)) == 0


class ModelSupervisor:
    """Owns the lifecycle of the chat model server (port 8080)."""

    def __init__(
        self,
        root: Path,
        config: Any = None,
        *,
        port: int = 8080,
        health_timeout_s: float = 420.0,
        start_grace_s: float = 1.5,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.models_dir = getattr(getattr(config, "paths", None), "models_dir", self.root / "models")
        self.port = int(port)
        self.health_timeout_s = health_timeout_s
        self.start_grace_s = start_grace_s
        self._lock = threading.Lock()
        self._progress: Dict[str, Any] = {"state": "idle", "detail": "", "tier": None, "since": None}

    # -- config ------------------------------------------------------------------------
    @property
    def models_config_path(self) -> Path:
        return self.root / "config" / "models.json"

    @property
    def pids_path(self) -> Path:
        return self.root / "data" / "model-server.pids.json"

    def _models_config(self) -> Dict[str, Any]:
        try:
            return json.loads(self.models_config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelSwitchError(f"读不到 {self.models_config_path}：{exc}") from exc

    def tiers(self) -> List[Tier]:
        return [Tier.from_dict(raw) for raw in (self._models_config().get("chat_tiers") or [])]

    def tier(self, tier_id: str) -> Tier:
        for candidate in self.tiers():
            if candidate.id == tier_id:
                return candidate
        available = ", ".join(item.id for item in self.tiers())
        raise ModelSwitchError(f"未知档位 {tier_id}（可选：{available}）")

    def default_server_args(self) -> List[str]:
        return [str(item) for item in (self._models_config().get("default_server_args") or [])]

    @property
    def server_exe(self) -> Path:
        return self.root / "bin" / "llama.cpp" / "llama-server.exe"

    def download_command(self, tier_id: str) -> str:
        return f"powershell -ExecutionPolicy Bypass -File scripts\\download-models.ps1 -Tier {tier_id} -Mirror"

    # -- state -------------------------------------------------------------------------
    def _pids(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.pids_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_pids(self, values: Dict[str, Any]) -> None:
        merged = self._pids()
        merged.update(values)
        try:
            self.pids_path.parent.mkdir(parents=True, exist_ok=True)
            self.pids_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("could not record model pids", extra={"error": str(exc)})

    def served_aliases(self) -> List[str]:
        """What the running server actually answers to (``/v1/models``)."""
        payload = _http_json(f"http://127.0.0.1:{self.port}/v1/models", timeout=4.0)
        if not isinstance(payload, dict):
            return []
        return [str(item.get("id")) for item in (payload.get("data") or []) if item.get("id")]

    def current_tier(self) -> Optional[str]:
        """Which tier is being served right now: ask the server, fall back to the state file."""
        aliases = self.served_aliases()
        known = {tier.id for tier in self.tiers()}
        for alias in aliases:
            if alias in known:
                return alias
        recorded = self._pids().get("chat_tier")
        if isinstance(recorded, str) and recorded in known:
            return recorded
        configured = None
        if self.config is not None:
            configured = getattr(getattr(self.config, "llm", None), "model", None)
        if isinstance(configured, str) and configured in known:
            return configured
        return None

    def gpu_memory(self) -> Dict[str, int]:
        """Best-effort ``nvidia-smi`` reading (total/free MiB); empty when unavailable."""
        try:
            completed = subprocess.run(  # noqa: S603 - fixed command
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=8,
            )
            first = (completed.stdout or "").strip().splitlines()[0]
            total, free = (int(part.strip()) for part in first.split(","))
            return {"total_mib": total, "free_mib": free}
        except Exception:  # noqa: BLE001
            return {}

    def status(self) -> Dict[str, Any]:
        listening = port_listening(self.port)
        current = self.current_tier() if listening else None
        tiers = []
        for tier in self.tiers():
            path = tier.path(self.root, self.models_dir)
            mmproj = tier.mmproj_path(self.root, self.models_dir)
            downloaded = path.is_file() and (mmproj is None or mmproj.is_file())
            tiers.append(
                {
                    "id": tier.id,
                    "label": tier.label,
                    "size_gb": tier.size_gb,
                    "ctx": tier.ctx,
                    "n_gpu_layers": tier.n_gpu_layers,
                    "expected_tps": tier.expected_tps,
                    "notes": tier.notes,
                    "downloaded": downloaded,
                    "mmproj_downloaded": mmproj is None or mmproj.is_file(),
                    "current": tier.id == current,
                    "server_args": tier.server_args,
                    "mmproj_local_name": tier.mmproj_local_name or None,
                    "download_command": self.download_command(tier.id) if not downloaded else "",
                }
            )
        return {
            "port": self.port,
            "listening": listening,
            "current": current,
            "tiers": tiers,
            "gpu": self.gpu_memory(),
            "busy": self._progress.get("state") == "switching",
            "progress": dict(self._progress),
        }

    # -- switching ---------------------------------------------------------------------
    def check_audio_training(self) -> None:
        """Keep an owned finite audio experiment from racing another model load."""
        try:
            state = json.loads((self.root / 'data/asmr/lab-state.json').read_text(encoding='utf-8'))
            if state.get('state') != 'running' or state.get('pid') == os.getpid():
                return
            import psutil
            owner = psutil.Process(int(state['pid']))
            if any(Path(arg).name == 'asmr_lab.py' for arg in owner.cmdline()):
                raise ModelSwitchError('ASMR 生成/训练正在使用显卡，完成后会恢复聊天模型；请稍后切换。')
        except ModelSwitchError:
            raise
        except (OSError, ValueError, KeyError, TypeError, ImportError):
            return
        except Exception:
            return  # An expired/reused process record must never prevent normal startup.

    def switch(self, tier_id: str, *, on_ready: Optional[Any] = None) -> Dict[str, Any]:
        """Stop the current chat server and bring up ``tier_id``. Blocking (30–120 s)."""
        self.check_audio_training()
        tier = self.tier(tier_id)
        model_path = tier.path(self.root, self.models_dir)
        mmproj_path = tier.mmproj_path(self.root, self.models_dir)
        if not model_path.is_file() or (mmproj_path is not None and not mmproj_path.is_file()):
            raise ModelSwitchError(
                f"档位 {tier.id} 的模型还没下载：{model_path.name}\n"
                f"先在终端里执行：{self.download_command(tier.id)}"
            )
        if not self.server_exe.exists():
            raise ModelSwitchError(f"找不到 llama-server：{self.server_exe}")

        if not self._lock.acquire(blocking=False):
            raise ModelSwitchError("上一次切换还在进行中，请稍等")
        try:
            self._progress = {"state": "switching", "tier": tier.id, "detail": "正在停止旧模型", "since": time.time()}
            stopped = self.stop_locked()

            self._progress["detail"] = "正在启动新模型"
            pid = self._spawn(tier, model_path)
            self._write_pids({"chat": pid, "chat_tier": tier.id})

            self._progress["detail"] = "等待模型加载"
            self._wait_healthy(pid, tier)

            self._progress = {"state": "idle", "tier": tier.id, "detail": "已就绪", "since": time.time()}
            if on_ready is not None:
                on_ready(tier)
            log.info("chat model switched", extra={"tier": tier.id, "pid": pid, "stopped": stopped})
            return {
                "ok": True,
                "tier": tier.id,
                "label": tier.label,
                "pid": pid,
                "stopped": stopped,
                "gpu": self.gpu_memory(),
            }
        except Exception:
            self._progress = {"state": "failed", "tier": tier.id, "detail": "切换失败", "since": time.time()}
            raise
        finally:
            self._lock.release()

    def stop(self) -> Dict[str, Any]:
        """Stop the chat server (used by the API and by tests)."""
        if not self._lock.acquire(blocking=False):
            raise ModelSwitchError("正在切换中，稍后再试")
        try:
            return self.stop_locked()
        finally:
            self._lock.release()

    def stop_locked(self) -> Dict[str, Any]:
        """Kill whatever serves :8080 and wait for the port to be released."""
        pid = self._find_chat_pid()
        if pid is None:
            return {"stopped": False, "reason": "端口没人监听" if not port_listening(self.port) else "找不到占用进程"}
        if pid == os.getpid():
            raise ModelSwitchError("拒绝杀死当前进程（端口被引擎自己占用？）")
        self._kill(pid)
        deadline = time.time() + 30
        while time.time() < deadline and port_listening(self.port):
            time.sleep(0.4)
        time.sleep(self.start_grace_s)  # let the driver hand the VRAM back
        return {"stopped": True, "pid": pid, "port_free": not port_listening(self.port)}

    def _find_chat_pid(self) -> Optional[int]:
        """Who listens on our port? psutil first, then the recorded pid, then the command line."""
        try:
            import psutil  # noqa: PLC0415 - optional dependency, imported on demand

            for connection in psutil.net_connections(kind="tcp"):
                if connection.status == psutil.CONN_LISTEN and connection.laddr and connection.laddr.port == self.port:
                    if connection.pid:
                        return int(connection.pid)
        except Exception as exc:  # noqa: BLE001
            log.debug("psutil lookup failed", extra={"error": str(exc)})

        recorded = self._pids().get("chat")
        if isinstance(recorded, int) and self._pid_alive(recorded):
            return recorded

        try:
            import psutil  # noqa: PLC0415

            for process in psutil.process_iter(["name", "cmdline"]):
                try:
                    if not (process.info["name"] or "").lower().startswith("llama-server"):
                        continue
                    cmdline = " ".join(process.info["cmdline"] or [])
                    if f"--port {self.port}" in cmdline:
                        return int(process.pid)
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pass
        return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            import psutil  # noqa: PLC0415

            return psutil.pid_exists(pid)
        except Exception:  # noqa: BLE001
            return False

    def _kill(self, pid: int) -> None:
        try:
            import psutil  # noqa: PLC0415

            process = psutil.Process(pid)
            process.terminate()
            try:
                process.wait(timeout=8)
            except Exception:  # noqa: BLE001
                process.kill()
                process.wait(timeout=5)
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("psutil terminate failed, falling back to taskkill", extra={"error": str(exc)})
        subprocess.run(  # noqa: S603 - fixed command, pid is an int we looked up
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            capture_output=True,
            text=True,
            timeout=20,
        )

    def _spawn(self, tier: Tier, model_path: Path) -> int:
        argv = [
            str(self.server_exe),
            "--model", str(model_path),
            # One alias, equal to the tier id — exactly what scripts/start-models.ps1 does.
            # A comma list looks tempting ("tier id plus a stable name") but llama-server
            # registers only the last one, which then makes the engine's health check report
            # "服务已启动但未加载模型 <tier>" (it compares /v1/models against llm.model).
            # Switching therefore rebinds the live client instead — see App.rebind_chat_model.
            "--alias", tier.id,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", str(tier.ctx),
            "--n-gpu-layers", str(tier.n_gpu_layers),
            *self.default_server_args(),
            *tier.server_args,
        ]
        mmproj_path = tier.mmproj_path(self.root, self.models_dir)
        if mmproj_path is not None:
            if not mmproj_path.is_file():
                raise ModelSwitchError(f"模型的图像投影文件不存在：{mmproj_path.name}，请重新下载 {tier.id}")
            argv.extend(["--mmproj", str(mmproj_path)])
        logs = self.root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        out = open(logs / "llama-chat.log", "ab", buffering=0)
        err = open(logs / "llama-chat.err.log", "ab", buffering=0)
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            process = subprocess.Popen(argv, stdout=out, stderr=err, creationflags=flags)
        except OSError as exc:
            raise ModelSwitchError(f"启动 llama-server 失败：{exc}") from exc
        finally:
            out.close()
            err.close()
        log.info("spawned llama-server", extra={"tier": tier.id, "pid": process.pid, "argv": " ".join(argv[1:])})
        return int(process.pid)

    def _wait_healthy(self, pid: int, tier: Tier) -> None:
        deadline = time.time() + self.health_timeout_s
        url = f"http://127.0.0.1:{self.port}/health"
        while time.time() < deadline:
            if not self._pid_alive(pid):
                raise ModelSwitchError(
                    f"{tier.id} 启动后立刻退出了 —— 多半是显存不够。\n最近日志：\n{self._log_tail()}"
                )
            payload = _http_json(url, timeout=4.0)
            if isinstance(payload, dict) and str(payload.get("status", "")).lower() in {"ok", "ready"}:
                return
            if port_listening(self.port) and payload is None:
                pass  # still loading weights
            time.sleep(2.0)
        raise ModelSwitchError(
            f"等待 {tier.id} 就绪超时（{int(self.health_timeout_s)} 秒）。\n最近日志：\n{self._log_tail()}"
        )

    def _log_tail(self, lines: int = 12) -> str:
        path = self.root / "logs" / "llama-chat.err.log"
        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "(没有日志)"
        return "\n".join(content[-lines:])
