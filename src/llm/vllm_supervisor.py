"""vLLM on an owned WSL2 distribution, sharing the app's model manager interface."""
import json
from pathlib import Path
import subprocess
import time

from llm.supervisor import ModelSupervisor, ModelSwitchError, _http_json, port_listening


def wsl_path(value):
    value = str(value).replace("\\", "/")
    if len(value) > 2 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


class VllmSupervisor(ModelSupervisor):
    def specification(self):
        return json.loads((self.root / "config/vllm.json").read_text(encoding="utf-8-sig"))

    def installed(self):
        try:
            return json.loads((self.root / "data/vllm-install.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {"state": "not_installed", "detail": "运行 scripts/install-vllm.ps1 安装 WSL2 / vLLM"}

    def command(self, action, model=None):
        cfg = self.specification()
        argv = ["wsl.exe", "--distribution", cfg["distro"], "--user", "root", "--exec", cfg["python"],
                wsl_path(self.root / "scripts/vllm_service.py"), action, wsl_path(self.root / "config/vllm.json")]
        if model:
            argv.extend(["--model", model])
        return argv

    def run(self, action, model=None):
        result = subprocess.run(self.command(action, model), capture_output=True, timeout=60,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise ModelSwitchError("vLLM/WSL 启动失败；查看 logs/vllm.log，运行 scripts/install-vllm.ps1 完成安装。")
        try:
            return json.loads(result.stdout.decode("utf-8").strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise ModelSwitchError("vLLM 进程管理器未返回有效状态")

    def status(self):
        aliases = self.served_aliases()
        install = self.installed()
        ready = install.get("state") == "ready"
        tiers = []
        for spec in self.specification()["models"]:
            downloaded = (Path(spec["path"]) / ".complete.json").is_file()
            tiers.append({"id": spec["id"], "label": spec["id"], "size_gb": spec["size_gb"], "ctx": spec["ctx"],
                          "downloaded": downloaded, "runnable": downloaded and ready, "current": spec["id"] in aliases,
                          "expected_tps": "", "notes": "vLLM / WSL2，INT4 safetensors；视觉塔保留 BF16",
                          "download_command": "python scripts/download_vllm_model.py" if not downloaded else ""})
        return {"port": self.port, "listening": port_listening(self.port), "current": aliases[0] if aliases else None,
                "tiers": tiers, "runtime": "vllm", "installation": install, "gpu": self.gpu_memory(),
                "busy": self._lock.locked(), "progress": dict(self._progress)}

    def switch(self, tier_id, *, on_ready=None):
        self.check_audio_training()
        spec = next((s for s in self.specification()["models"] if s["id"] == tier_id), None)
        if spec is None:
            raise ModelSwitchError("请选择 vLLM 对应的 safetensors 模型；旧 GGUF 不是此运行时的可切换档位")
        if self.installed().get("state") != "ready":
            raise ModelSwitchError(self.installed().get("detail", "vLLM 尚未安装"))
        if not (Path(spec["path"]) / ".complete.json").is_file():
            raise ModelSwitchError("权重尚未校验完成：python scripts/download_vllm_model.py")
        if not self._lock.acquire(blocking=False):
            raise ModelSwitchError("模型正在切换")
        try:
            self._progress = {"state": "switching", "detail": "vLLM 正在加载模型", "tier": tier_id, "since": time.time()}
            record = self.run("start", tier_id)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if tier_id in self.served_aliases():
                    self._progress = {"state": "idle", "detail": "vLLM 已就绪", "tier": tier_id, "since": time.time()}
                    return {"ok": True, "tier": tier_id, "pid": record.get("pid"), "runtime": "vllm"}
                time.sleep(2)
            raise ModelSwitchError("vLLM 未能在 10 分钟内就绪，查看 logs/vllm.log")
        except Exception:
            self._progress["state"] = "failed"
            raise
        finally:
            self._lock.release()

    def stop(self):
        if self.installed().get("state") != "ready":
            return {"stopped": False}
        with self._lock:
            return self.run("stop")
