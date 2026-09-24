"""Linux-side owned-process lifecycle. Called through WSL without a command shell."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def linux_path(value):
    value = str(value).replace("\\", "/")
    return f"/mnt/{value[0].lower()}/{value[3:]}" if len(value) > 2 and value[1:3] == ":/" else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("config")
    parser.add_argument("--model")
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    root = cfg_path.parent.parent
    record_path = root / "data/vllm-process.json"
    lock_file = open(root / "data/vllm-process.lock", "a")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        record = {}
    pid = record.get("pid", 0)
    try:
        command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        owned = "vllm" in command and str(record.get("model_path", "__missing__")) in command and "--served-model-name" in command
    except (OSError, ValueError):
        owned = False
    if args.action == "status":
        print(json.dumps({"running": owned, "pid": pid if owned else None, "model": record.get("model")}))
        return
    if args.action == "stop" or (args.action == "start" and owned and record.get("model") != args.model):
        if owned:
            os.killpg(pid, signal.SIGTERM)
            for _ in range(30):
                if not Path(f"/proc/{pid}").exists():
                    break
                time.sleep(.5)
            if Path(f"/proc/{pid}").exists():
                os.killpg(pid, signal.SIGKILL)
        record_path.unlink(missing_ok=True)
        if args.action == "stop":
            print(json.dumps({"stopped": owned}))
            return
        owned = False
    if owned:
        print(json.dumps(record))
        return
    name = args.model or cfg["default_model"]
    spec = next(item for item in cfg["models"] if item["id"] == name)
    model_path = linux_path(spec["path"])
    if not (Path(model_path) / ".complete.json").is_file():
        raise RuntimeError("Model download/hash verification is incomplete; run scripts/download_vllm_model.py")
    command = [str(Path(sys.executable).parent / "vllm"), "serve", model_path, "--served-model-name", name,
               "--host", "127.0.0.1", "--port", str(cfg["port"]), "--max-model-len", str(spec["ctx"]), *spec["args"]]
    env = dict(os.environ, HF_HOME="/mnt/d/Models/hf-cache", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    with (root / "logs/vllm.log").open("ab", buffering=0) as log:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, start_new_session=True)
    record = {"pid": proc.pid, "model": name, "model_path": model_path, "started_at": time.time()}
    record_path.write_text(json.dumps(record), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
