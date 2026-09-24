"""Own the GPU for a finite lab run, then restore this project's chat model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'generate', 'experiment', 'compare'])
    args, remaining = parser.parse_known_args()
    from core.config import load_config
    from llm.supervisor import ModelSupervisor, port_listening
    from llm.vllm_supervisor import VllmSupervisor
    config = load_config(ROOT)
    manager = (VllmSupervisor if config.llm.backend == 'vllm' else ModelSupervisor)(ROOT, config)
    active = manager.served_aliases()
    restore = active[0] if active else None
    state_path = ROOT / 'data/asmr/lab-state.json'
    state_path.parent.mkdir(parents=True, exist_ok=True)
    stopped = False
    result = 1
    # One independent lab process at a time. No broad process-name/GPU-process kills.
    import msvcrt
    with (state_path.parent / 'lab.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            state_path.write_text(json.dumps({'state':'running','action':args.action,'restore_model':restore,'pid':os.getpid()}), encoding='utf-8')
            if restore:
                if config.llm.backend != 'vllm':
                    import psutil
                    pid = manager._find_chat_pid()
                    proc = psutil.Process(pid)
                    command = proc.cmdline()
                    known = {t.id: t for t in manager.tiers()}
                    if restore not in known or Path(proc.exe()).resolve() != manager.server_exe.resolve():
                        raise RuntimeError('Port 8080 is not an owned model; refusing to stop it')
                    if not any('llama-server' in str(x).lower() for x in command):
                        raise RuntimeError('Recorded process is not the expected owned model')
                stopped = bool(manager.stop().get('stopped'))
                if port_listening(8080):
                    raise RuntimeError('Chat model still owns GPU port')
            py = ROOT / 'venvs/asmr/Scripts/python.exe'
            result = subprocess.call([str(py), str(ROOT / 'tools/asmr_model.py'), args.action, *remaining], cwd=ROOT)
        finally:
            if stopped and restore:
                print('Restoring chat model:', restore, flush=True)
                manager.switch(restore)
            state_path.write_text(json.dumps({'state':'finished','exit_code':result,'restored_model':restore if stopped else None}), encoding='utf-8')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
