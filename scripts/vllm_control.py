"""Windows command entry: start/stop/activate the configured vLLM runtime."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.config import load_config
from llm.vllm_supervisor import VllmSupervisor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "stop", "status", "activate"])
    p.add_argument("--model")
    args = p.parse_args()
    manager = VllmSupervisor(ROOT, load_config())
    if args.action == "status":
        print(json.dumps(manager.status(), ensure_ascii=False))
    elif args.action == "stop":
        print(json.dumps(manager.stop()))
    else:
        name = args.model or manager.specification()["default_model"]
        result = manager.switch(name)
        if args.action == "activate":
            import urllib.request
            payload = {"model": name, "messages": [{"role": "user", "content": "What is 15% of 240? Reply with just the answer."}], "max_tokens": 64, "temperature": 0}
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with opener.open(request, timeout=120) as response:
                probe = json.load(response)
            if not str(probe["choices"][0]["message"].get("content") or "").strip():
                raise RuntimeError("vLLM returned no completion; activation was not persisted")
            import yaml
            path = ROOT / "config/local.yaml"
            local = yaml.safe_load(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
            local = local or {}
            local.setdefault("llm", {}).update(mode="local", backend="vllm", base_url="http://127.0.0.1:8080/v1", model=name,
                                               api_key="sk-local", mock=False, context_tokens=8192)
            temp = path.with_suffix(".yaml.tmp")
            temp.write_text(yaml.safe_dump(local, allow_unicode=True, sort_keys=False), encoding="utf-8")
            temp.replace(path)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
