"""Run headless browser checks against an isolated mock app, never real memories."""
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVER = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from core.config import AppConfig, Paths
from core.app import ChatBotApp
from api.server import create_asgi_app
import uvicorn
cfg = AppConfig(paths=Paths(root=Path(sys.argv[1]), web_dir=Path.cwd()/"web"))
cfg.llm.mock = True
cfg.memory.embedder = "hash"
cfg.memory.reranker = "heuristic"
cfg.memory.consolidate_enabled = False
cfg.extra['emotion'] = {'enabled': True, 'model_analysis': False}
cfg.speech.stt_backend = "mock"
cfg.speech.tts_preference = ["tone"]
uvicorn.run(create_asgi_app(ChatBotApp(cfg)), host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")
'''


def main():
    with tempfile.TemporaryDirectory(prefix="anime-ui-") as tmp:
        script = Path(tmp) / "server.py"
        script.write_text(SERVER, encoding="utf-8")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with (ROOT / "logs/anime-ui-server.txt").open("w", encoding="utf-8") as log:
            server = subprocess.Popen([sys.executable, str(script), tmp, str(port)], cwd=ROOT, stdout=log, stderr=log)
            try:
                for _ in range(80):
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=.3):
                            break
                    except OSError:
                        time.sleep(.2)
                result = subprocess.run([sys.executable, "tests/ui_check.py", "--url", f"http://127.0.0.1:{port}/",
                                         "--screenshot", str(ROOT / "logs/anime-web-preview.png")], cwd=ROOT, capture_output=True, timeout=180)
                (ROOT / "logs/anime-ui-check.txt").write_bytes(result.stdout + result.stderr)
                print("UI exit", result.returncode)
                return result.returncode
            finally:
                server.terminate()
                server.wait(timeout=20)


if __name__ == "__main__":
    raise SystemExit(main())
