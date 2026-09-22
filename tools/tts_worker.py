"""IndexTTS2 worker: runs inside the dedicated TTS environment.

Why a separate process instead of importing the model directly:

* ``indextts`` pins ``numpy==2.2.6``, ``torch==2.8.*``, ``transformers==4.52.1``,
  ``keras==2.9.0``, ``opencv-python``, ``numba``, ``modelscope``… Installing that into
  the application's environment would downgrade numpy under the running app (which uses
  onnxruntime/ctranslate2) and risk breaking a working install for a feature that is
  optional. The main environment therefore stays clean.
* The model is ~6GB of weights plus a multi-GB CUDA torch; keeping it in its own venv
  also means it can be installed, upgraded or deleted without touching the app.

Protocol: one JSON object per line on stdin, one JSON object per line on stdout. Every
``indextts`` print is redirected to stderr so stdout stays parseable.

    {"id": 1, "cmd": "ping"}
    {"id": 2, "cmd": "synthesize", "text": "...", "output": "C:\\tmp\\out.wav",
     "reference": "models/voices/default_sweet.wav", "model_dir": "models/tts/IndexTTS2",
     "emotion": "sweet", "emotion_alpha": 0.9, "fp16": true, "device": "cuda:0"}
    {"id": 3, "cmd": "unload"}
    {"id": 4, "cmd": "info"}

Run it manually to debug::

    venvs\\tts\\Scripts\\python.exe tools\\tts_worker.py
    {"id":1,"cmd":"ping"}
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Progress bars and library banners would corrupt the JSON stream.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_EMOTION_TEXT = {
    "sweet": "用甜美、温柔、撒娇的语气说",
    "cheerful": "用轻快、开心、有活力的语气说",
    "calm": "用平静、舒缓、放松的语气说",
    "serious": "用认真、严肃、沉稳的语气说",
    "sad": "用低落、难过的语气说",
    "gentle": "用轻柔、体贴的语气说",
    "excited": "用兴奋、激动的语气说",
    "shy": "用害羞、腼腆的语气说",
}


def _log(message: str) -> None:
    """Diagnostics go to stderr; stdout is reserved for the protocol."""
    print(f"[tts-worker] {message}", file=sys.stderr, flush=True)


class Worker:
    def __init__(self) -> None:
        self._tts: Any = None
        self._model_dir: Optional[str] = None
        self._device: str = "cuda:0"
        self._fp16: bool = True

    # -- model -------------------------------------------------------------------------
    def load(self, model_dir: str, device: str, fp16: bool) -> Any:
        if self._tts is not None and self._model_dir == model_dir:
            return self._tts
        with contextlib.redirect_stdout(sys.stderr):
            from indextts.infer_v2 import IndexTTS2  # type: ignore

            _log(f"loading IndexTTS2 from {model_dir} on {device} (fp16={fp16})")
            started = time.time()
            try:
                self._tts = IndexTTS2(
                    model_dir=model_dir, cfg_path=None, use_fp16=fp16, device=device
                )
            except TypeError:
                self._tts = IndexTTS2(model_dir=model_dir, is_fp16=fp16, device=device)
            _log(f"loaded in {time.time() - started:.1f}s")
        self._model_dir, self._device, self._fp16 = model_dir, device, fp16
        return self._tts

    def unload(self) -> None:
        self._tts = None
        self._model_dir = None
        try:
            import gc

            gc.collect()
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        _log("model unloaded")

    # -- commands ----------------------------------------------------------------------
    def ping(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"ok": True, "pid": os.getpid(), "loaded": self._tts is not None}
        # Report whether the model package is importable. torch alone is not enough: an
        # environment with torch but without ``indextts`` would otherwise look healthy and
        # only fail mid-sentence.
        try:
            import importlib.util

            info["indextts"] = importlib.util.find_spec("indextts") is not None
        except Exception:  # noqa: BLE001
            info["indextts"] = False
        try:
            import torch  # type: ignore

            info["torch"] = getattr(torch, "__version__", "?")
            info["cuda"] = bool(torch.cuda.is_available())
            if info["cuda"]:
                info["device_name"] = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info()
                info["vram_free_mb"] = int(free / 1024 / 1024)
                info["vram_total_mb"] = int(total / 1024 / 1024)
        except Exception as exc:  # noqa: BLE001
            info["torch_error"] = f"{type(exc).__name__}: {exc}"
        return info

    def synthesize(self, request: Dict[str, Any]) -> Dict[str, Any]:
        text = str(request.get("text") or "").strip()
        output = request.get("output")
        reference = request.get("reference")
        if not text:
            raise ValueError("text 为空")
        if not output:
            raise ValueError("缺少 output 路径")
        if not reference:
            raise ValueError("缺少参考音频（voice.reference_audio），IndexTTS2 是零样本克隆，必须有参考音色")
        reference_path = Path(reference)
        if not reference_path.exists():
            raise FileNotFoundError(f"参考音频不存在：{reference}")

        tts = self.load(
            str(request.get("model_dir") or self._model_dir or ""),
            str(request.get("device") or "cuda:0"),
            bool(request.get("fp16", True)),
        )
        kwargs: Dict[str, Any] = {
            "spk_audio_prompt": str(reference_path),
            "output_path": str(output),
            "verbose": False,
        }
        emotion = str(request.get("emotion") or "").strip()
        alpha = request.get("emotion_alpha")
        vector = request.get("emotion_vector")
        if vector is not None:
            kwargs["emo_vector"] = vector
            kwargs["emo_alpha"] = float(alpha if alpha is not None else 0.8)
        elif emotion and emotion != "neutral" and request.get("use_emotion_text", True):
            kwargs["use_emo_text"] = True
            kwargs["emo_text"] = str(request.get("emotion_text") or _EMOTION_TEXT.get(emotion, f"用{emotion}的语气说"))
            kwargs["emo_alpha"] = float(alpha if alpha is not None else 0.8)

        started = time.time()
        with contextlib.redirect_stdout(sys.stderr):
            try:
                tts.infer(text=text, **kwargs)
            except TypeError as exc:
                # Older revisions have a smaller signature; keep audio flowing.
                _log(f"infer() signature mismatch ({exc}), retrying with minimal args")
                tts.infer(spk_audio_prompt=str(reference_path), text=text, output_path=str(output))
        path = Path(output)
        size = path.stat().st_size if path.exists() else 0
        if not size:
            raise RuntimeError("IndexTTS2 没有生成音频文件")
        return {"ok": True, "ms": round((time.time() - started) * 1000, 1), "bytes": size}


def main() -> int:
    worker = Worker()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request: Dict[str, Any] = {}
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "ping":
                result = worker.ping()
            elif command == "info":
                result = {
                    "ok": True,
                    "model_dir": worker._model_dir,
                    "device": worker._device,
                    "fp16": worker._fp16,
                    "loaded": worker._tts is not None,
                }
            elif command == "unload":
                worker.unload()
                result = {"ok": True}
            elif command == "synthesize":
                result = worker.synthesize(request)
            elif command == "exit":
                worker.unload()
                print(json.dumps({"id": request.get("id"), "ok": True}), flush=True)
                return 0
            else:
                result = {"ok": False, "error": f"未知命令：{command}"}
        except Exception as exc:  # noqa: BLE001 - the protocol must survive any failure
            result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-1200:],
            }
            _log(f"error: {result['error']}")
        result["id"] = request.get("id")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
