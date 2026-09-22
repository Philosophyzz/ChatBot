"""Base-model switching: tier listing, error paths, and the engine rebind.

The live path (stop llama-server → start another → wait for /health) was exercised by hand
against two real models (9B → 27B → 9B). What is pinned here is everything that would silently
rot: the tier list must reflect what is actually on disk, a missing file must produce the
download command instead of a traceback, and a switch must repoint the *live* client — the
engine's health check compares ``/v1/models`` with ``llm.model``, so a stale name turns the UI
red ("服务已启动但未加载模型 …") even though the server is fine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_models_config(root: Path, *, with_args: bool = True) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "models" / "gguf").mkdir(parents=True, exist_ok=True)
    config = {
        "default_tier": "small",
        "default_server_args": ["--flash-attn", "on"] if with_args else [],
        "chat_tiers": [
            {
                "id": "small",
                "label": "小模型",
                "local_name": "small.gguf",
                "size_gb": 1.0,
                "ctx": 4096,
                "n_gpu_layers": -1,
                "expected_tps": "100 tok/s",
                "notes": "测试用",
            },
            {
                "id": "big",
                "label": "大模型",
                "local_name": "big.gguf",
                "size_gb": 9.0,
                "ctx": 8192,
                "n_gpu_layers": 20,
                "server_args": ["--n-cpu-moe", "8"],
                "notes": "测试用",
            },
        ],
    }
    (root / "config" / "models.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")


def test_tiers_report_download_state(tmp_path) -> None:
    from llm.supervisor import ModelSupervisor

    _write_models_config(tmp_path)
    (tmp_path / "models" / "gguf" / "small.gguf").write_bytes(b"GGUF")

    supervisor = ModelSupervisor(tmp_path, port=9)
    status = supervisor.status()
    by_id = {tier["id"]: tier for tier in status["tiers"]}
    assert by_id["small"]["downloaded"] is True
    assert by_id["big"]["downloaded"] is False
    assert by_id["big"]["download_command"].endswith("-Tier big -Mirror"), "未下载时必须给出下载命令"
    assert by_id["small"]["download_command"] == ""
    assert by_id["big"]["server_args"] == ["--n-cpu-moe", "8"], "档位级参数要带到界面/命令里"
    assert status["listening"] is False


def test_switching_to_a_missing_tier_explains_the_download(tmp_path) -> None:
    from llm.supervisor import ModelSupervisor, ModelSwitchError

    _write_models_config(tmp_path)
    supervisor = ModelSupervisor(tmp_path, port=9)
    with pytest.raises(ModelSwitchError) as excinfo:
        supervisor.switch("big")
    message = str(excinfo.value)
    assert "还没下载" in message and "download-models.ps1 -Tier big" in message

    with pytest.raises(ModelSwitchError) as unknown:
        supervisor.switch("does-not-exist")
    assert "未知档位" in str(unknown.value)


def test_default_server_args_come_from_models_json(tmp_path) -> None:
    """The start script and this supervisor read the same key — otherwise they drift."""
    from llm.supervisor import ModelSupervisor

    _write_models_config(tmp_path)
    supervisor = ModelSupervisor(tmp_path, port=9)
    assert supervisor.default_server_args() == ["--flash-attn", "on"]

    # …and an older file without the key must not explode.
    _write_models_config(tmp_path, with_args=False)
    assert ModelSupervisor(tmp_path, port=9).default_server_args() == []


def test_rebind_points_the_live_client_at_the_new_alias() -> None:
    """A switch must update the running client, not just the config file."""
    from core.app import ChatBotApp

    class FakeClient:
        def __init__(self) -> None:
            self.model = "fast-9b"

    app = ChatBotApp.__new__(ChatBotApp)  # avoid load_config()/logging side effects
    app.config = type("Cfg", (), {"llm": type("Llm", (), {"model": "fast-9b"})()})()
    app.llm = FakeClient()
    app.engine = type("Engine", (), {"llm": app.llm})()

    app.rebind_chat_model("quality-35b")
    assert app.config.llm.model == "quality-35b"
    assert app.llm.model == "quality-35b", "客户端里的别名没更新，健康检查会报'未加载模型'"


def test_llm_port_is_read_from_the_configured_url() -> None:
    from core.app import ChatBotApp

    app = ChatBotApp.__new__(ChatBotApp)
    app.config = type("Cfg", (), {"llm": type("Llm", (), {"base_url": "http://127.0.0.1:18080/v1"})()})()
    assert app._llm_port() == 18080
    app.config.llm.base_url = "not-a-url"
    assert app._llm_port() == 8080
