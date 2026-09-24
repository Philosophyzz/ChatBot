"""Contracts spanning character identity, interaction and model connection changes."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from core.config import load_config
from core.types import LLMRequest, Message, Role
from persona.catalog import CATALOG, LEGACY_IDS
from persona.manager import PersonaManager, build_system_prompt
from pet.settings import PetSettings
from pet.skin import builtin_skin_path


def test_catalogue_matches_art_and_initial_memory(temp_config):
    manager = PersonaManager(temp_config)
    manager.overrides_file.write_text(json.dumps({"tech_mentor": {"name": "retired"}}), encoding="utf-8")
    assert {p.id for p in manager.list()} == set(CATALOG)
    settings = PetSettings(temp_config.paths.data_dir / "pet.json")
    settings.set_skin(None, "builtin:sakura_cat")
    for persona in manager.list():
        assert persona.skin_id == persona.id
        assert settings.skin_for(persona.id) == str(builtin_skin_path(persona.id))
        assert persona.initial_memory in build_system_prompt(persona)
        assert "【角色初始记忆】" in build_system_prompt(persona)
        assert persona.voice.backend == "edge"
        assert set(persona.options["touch"]) >= {"head", "face", "body", "hold", "feed"}
    for old, new in LEGACY_IDS.items():
        assert manager.get(old).id == new


def test_model_connection_persists_without_erasing_settings(temp_config, run):
    from core.app import ChatBotApp
    from api.server import create_asgi_app
    app = ChatBotApp(temp_config)
    run(app.start())
    settings = temp_config.paths.config_dir / "local.yaml"
    settings.write_text('speech:\n  stt_device: cpu\ndebug: true\n', encoding="utf-8")
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_asgi_app(app)), base_url="http://test") as client:
            payload = {"mode": "api", "base_url": "https://example.test/v1", "model": "anime-chat", "api_key": "test-secret"}
            response = await client.put("/api/models/connection", json=payload)
            assert response.status_code == 200, response.text
            assert "test-secret" not in response.text
            assert app.engine.llm is app.memory.extractor.llm is app.memory.consolidator.llm is app.llm
            models = (await client.get("/api/models")).json()
            assert models["mode"] == "api" and models["tiers"] == []
            assert (await client.post("/api/models/switch", json={"tier": "Qwen3.5-9B-Q4_K_M"})).status_code == 409
            loaded = load_config(root=temp_config.paths.root)
            assert loaded.llm.mode == "api" and loaded.llm.api_key == "test-secret"
            assert loaded.speech.stt_device == "cpu" and loaded.debug
            payload.update(mode="local", base_url="http://127.0.0.1:8080/v1", model="MiMo-V2.6-Distill-Qwen-9B-INT4-W4A16-AutoRound", api_key="")
            assert (await client.put("/api/models/connection", json=payload)).status_code == 200
            assert app.llm.name == "vllm"
            assert app.llm.api_key == "sk-local"
        await app.stop()
    run(scenario())


def test_external_chat_has_no_llamacpp_extensions(run):
    from llm.openai_compat import OpenAICompatBackend
    async def scenario():
        llm = OpenAICompatBackend(base_url="https://example.test/v1", model="anime", api_key="test", name="openai")
        seen = []
        def respond(request):
            seen.append(json.loads(request.content))
            assert request.headers["Authorization"] == "Bearer test"
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(200, json={"choices": [{"message": {"content": "樱樱在呢。"}}]})
        llm._client = httpx.AsyncClient(base_url=llm.base_url, headers={"Authorization": "Bearer test"}, transport=httpx.MockTransport(respond))
        request = LLMRequest(messages=[Message(Role.USER, "你好")], json_schema={"type": "object"})
        assert "樱樱" in (await llm.complete(request)).text
        assert "json_schema" not in seen[0]
        assert "cache_prompt" not in llm._payload(request, stream=True)
        await llm.aclose()
    run(scenario())


def test_live2d_rejects_incomplete_models(tmp_path):
    from pet.live2d import validate_model
    model = tmp_path / "cat.model3.json"
    model.write_text(json.dumps({"FileReferences": {"Moc": "cat.moc3", "Textures": ["cat.png"]}}))
    with pytest.raises(ValueError, match="资源缺失"):
        validate_model(model)
    (tmp_path / "cat.moc3").write_bytes(b"test")
    (tmp_path / "cat.png").write_bytes(b"test")
    assert validate_model(model) == str(model)


def test_scaled_touch_and_browser_only_from_pet_menu(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QDesktopServices, QPixmap
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from pet.client import BackendClient
    from pet.window import PetWindow

    app = QApplication.instance() or QApplication([])
    settings = PetSettings(tmp_path / "pet.json")
    monkeypatch.setattr("pet.window.get_settings", lambda: settings)
    monkeypatch.setattr(PetWindow, "refresh_personas", lambda self: None)
    monkeypatch.setattr(PetWindow, "refresh_models", lambda self: None)
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    window = PetWindow(BackendClient("http://127.0.0.1:1"), persona_id="sakura_cat", with_tray=False)
    window.muted = True
    window.state = "idle"
    try:
        window.set_scale(1.5)
        assert window.width() == 315 and PetSettings(settings.path).get("scale") == 1.5
        window.set_scale(20)
        assert window.width() == 630
        window.set_scale(1.5)
        canvas = QPixmap(window.size())
        window.render(canvas)
        assert canvas.toImage().pixelColor(157, 290).alpha() > 200
        recorded = []
        monkeypatch.setattr(window, "start_recording", lambda: recorded.append(True))
        QTest.mouseClick(window, Qt.LeftButton, pos=QPoint(157, 290))
        assert recorded == [True], "Scaled mic must stay visible and clickable"
        QTest.mouseClick(window, Qt.LeftButton, pos=QPoint(150, 70))
        QTest.qWait(QApplication.doubleClickInterval() + 30)
        assert "耳朵" in window.bubble.label.text()
        QTest.mouseDClick(window, Qt.LeftButton, pos=QPoint(150, 100))
        assert window.bubble.input.isVisible()
        assert opened == []
        assert not any(a.text() == "打开网页" for a in window._menu().actions())
        action = next(a for a in window._menu(from_pet=True).actions() if a.text() == "打开网页")
        action.trigger()
        assert opened == ["http://127.0.0.1:1/"]
        assert not {"切换人设", "互动"}.intersection(a.text() for a in window._menu(from_pet=True).actions())
    finally:
        window._anim.stop()
        window.bubble.close()
        window.close()
        window.client.close()


def test_launchers_never_open_browser(project_root):
    launch = (project_root / "launcher/start_chatbot.py").read_text(encoding="utf-8")
    script = (project_root / "scripts/start-all.ps1").read_text(encoding="utf-8-sig")
    assert "webbrowser.open" not in launch
    assert "Start-Process $url" not in script
    assert "$connectionMode -ne 'api'" in script
