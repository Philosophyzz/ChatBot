"""Chat engine, persona, speech and API surface tests — the end-to-end paths."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ------------------------------------------------------------------------------------
# Persona
# ------------------------------------------------------------------------------------


def test_builtin_personas_have_voice_and_prompt(tmp_path) -> None:
    from core.config import AppConfig, Paths
    from persona.manager import PersonaManager

    config = AppConfig(paths=Paths(root=tmp_path))
    manager = PersonaManager(config)
    personas = manager.list()
    assert len(personas) >= 4
    for persona in personas:
        assert persona.system_prompt.strip(), f"{persona.id} 缺少系统提示词"
        assert persona.voice.voice_id, f"{persona.id} 缺少音色"
        assert 0.0 <= persona.temperature <= 2.0
    # 默认人设必须存在，否则启动就会失败
    assert manager.get(None).id == config.default_persona


def test_unknown_persona_falls_back_to_default(tmp_path) -> None:
    from core.config import AppConfig, Paths
    from persona.manager import PersonaManager

    manager = PersonaManager(AppConfig(paths=Paths(root=tmp_path)))
    assert manager.get("nope").id == manager.config.default_persona


def test_persona_crud_roundtrip(tmp_path) -> None:
    from core.config import AppConfig, Paths
    from persona.manager import PersonaManager

    manager = PersonaManager(AppConfig(paths=Paths(root=tmp_path)))
    created = manager.upsert(
        "test_persona",
        {
            "name": "测试人设",
            "system_prompt": "你是测试角色。",
            "voice": {"voice_id": "zh-CN-XiaoyiNeural", "emotion": "sweet", "speed": 1.1},
        },
    )
    assert created.name == "测试人设"
    assert created.voice.emotion == "sweet"
    assert manager.exists("test_persona")

    # 重新加载后仍然存在（真的写到磁盘了）
    reloaded = PersonaManager(AppConfig(paths=Paths(root=tmp_path)))
    assert reloaded.exists("test_persona")
    assert reloaded.get("test_persona").voice.voice_id == "zh-CN-XiaoyiNeural"

    assert manager.delete("test_persona") is True
    assert not PersonaManager(AppConfig(paths=Paths(root=tmp_path))).exists("test_persona")


def test_builtin_persona_cannot_be_deleted(tmp_path) -> None:
    from core.config import AppConfig, Paths
    from core.errors import BadRequest
    from persona.manager import PersonaManager

    manager = PersonaManager(AppConfig(paths=Paths(root=tmp_path)))
    with pytest.raises(BadRequest):
        manager.delete("sweet_companion")


def test_system_prompt_assembly_includes_memory_and_voice_rules(tmp_path) -> None:
    from core.config import AppConfig, Paths
    from persona.manager import PersonaManager, build_system_prompt

    manager = PersonaManager(AppConfig(paths=Paths(root=tmp_path)))
    persona = manager.get("sweet_companion")
    prompt = build_system_prompt(
        persona,
        memory_block="【长期记忆：偏好】\n- 用户喜欢美式咖啡",
        voice_mode=True,
    )
    assert persona.system_prompt.strip()[:10] in prompt
    assert "美式咖啡" in prompt
    assert "当前时间" in prompt
    assert "语音朗读" in prompt, "语音模式必须附加朗读约束"


def test_prompt_never_asks_the_user_to_supply_memories() -> None:
    """A small model told "you have long-term memory" turns into an interviewer.

    Left alone it asks for a name, then a job, then preferences, because its memory is
    empty and it reads the empty slots as questions. Memory here is extracted from the
    conversation automatically, so every assembled prompt must say so explicitly —
    this test is what keeps that rule from being dropped in a later edit.
    """
    from core.config import AppConfig, Paths
    from persona.manager import _MEMORY_CONDUCT, PersonaManager, build_system_prompt

    manager = PersonaManager(AppConfig(paths=Paths(root=Path("."))))
    for persona in manager.list():
        prompt = build_system_prompt(persona)
        assert _MEMORY_CONDUCT in prompt, f"{persona.id} 缺少记忆使用约定"
        assert "不需要手动填写" in prompt
        # The old built-in wording invited the model to ask the user for the missing
        # information, which is exactly the behaviour the user complained about.
        assert "邀请用户告诉你" not in persona.system_prompt


def test_user_personas_are_listed_before_the_builtins() -> None:
    from core.config import AppConfig, Paths
    from persona.manager import PersonaManager

    personas = PersonaManager(AppConfig(paths=Paths(root=Path(".")))).list()
    flags = [persona.builtin for persona in personas]
    assert flags == sorted(flags), "自己写的人设要排在前面，内置人设只做兜底"


def test_default_persona_points_at_an_authored_preset(project_root: Path) -> None:
    """``default_persona`` must name a persona that actually ships in personas.yaml.

    If it drifts (a rename, a typo), startup still succeeds — the manager silently
    falls back to a *built-in* persona, so the user gets a stranger's character with a
    different voice and never learns why.
    """
    from core.config import load_yaml_or_json

    config = load_yaml_or_json(project_root / "config" / "config.yaml")
    presets = (load_yaml_or_json(project_root / "config" / "personas.yaml") or {}).get("personas") or {}
    default = config.get("default_persona")
    assert default in presets, f"default_persona={default!r} 不在 personas.yaml 里：{sorted(presets)}"


def test_memory_panel_states_that_memories_are_automatic(project_root: Path) -> None:
    """The memory tab looks like an editor; the UI must say it is not one.

    Reported symptom: "我点开 8077，它强制让我填写记忆". The panel never had an input
    for memories, but nothing on screen said so either. One permanent line fixes it,
    and this test keeps that line from being refactored away.
    """
    app_js = (project_root / "web" / "assets" / "app.js").read_text(encoding="utf-8")
    html = (project_root / "web" / "index.html").read_text(encoding="utf-8")
    assert "prependMemoryNotice" in app_js
    assert "不需要你填写" in app_js
    assert "自动" in app_js
    # No form and no required field anywhere: nothing on the page can demand input.
    assert "<form" not in html.lower()
    assert "required" not in html.lower()


def test_hidden_attribute_actually_hides(project_root: Path) -> None:
    """Regression: both dialogs used to sit on screen and could not be dismissed.

    ``.modal { display: flex }`` and ``.recording-hint { display: flex }`` are author
    rules, and author rules beat the user-agent's ``[hidden] { display: none }``. So
    ``element.hidden = true`` had no visual effect at all: on page load the 编辑人设 and
    编辑记忆 dialogs covered the whole app, 取消 did nothing, and a "正在录音…" banner
    stayed under the composer. A verified headless screenshot showed exactly that.

    Two independent guards: the CSS must hide ``[hidden]``, and JS must not rely on the
    attribute alone when closing a dialog.
    """
    css = (project_root / "web" / "assets" / "style.css").read_text(encoding="utf-8")
    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important", css), (
        "缺少 [hidden] { display: none !important }：作者样式会盖掉浏览器默认行为，"
        "弹窗会变成关不掉的遮罩"
    )

    app_js = (project_root / "web" / "assets" / "app.js").read_text(encoding="utf-8")
    assert "function setModalOpen" in app_js
    assert "closeAllModals" in app_js
    # Dialogs must never be toggled with a bare `hidden` assignment again.
    assert not re.search(r"\$\('#modal-[a-z]+'\)\.hidden\s*=", app_js)
    # ...and the user always has a way out of a stuck dialog.
    assert "Escape" in app_js


# ------------------------------------------------------------------------------------
# Speech
# ------------------------------------------------------------------------------------


def test_wav_roundtrip_and_resample() -> None:
    from speech.audio import TARGET_SAMPLE_RATE, pcm_duration_s, pcm_to_wav, probe_wav, resample_pcm, trim_silence

    pcm = b"\x00\x00" * 16000  # 1 秒静音
    wav = pcm_to_wav(pcm)
    info = probe_wav(wav)
    assert info is not None
    assert info["sample_rate"] == TARGET_SAMPLE_RATE
    assert info["channels"] == 1
    assert abs(info["duration_s"] - 1.0) < 0.01
    assert abs(pcm_duration_s(pcm) - 1.0) < 0.01

    doubled = resample_pcm(pcm, 16000, 32000)
    assert len(doubled) == len(pcm) * 2
    assert trim_silence(pcm) == pcm, "全静音时不应破坏原始数据"


def test_sniff_format_detects_containers() -> None:
    from speech.audio import sniff_format

    assert sniff_format(b"RIFF....WAVE")[0] == "wav"
    assert sniff_format(b"\x1aE\xdf\xa3rest")[0] == "webm"
    assert sniff_format(b"OggS\x00\x02")[0] == "ogg"
    assert sniff_format(b"\x00\x00\x00\x18ftypmp42")[0] == "m4a"
    assert sniff_format(b"????")[0] == "bin"


def test_decode_rejects_non_audio() -> None:
    from core.errors import AudioError
    from speech.audio import decode_to_pcm

    with pytest.raises(AudioError):
        decode_to_pcm(b"")


def test_tone_tts_backend_produces_valid_wav(run) -> None:
    from core.types import VoiceSpec
    from speech.audio import probe_wav
    from speech.tts import ToneTTSBackend

    async def scenario():
        backend = ToneTTSBackend()
        chunks = [chunk async for chunk in backend.synthesize("你好呀，这是测试。", VoiceSpec())]
        return chunks

    chunks = run(scenario())
    assert len(chunks) == 1
    assert chunks[0].is_final
    info = probe_wav(chunks[0].data)
    assert info is not None and info["duration_s"] > 0.1


def test_tts_router_prefers_offline_and_falls_back(run) -> None:
    """The router must skip a failing backend instead of failing the request."""
    from core.types import SpeechChunk, VoiceSpec
    from speech.tts import TTSRouter, ToneTTSBackend

    class BrokenBackend:
        name = "broken"
        supports_cloning = False
        offline = True

        async def synthesize(self, text, voice, *, stream=False):
            raise RuntimeError("本地模型没装好")
            yield  # pragma: no cover

        async def health(self):
            return {"ok": True, "name": self.name}

    class OnlineBackend(ToneTTSBackend):
        name = "online"
        offline = False

    async def scenario():
        router = TTSRouter(
            {"broken": BrokenBackend(), "online": OnlineBackend()},
            preference=["broken", "online"],
            prefer_offline=True,
            failure_cooldown_s=60,
        )
        chunks = [chunk async for chunk in router.synthesize("测试一句话。", VoiceSpec())]
        health = await router.health()
        return chunks, health

    chunks, health = run(scenario())
    assert chunks, "必须回退到可用后端并产出音频"
    assert chunks[0].mime == "audio/wav"
    assert health["last_used"] == "online"


def test_tts_router_honours_pinned_backend(run) -> None:
    from core.types import VoiceSpec
    from speech.tts import TTSRouter, ToneTTSBackend

    class OtherTone(ToneTTSBackend):
        name = "other"

    async def scenario():
        router = TTSRouter({"tone": ToneTTSBackend(), "other": OtherTone()}, preference=["tone", "other"])
        backend = await router.select(VoiceSpec(backend="other"))
        return backend.name

    assert run(scenario()) == "other"


def test_mock_stt_returns_transcript(run) -> None:
    from core.types import AudioClip
    from speech.stt import MockSTT

    async def scenario():
        return await MockSTT(text="你好世界").transcribe(AudioClip(data=b"\x00" * 100))

    transcript = run(scenario())
    assert transcript.text == "你好世界"


# ------------------------------------------------------------------------------------
# Chat engine
# ------------------------------------------------------------------------------------


def _build_engine(config, mock_llm, run):
    from core.session import SessionManager
    from core.chat import ChatEngine
    from memory.db import Database
    from memory.engine import MemoryEngine
    from memory.vector import SqliteVectorStore
    from llm.embed import HashingEmbedder, HeuristicReranker
    from persona.manager import PersonaManager

    async def _build():
        db = Database(config.paths.db_path)
        db.init()
        memory = await MemoryEngine.create(
            config,
            llm=mock_llm,
            db=db,
            embedder=HashingEmbedder(dim=config.memory.dim),
            vectors=SqliteVectorStore(db),
            reranker=HeuristicReranker(),
        )
        personas = PersonaManager(config)
        sessions = SessionManager(memory.store)
        engine = ChatEngine(
            config=config, llm=mock_llm, personas=personas, memory=memory, sessions=sessions
        )
        return engine, memory

    return run(_build())


def test_engine_streams_events_in_order(temp_config, mock_llm, run) -> None:
    engine, memory = _build_engine(temp_config, mock_llm, run)

    async def scenario():
        events: List[Dict[str, Any]] = []
        async for event in engine.stream_turn(session_id=None, user_text="你好呀"):
            events.append(event)
        await memory.close()
        return events

    events = run(scenario())
    kinds = [event["kind"] for event in events]
    assert kinds[0] == "session"
    assert "start" in kinds
    assert "text" in kinds
    assert kinds[-1] == "done"
    assert "error" not in kinds
    text = "".join(event.get("text", "") for event in events if event["kind"] == "text")
    assert text.strip()
    assert events[-1]["text"] == text


def test_engine_persists_and_remembers_across_sessions(temp_config, mock_llm, run) -> None:
    """The headline feature: a fact told in one session is recalled in the next."""
    engine, memory = _build_engine(temp_config, mock_llm, run)

    async def scenario():
        first = await engine.complete_turn(session_id=None, user_text="我叫小明，我喜欢喝不加糖的美式咖啡")
        session_id = None
        for event in first.events:
            if event["kind"] == "done":
                session_id = event["session_id"]
        assert session_id

        # 抽出记忆（异步队列 → 直接等待，保证确定性）
        await memory._run_extraction(
            {
                "session_id": session_id,
                "user_message": engine.sessions._sessions[session_id].messages[-2],
                "assistant_message": engine.sessions._sessions[session_id].messages[-1],
                "persona_id": "sweet_companion",
                "scope": "global",
            }
        )
        memories = await memory.store.list_memories()
        context = await memory.build_context("我喜欢喝什么咖啡？", session_id="another")
        await memory.close()
        return memories, context

    memories, context = run(scenario())
    assert memories, "应当写入长期记忆"
    assert "咖啡" in context.rendered
    assert any("咖啡" in memory.content for memory in memories)


def test_engine_switches_persona_and_voice(temp_config, mock_llm, run) -> None:
    engine, memory = _build_engine(temp_config, mock_llm, run)

    async def scenario():
        result = await engine.complete_turn(
            session_id=None, user_text="你好", persona_id="calm_assistant"
        )
        await memory.close()
        return result

    result = run(scenario())
    persona_events = [event for event in result.events if event["kind"] == "start"]
    assert persona_events
    assert persona_events[0]["persona"]["id"] == "calm_assistant"
    assert persona_events[0]["persona"]["temperature"] == 0.4


def test_engine_trims_long_history_within_budget(temp_config, mock_llm, run) -> None:
    """The assembled prompt must fit the context window, and say what it dropped.

    History is built directly rather than by running many turns: driving real turns
    through the mock model costs wall-clock time and makes the token total depend on
    how verbose the mock happens to be, which is exactly the kind of flakiness that
    hides a real regression. Here the only variable is the trimming logic, which is
    what this test is about.
    """
    from core.types import Message, Role
    from core.utils import estimate_messages_tokens

    temp_config.llm.context_tokens = 4000
    temp_config.llm.reserve_tokens = 800
    engine, memory = _build_engine(temp_config, mock_llm, run)

    async def scenario():
        session = await engine.sessions.create(persona_id="sweet_companion")
        # More messages than the in-memory window holds, to prove the window trims
        # (Session.append keeps the newest DEFAULT_WINDOW entries) and that the
        # prompt builder then trims *again* to the token budget.
        for index in range(40):
            session.append(Message(role=Role.USER, content=f"第 {index} 条很长的测试消息" * 8))
            session.append(Message(role=Role.ASSISTANT, content=f"这是第 {index} 条回复" * 8))
        prompt = await engine.build_prompt(session, "现在总结一下")
        await memory.close()
        return prompt, session

    prompt, session = run(scenario())
    budget = temp_config.llm.context_tokens
    total = estimate_messages_tokens(prompt["messages"])
    from core.session import DEFAULT_WINDOW

    assert len(session.messages) == DEFAULT_WINDOW, (
        f"会话窗口应裁剪到 {DEFAULT_WINDOW} 条，实际 {len(session.messages)}"
    )
    assert total <= budget, f"prompt {total} tokens 超过预算 {budget}"
    assert prompt["messages"][0].role.value == "system"
    assert prompt["messages"][-1].content == "现在总结一下"
    # Trimming must leave an explicit note rather than silently losing the middle.
    assert any("未展示" in message.content for message in prompt["messages"])
    assert prompt["history_turns"] < DEFAULT_WINDOW


def test_engine_reports_backend_failure_as_error_event(temp_config, run) -> None:
    from core.errors import BackendUnavailable
    from core.session import SessionManager
    from core.chat import ChatEngine
    from persona.manager import PersonaManager

    class FailingLLM:
        name = "failing"
        model = "none"

        async def stream(self, request):
            raise BackendUnavailable("模型服务未启动")
            yield  # pragma: no cover

        async def complete(self, request):
            raise BackendUnavailable("模型服务未启动")

        async def health(self):
            return {"ok": False, "error": "down"}

    engine = ChatEngine(
        config=temp_config,
        llm=FailingLLM(),
        personas=PersonaManager(temp_config),
        memory=None,
        sessions=SessionManager(None),
    )

    async def scenario():
        return [event async for event in engine.stream_turn(session_id=None, user_text="在吗")]

    events = run(scenario())
    kinds = [event["kind"] for event in events]
    assert "error" in kinds, "后端不可用必须转成可读的 error 事件，而不是抛异常炸掉连接"
    assert "模型服务未启动" in json.dumps(events, ensure_ascii=False)


# ------------------------------------------------------------------------------------
# API surface
# ------------------------------------------------------------------------------------


@pytest.fixture
def api_client(temp_config, monkeypatch):
    """TestClient over the real ASGI app in mock mode (no GPU, no downloads)."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import plugs  # noqa: F401
    from core.app import ChatBotApp
    from api.server import create_asgi_app
    from llm.embed import HashingEmbedder, HeuristicReranker
    from memory.engine import MemoryEngine

    async def fake_create(config, **kwargs):
        # 确定性组件：测试不去连真实的嵌入/重排服务
        kwargs.setdefault("embedder", HashingEmbedder(dim=config.memory.dim))
        kwargs.setdefault("reranker", HeuristicReranker())
        # NOTE: must call the *saved original*. Calling ``MemoryEngine.create`` here
        # would call this replacement again — infinite recursion — because the
        # attribute has already been rebound to this function.
        return await _original_create(config, **kwargs)

    _original_create = MemoryEngine.create
    # staticmethod, not classmethod: re-wrapping a closure with classmethod binds the
    # class into the first positional parameter *at call time on some Python
    # versions*, which silently shifts ``config`` and raises a confusing TypeError.
    monkeypatch.setattr(MemoryEngine, "create", staticmethod(fake_create))

    app = ChatBotApp(temp_config)
    asgi = create_asgi_app(app)
    with TestClient(asgi) as client:
        yield client


def test_app_builds_real_llm_backend(temp_config) -> None:
    """Regression: constructing the configured backend must not collide on 'name'.

    ``Registry.create(kind, name, **kwargs)`` already receives the plugin name
    positionally. Passing ``name=`` in the constructor kwargs as well raised
    ``TypeError: Registry.create() got multiple values for argument 'name'``, which
    happened during app startup — so the server refused to boot with any non-mock
    backend, and the traceback pointed at the registry rather than the caller.
    """
    pytest.importorskip("httpx")
    from core.app import ChatBotApp
    from core.registry import KIND_LLM, registry

    config = temp_config
    config.llm.mock = False
    config.llm.backend = "llamacpp"
    # Point at a dead port: construction must succeed without any network access.
    config.llm.base_url = "http://127.0.0.1:59999/v1"
    config.llm.model = "some-alias"

    app = ChatBotApp(config)
    registry.load_plugins("plugs")
    instance = app._build_llm(use_mock=False)

    assert instance is not None
    # The plugin's own ``name`` attribute comes from its default, not from the caller.
    assert getattr(instance, "name", ""), "后端实例应带有 name 属性"
    assert instance.model == "some-alias", "构造参数应被透传"
    assert instance.base_url.endswith("/v1")

    # And every registered LLM plugin must be constructible this way.
    for plugin in ("llamacpp", "ollama", "openai", "vllm", "mock"):
        built = registry.create(KIND_LLM, plugin, base_url="http://127.0.0.1:1/v1", model="m")
        assert built is not None, plugin


def test_start_models_alias_matches_config_model(project_root) -> None:
    """The llama-server alias must equal ``llm.model`` or every request 404s.

    llama-server only answers to the alias it was launched with. The launcher used to
    hardcode ``--alias qwen3.6-27b``, so switching tiers made the app ask for a model
    the server did not know. The alias is now the tier id; this test keeps the two
    sides in sync.
    """
    import json
    import re

    from core.config import load_config

    text = (project_root / "scripts" / "start-models.ps1").read_text(encoding="utf-8")
    assert "$chatAlias = $Tier" in text, "启动脚本应使用档位 id 作为别名"
    assert not re.search(r"-Alias\s+'qwen", text), "别名不应再硬编码具体模型名"

    config = load_config(project_root)
    models = json.loads((project_root / "config" / "models.json").read_text(encoding="utf-8"))
    tier_ids = {tier["id"] for tier in models["chat_tiers"]}
    assert config.llm.model in tier_ids, (
        f"config.yaml 的 llm.model={config.llm.model!r} 不是任何档位 id {sorted(tier_ids)}；"
        "启动脚本会用档位 id 作别名，两边必须一致"
    )


def test_a_tier_exists_that_fits_entirely_in_vram(project_root) -> None:
    """At least one tier must fit fully in a 16 GB card, and the catalogue must say so.

    The shipped default is the quality tier (15.66 GB), which needs partial CPU offload
    and runs at 4-8 tok/s on this class of GPU. That is a legitimate default — but only
    because a genuinely fast alternative is one download away. This test keeps that
    escape hatch present instead of letting the catalogue drift into "every option is
    slow", and validates the fields the launcher indexes by.
    """
    import json
    import re

    models = json.loads((project_root / "config" / "models.json").read_text(encoding="utf-8"))
    tiers = models["chat_tiers"]
    assert len(tiers) >= 2, "至少要有质量档与速度档两个选择"

    # A 16 GB card has ~12.5 GB usable once the desktop takes its share.
    usable_gb = 12.5
    fits = [tier for tier in tiers if tier["size_gb"] <= usable_gb and tier["n_gpu_layers"] == -1]
    assert fits, (
        "没有任何档位能在 16GB 卡上全量进显存（size<=12.5GB 且 n_gpu_layers=-1）；"
        f"现有档位：{[(t['id'], t['size_gb'], t['n_gpu_layers']) for t in tiers]}"
    )

    for tier in tiers:
        # Tier ids become llama-server --alias values, so they must be safe argv.
        assert re.fullmatch(r"[a-z0-9][a-z0-9._-]*", tier["id"]), (
            f"档位 id {tier['id']!r} 含不适合做命令行别名的字符"
        )
        assert tier["ctx"] >= 4096, f"{tier['id']} 的上下文太小"
        assert tier["local_name"].endswith(".gguf"), tier["local_name"]


def test_default_tier_matches_config_model(project_root) -> None:
    """``default_tier``, ``llm.model`` and the launcher alias must all agree.

    Three places encode the same choice: ``config/models.json:default_tier`` (which tier
    the download and start scripts use by default), ``config/config.yaml:llm.model``
    (the alias the app asks for), and the launcher, which uses the tier id as
    ``--alias``. If they disagree the app starts and then cannot answer a single
    request, or the start script looks for a file that was never downloaded. This was
    hit for real while switching tiers.
    """
    import json

    from core.config import load_config

    models = json.loads((project_root / "config" / "models.json").read_text(encoding="utf-8"))
    config = load_config(project_root)
    tier_ids = [tier["id"] for tier in models["chat_tiers"]]

    assert models["default_tier"] in tier_ids, models["default_tier"]
    assert config.llm.model == models["default_tier"], (
        f"config.yaml 的 llm.model={config.llm.model!r} 与 "
        f"models.json 的 default_tier={models['default_tier']!r} 不一致："
        "启动脚本会用 default_tier 找模型、用档位 id 作别名，两者必须相同"
    )


def test_registry_create_rejects_unknown_plugin_cleanly() -> None:
    import plugs  # noqa: F401

    from core.registry import KIND_LLM, registry

    with pytest.raises(KeyError) as excinfo:
        registry.create(KIND_LLM, "definitely-not-a-plugin")
    assert "definitely-not-a-plugin" in str(excinfo.value)
    assert "available" in str(excinfo.value)


def test_api_health_and_plugins(api_client) -> None:
    response = api_client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["llm"]["ok"] is True

    plugins = api_client.get("/api/plugins").json()
    assert "llm" in plugins["registry"]
    assert "mock" in plugins["registry"]["llm"]


def test_api_personas_and_sessions(api_client) -> None:
    body = api_client.get("/api/personas").json()
    personas = body["personas"]
    assert len(personas) >= 4
    # The UI needs the server to name the default: it cannot infer it from list order.
    assert body["default"] in [persona["id"] for persona in personas]

    created = api_client.post("/api/sessions", json={"persona_id": personas[0]["id"]}).json()
    session_id = created["session"]["id"]
    assert created["greeting"] == personas[0]["greeting"]

    listing = api_client.get("/api/sessions").json()["sessions"]
    assert any(item["id"] == session_id for item in listing)

    detail = api_client.get(f"/api/sessions/{session_id}").json()["session"]
    assert detail["id"] == session_id

    assert api_client.delete(f"/api/sessions/{session_id}").json()["ok"] is True


@pytest.fixture
def static_client(temp_config):
    """TestClient whose web root actually contains the shell files.

    ``temp_config`` only creates an empty ``web/``; serving it would raise
    "File at path ... does not exist" from FileResponse, so the two files are written
    here. The memory engine needs no stubbing: ``temp_config`` already selects the
    local hash embedder and the heuristic reranker.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import plugs  # noqa: F401
    from core.app import ChatBotApp
    from api.server import create_asgi_app

    web = Path(temp_config.paths.web_dir)
    (web / "assets").mkdir(parents=True, exist_ok=True)
    (web / "index.html").write_text("<!DOCTYPE html><title>shell</title>", encoding="utf-8")
    (web / "assets" / "app.js").write_text("// shell\n", encoding="utf-8")

    with TestClient(create_asgi_app(ChatBotApp(temp_config))) as client:
        yield client


def test_static_shell_is_revalidated_not_cached(static_client) -> None:
    """A stale page must be impossible: it looks exactly like a broken fix.

    Without an explicit Cache-Control, browsers apply heuristic freshness to ``/`` and
    ``/assets/app.js``; the user then keeps seeing an older UI after an update and
    reports the bug as unfixed.
    """
    for path in ("/", "/assets/app.js"):
        response = static_client.get(path)
        assert response.status_code == 200, path
        assert "no-cache" in response.headers.get("cache-control", ""), path


def test_missing_index_html_is_a_404_not_a_500(temp_config) -> None:
    """Robustness: an empty web dir must not produce a confusing traceback."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import plugs  # noqa: F401
    from core.app import ChatBotApp
    from api.server import create_asgi_app

    with TestClient(create_asgi_app(ChatBotApp(temp_config))) as client:
        assert client.get("/").status_code == 404


def test_shutdown_endpoint_is_localhost_only(api_client) -> None:
    """A LAN client must not be able to stop the assistant.

    The server can be bound to 0.0.0.0 so a phone can be used as microphone and
    speaker, and a remote caller finding ``/api/shutdown`` would otherwise be able to
    take it down. TestClient reports itself as the client host "testclient", which is
    exactly the not-local case.
    """
    response = api_client.post("/api/shutdown")
    assert response.status_code == 403


def test_shutdown_endpoint_flips_the_uvicorn_flag() -> None:
    """Localhost callers get a graceful exit request (see ``Server.should_exit``).

    The handler is driven directly with hand-built ASGI scopes: ``TestClient`` always
    reports itself as "testclient", so it cannot exercise the loopback branch, and
    the guard itself is exactly what needs testing.
    """
    import asyncio

    from fastapi import HTTPException
    from starlette.requests import Request

    from api import routes

    class FakeServer:
        should_exit = False

    class FakeApp:
        state = type("S", (), {"uvicorn_server": FakeServer()})()

    def make_request(host: str) -> Request:
        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/shutdown",
                "headers": [(b"x-forwarded-for", b"127.0.0.1")],
                "client": (host, 51234),
                "app": FakeApp(),
            }
        )

    async def drive() -> Any:
        refused = None
        try:
            await routes.shutdown(make_request("192.168.1.20"))
        except HTTPException as exc:
            refused = exc.status_code
        allowed = await routes.shutdown(make_request("127.0.0.1"))
        await asyncio.sleep(0.4)  # the flag is flipped on a short timer
        return refused, allowed

    refused, allowed = asyncio.run(drive())
    assert refused == 403, "局域网客户端不得停掉服务（伪造 X-Forwarded-For 也不行）"
    assert allowed["ok"] is True
    assert FakeApp.state.uvicorn_server.should_exit is True


def test_stop_script_asks_for_a_graceful_shutdown(project_root) -> None:
    """``Stop-Process -Force`` never checkpoints the SQLite WAL.

    On Windows there is no SIGTERM, so a hard kill leaves the database unclosed: the
    ``-wal`` file grows to megabytes and ``memory.sqlite3`` stays locked. The stop
    script must therefore *first* ask the service to exit, and only force-kill as a
    fallback — including for a server started on another port.
    """
    script = (project_root / "scripts" / "stop.ps1").read_text(encoding="utf-8-sig")
    assert "/api/shutdown" in script
    assert "Invoke-LocalHttp" in script, "本机请求必须走绕开系统代理的辅助函数"
    common = (project_root / "scripts" / "common.ps1").read_text(encoding="utf-8-sig")
    assert "UseProxy = $false" in common


def test_api_chat_non_streaming(api_client) -> None:
    response = api_client.post(
        "/api/chat",
        json={"message": "你好，你在吗？", "stream": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"].strip()
    assert body["stats"]["model"]


def test_api_chat_streaming_sse(api_client) -> None:
    with api_client.stream(
        "POST", "/api/chat", json={"message": "我记得我喜欢喝什么吗？", "stream": True}
    ) as response:
        assert response.status_code == 200
        payload = "".join(response.iter_text())
    assert "event: session" in payload
    assert "event: done" in payload
    assert '"kind": "text"' in payload or '"kind":"text"' in payload


def test_api_memory_crud_and_search(api_client) -> None:
    created = api_client.post(
        "/api/memory/items",
        json={"content": "用户的猫叫布丁", "kind": "fact", "importance": 0.8},
    ).json()
    assert created["created"] is True
    memory_id = created["id"]

    items = api_client.get("/api/memory/items").json()["items"]
    assert any(item["id"] == memory_id for item in items)

    updated = api_client.patch(
        f"/api/memory/items/{memory_id}", json={"importance": 0.95, "content": "用户的猫叫布丁，是一只橘猫"}
    ).json()
    assert updated["updated"] is True

    search = api_client.get("/api/memory/search", params={"q": "猫叫什么"}).json()
    assert "memories" in search

    assert api_client.delete(f"/api/memory/items/{memory_id}").json()["deleted"] is True
    assert api_client.delete(f"/api/memory/items/{memory_id}?hard=true").json()["deleted"] is True


def test_api_profile_endpoints(api_client) -> None:
    assert api_client.put(
        "/api/memory/profile", json={"key": "名字", "value": "小明", "confidence": 0.9}
    ).json()["ok"] is True
    profile = api_client.get("/api/memory/profile").json()["profile"]
    assert any(entry["key"] == "名字" and entry["value"] == "小明" for entry in profile)
    assert api_client.delete("/api/memory/profile", params={"key": "名字"}).json()["ok"] is True


def test_api_tts_returns_audio(api_client) -> None:
    """The TTS endpoint must return real, non-empty audio of the declared type.

    Either a WAV (``RIFF``) or an MP3 (``ID3``/frame sync) is acceptable — which one
    depends on the backend the router picks. What is *not* acceptable is the failure
    mode this test exists to catch: a 200 response carrying a 46-byte header-only WAV
    with zero audio frames, which plays as silence and hides a broken engine.
    """
    response = api_client.post("/api/voice/tts", json={"text": "测试语音合成"})
    assert response.status_code == 200
    body = response.content
    content_type = response.headers["content-type"]
    assert content_type.startswith("audio/"), content_type
    assert len(body) > 1000, f"音频过短（{len(body)} 字节），可能是空音频"
    if body[:4] == b"RIFF":
        assert content_type.startswith("audio/wav")
    elif body[:3] == b"ID3" or body[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        assert "mpeg" in content_type or "mp3" in content_type
    else:
        raise AssertionError(f"既不是 WAV 也不是 MP3，前 8 字节：{body[:8]!r}")


def test_speech_is_chunked_small_enough_to_start_early() -> None:
    """The chunk budget *is* the user's wait before the first word.

    Measured on this machine: one request for an 80-character reply took edge-tts 13.4s,
    and nothing was audible until it finished. Speech is synthesized chunk by chunk, so
    the budget must stay small (the default of 120 characters produced exactly one
    chunk — i.e. no streaming at all).
    """
    from core.utils import speakable_text, split_sentences

    reply = (
        "你这个问题我分三步说。首先确认模型服务在跑，端口是八零八零。"
        "然后看显存够不够，九B模型大概占八点七G。最后再检查网络，因为在线音色要走外网。"
    )
    text = speakable_text(reply)
    assert len(split_sentences(text, max_chars=120)) == 1, "旧预算下整段只有一块，等于没有流式"

    blocks = split_sentences(text, max_chars=40)
    assert len(blocks) >= 2, f"40 字预算应切出多块：{blocks}"
    assert all(len(block) <= 40 for block in blocks)
    assert "".join(blocks) == text.replace(" ", "")


def _parse_tts_frames(raw: bytes) -> List[tuple]:
    """Decode the length-prefixed frames the streaming endpoint emits."""
    import struct

    frames = []
    while len(raw) >= 5:
        length = struct.unpack(">I", raw[:4])[0]
        frames.append((raw[4:5], raw[5 : 5 + length]))
        raw = raw[5 + length :]
    assert not raw, f"残留 {len(raw)} 字节：帧长度与实际不符"
    return frames


def test_api_tts_stream_delivers_audio_before_finishing(api_client) -> None:
    """The progressive endpoint must send audio frames, in order, with metadata.

    Without it the browser had to wait for the entire reply (13.4s measured) before
    playing a single word.
    """
    payload = {
        "text": (
            "你这个问题我分三步说。首先确认模型服务在跑，端口是八零八零。"
            "然后看显存够不够，九B模型大概占八点七G。最后再检查网络，因为在线音色要走外网。"
        ),
        "persona_id": "sweet_companion",
    }
    with api_client.stream("POST", "/api/voice/tts/stream", json=payload) as response:
        assert response.status_code == 200
        raw = b"".join(response.iter_bytes())

    frames = _parse_tts_frames(raw)
    tags = [tag for tag, _ in frames]
    assert tags[0] == b"M", "首帧应是元数据"
    assert tags[-1] == b"M", "末帧应报告实际使用的后端"
    assert tags.count(b"A") >= 2, f"应至少有两块音频，实际 {tags}"

    first_meta = json.loads(frames[0][1].decode("utf-8"))
    assert first_meta["chunks"] >= 2
    last_meta = json.loads(frames[-1][1].decode("utf-8"))
    assert last_meta["frames"] == tags.count(b"A")
    assert "cache" in last_meta

    audio = b"".join(payload for tag, payload in frames if tag == b"A")
    assert len(audio) > 1000
    assert audio[:4] == b"RIFF" or audio[:3] == b"ID3" or audio[:2] in (b"\xff\xfb", b"\xff\xf3")


def test_tts_router_caches_identical_clips(tmp_path) -> None:
    """Repeated identical text must not pay the backend again.

    The greeting and short acknowledgements are synthesized over and over; with an
    online backend each repeat cost seconds. A cache hit is served from memory in
    microseconds (0.017s measured).
    """
    from core.config import AppConfig, Paths
    from core.types import VoiceSpec
    from speech.tts import ToneTTSBackend, TTSRouter

    router = TTSRouter({"tone": ToneTTSBackend()}, preference=["tone"])
    voice = VoiceSpec(voice_id="default")

    async def run_once() -> bytes:
        chunks = [chunk async for chunk in router.synthesize("你好呀", voice, stream=False)]
        return b"".join(chunk.data for chunk in chunks)

    first = asyncio.run(run_once())
    second = asyncio.run(run_once())
    assert first == second and len(first) > 44
    assert router.cache_stats()["hits"] == 1
    assert router.cache_stats()["entries"] == 1

    # A different voice must not reuse the other voice's audio.
    other = VoiceSpec(voice_id="default", speed=1.4)
    asyncio.run(_collect(router, "你好呀", other))
    assert router.cache_stats()["entries"] == 2, "缓存键必须包含音色参数"

    # Streaming bypasses the cache: progressive output is the point of that path.
    asyncio.run(_collect(router, "你好呀", voice, stream=True))
    assert router.cache_stats()["hits"] == 1


async def _collect(router: Any, text: str, voice: Any, *, stream: bool = False) -> bytes:
    chunks = [chunk async for chunk in router.synthesize(text, voice, stream=stream)]
    return b"".join(chunk.data for chunk in chunks)


def test_indextts_bridge_talks_to_a_worker_in_its_own_environment(tmp_path) -> None:
    """IndexTTS2 must run out-of-process.

    ``indextts`` pins ``torch==2.8``, ``numpy==2.2.6``, ``keras``, ``opencv``… Installing
    that into the app's environment would downgrade numpy under the running process, so
    the backend is expected to drive a JSON worker in ``venvs/tts`` instead. This test
    speaks the protocol with a stub worker (no 12GB install required) and asserts the
    audio comes back through the bridge.
    """
    import sys
    from pathlib import Path as _Path

    from core.types import VoiceSpec
    from speech.tts import IndexTTSBackend

    stub = tmp_path / "stub_worker.py"
    stub.write_text(
        "import json, struct, sys\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    cmd = request.get('cmd')\n"
        "    if cmd == 'synthesize':\n"
        "        pcm = b'\\x00\\x00' * 4000\n"
        "        header = b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 1, 1, 22050, 44100, 2, 16) + b'data' + struct.pack('<I', len(pcm))\n"
        "        open(request['output'], 'wb').write(header + pcm)\n"
        "        result = {'ok': True, 'ms': 1.5, 'bytes': len(pcm)}\n"
        "    elif cmd == 'ping':\n"
        # The health check requires the model package to be importable, not just torch:
        # an environment with torch but no indextts would otherwise look healthy.
        "        result = {'ok': True, 'cuda': True, 'torch': 'stub', 'loaded': False, 'indextts': True}\n"
        "    elif cmd == 'unload':\n"
        "        result = {'ok': True}\n"
        "    else:\n"
        "        result = {'ok': True}\n"
        "    result['id'] = request.get('id')\n"
        "    print(json.dumps(result), flush=True)\n",
        encoding="utf-8",
    )
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"RIFF" + b"\x00" * 200)

    backend = IndexTTSBackend(
        model_dir=str(tmp_path),
        python_exe=sys.executable,          # any interpreter works for the stub
        worker_script=str(stub),
        default_reference=str(reference),
        worker_timeout_s=30,
    )
    assert backend.uses_worker is True

    async def collect() -> bytes:
        chunks = [
            chunk
            async for chunk in backend.synthesize("你好呀", VoiceSpec(voice_id="clone"))
        ]
        return b"".join(chunk.data for chunk in chunks)

    audio = asyncio.run(collect())
    assert audio[:4] == b"RIFF" and len(audio) > 8000, "音频必须经 worker 原样返回"

    health = asyncio.run(backend.health())
    assert health["bridge"] == "worker" and health["ok"] is True
    assert health["torch"] == "stub"

    backend.stop_worker()
    assert backend._worker is None

    # A worker that cannot even start must surface as a readable audio error, not a hang.
    broken = IndexTTSBackend(
        model_dir=str(tmp_path),
        python_exe=str(_Path(sys.executable).parent / "definitely-missing-python.exe"),
        worker_script=str(stub),
        default_reference=str(reference),
    )
    assert broken.uses_worker is False, "解释器不存在时应回退到进程内模式"


def test_indextts_worker_missing_reference_is_reported_before_speaking(tmp_path) -> None:
    """No reference clip means cloning is impossible: say so up front."""
    import sys

    from core.errors import AudioError
    from core.types import VoiceSpec
    from speech.tts import IndexTTSBackend

    stub = tmp_path / "stub_worker.py"
    stub.write_text("import sys\nfor line in sys.stdin: print('{\"ok\": true}', flush=True)\n", encoding="utf-8")
    backend = IndexTTSBackend(
        model_dir=str(tmp_path),
        python_exe=sys.executable,
        worker_script=str(stub),
        default_reference=None,
    )

    async def collect() -> bytes:
        chunks = [chunk async for chunk in backend.synthesize("你好", VoiceSpec(voice_id="x"))]
        return b"".join(chunk.data for chunk in chunks)

    with pytest.raises(AudioError) as excinfo:
        asyncio.run(collect())
    assert "参考音频" in excinfo.value.message
    backend.stop_worker()


def test_tts_router_orders_by_quality_not_offline() -> None:
    """Regression: "offline" must not outrank voice quality.

    ``speech.prefer_offline`` originally sorted offline backends to the front, which
    promoted Windows SAPI (offline, but a 1990s voice) above edge-tts (online, good
    Chinese quality). The router then returned a header-only WAV. Quality priority is
    now the primary key, and offline-ness is only a tie-break — this test pins that
    ordering so the regression cannot come back silently.
    """
    from core.types import SpeechChunk, VoiceSpec
    from speech.tts import TTSRouter, ToneTTSBackend

    class FakeBackend(ToneTTSBackend):
        def __init__(self, name: str, offline: bool) -> None:
            super().__init__()
            self.name = name
            self.offline = offline
            self.used = False

        async def synthesize(self, text, voice, *, stream=False):
            self.used = True
            async for chunk in super().synthesize(text, voice, stream=stream):
                yield chunk

    good_online = FakeBackend("good_online", offline=False)
    dated_offline = FakeBackend("dated_offline", offline=True)
    router = TTSRouter(
        {"good_online": good_online, "dated_offline": dated_offline},
        preference=["good_online", "dated_offline"],
        prefer_offline=True,  # deliberately the setting that used to break it
        priorities={"good_online": 20, "dated_offline": 90},
    )

    async def scenario():
        selected = await router.select(VoiceSpec())
        chunks = [chunk async for chunk in router.synthesize("测试", VoiceSpec())]
        return selected.name, chunks

    selected, chunks = asyncio.run(scenario())
    assert selected == "good_online", f"应选择音质更好的后端，实际 {selected}"
    assert good_online.used and not dated_offline.used
    assert chunks and len(chunks[0].data) > 1000

    # And the tie-break still works: equal priority -> the offline one wins.
    a_online = FakeBackend("a_online", offline=False)
    b_offline = FakeBackend("b_offline", offline=True)
    tie = TTSRouter(
        {"a_online": a_online, "b_offline": b_offline},
        preference=["a_online", "b_offline"],
        prefer_offline=True,
        priorities={"a_online": 20, "b_offline": 20},
    )
    assert asyncio.run(tie.select(VoiceSpec())).name == "b_offline", (
        "同优先级时应按 prefer_offline 选择离线后端"
    )


def test_api_settings_patch_is_persisted(api_client, temp_config) -> None:
    response = api_client.patch("/api/config", json={"debug": True, "llm": {"temperature": 0.33}})
    assert response.status_code == 200
    assert response.json()["applied"]["llm.temperature"] == 0.33
    local = temp_config.paths.config_dir / "local.yaml"
    assert local.exists()
    assert "temperature" in local.read_text(encoding="utf-8")


def test_api_error_mapping(api_client) -> None:
    # Unknown persona is a 404 (strict lookup), not a silent fallback.
    assert api_client.get("/api/personas/不存在的人设").status_code == 404
    # A validation failure on the request body.
    assert api_client.post("/api/chat", json={"message": ""}).status_code == 422
    # Unknown memory id must be an explicit 404, not a 405/500.
    assert api_client.get("/api/memory/items/does-not-exist").status_code == 404
    assert api_client.patch("/api/memory/items/does-not-exist", json={"importance": 0.5}).status_code == 404
    assert api_client.delete("/api/memory/items/does-not-exist").status_code == 404
    # Unknown session is a 404, not an unhandled 500.
    assert api_client.get("/api/sessions/does-not-exist").status_code == 404
    # Missing required query parameter is a validation error.
    assert api_client.get("/api/memory/search").status_code == 422


def test_api_memory_single_item_roundtrip(api_client) -> None:
    created = api_client.post(
        "/api/memory/items", json={"content": "用户的猫叫布丁", "kind": "fact"}
    ).json()
    memory_id = created["id"]
    fetched = api_client.get(f"/api/memory/items/{memory_id}")
    assert fetched.status_code == 200
    assert fetched.json()["item"]["content"] == "用户的猫叫布丁"
    assert api_client.delete(f"/api/memory/items/{memory_id}?hard=true").status_code == 200
