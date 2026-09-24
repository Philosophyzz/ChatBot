"""Self-contained smoke check that needs no third-party packages at all.

This exists because the core of the application is deliberately dependency-free:
configuration, the plugin registry, the memory store, retrieval, extraction and
the chat engine all work with the standard library alone (numpy/httpx/yaml are
optional accelerators with fallbacks). Being able to prove that on a bare Python
install is worth a dedicated entry point — it isolates "the logic is broken" from
"a dependency is missing".

Run::

    python tests/smoke_stdlib.py

Exit code 0 means every layer below the HTTP surface produced correct results.
"""

from __future__ import annotations

import asyncio
import gc
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str):
    """Decorator: run a sync check and record its outcome."""

    def wrapper(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  [OK]   {name}")
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
        return fn

    return wrapper


def acheck(name: str):
    """Decorator for async checks."""

    def wrapper(fn):
        try:
            asyncio.run(fn())
            PASSED.append(name)
            print(f"  [OK]   {name}")
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
        return fn

    return wrapper


print("=" * 64)
print(" 标准库自检（不依赖任何第三方包）")
print("=" * 64)

TMP = Path(tempfile.mkdtemp(prefix="chatbot-smoke-"))

# ---------------------------------------------------------------------------- config
from core.config import AppConfig, Paths, load_config  # noqa: E402
from core.errors import AudioError, BackendUnavailable, ChatBotError  # noqa: E402
from core.registry import (  # noqa: E402
    KIND_EMBEDDER,
    KIND_LLM,
    KIND_RERANKER,
    KIND_STT,
    KIND_TTS,
    KIND_VECTOR_STORE,
    registry,
)
from core.session import SessionManager  # noqa: E402
from core.types import (  # noqa: E402
    AudioClip,
    LLMRequest,
    MemoryKind,
    Message,
    Role,
    RetrievalQuery,
    VoiceSpec,
    now_ts,
)
from core.utils import estimate_messages_tokens, estimate_tokens, speakable_text, split_sentences  # noqa: E402


@check("配置：默认值与路径派生")
def _config_defaults() -> None:
    config = AppConfig(paths=Paths(root=TMP))
    assert config.paths.gguf_dir == TMP / "models" / "gguf"
    assert config.llm.context_tokens > 0
    assert config.memory.dim > 0
    config.paths.ensure()
    assert config.paths.data_dir.exists()


@check("配置：模型路径越界会被拒绝（D 盘契约）")
def _config_path_guard() -> None:
    config = AppConfig(paths=Paths(root=TMP))
    try:
        config.paths.assert_on_project_root(TMP.parent / "outside.gguf")
    except ValueError:
        return
    raise AssertionError("越界路径未被拒绝")


@check("配置：极简 YAML 解析器与真实文件")
def _config_yaml() -> None:
    shutil.copy(ROOT / "config" / "config.yaml", TMP / "config" / "config.yaml")
    config = load_config(TMP)
    assert config.llm.base_url.startswith("http")
    assert config.speech.tts_preference, "tts_preference 不应为空"
    assert config.memory.token_budget > 0


@check("工具：token 估算与文本清洗")
def _utils() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("中文测试") >= 4
    assert estimate_tokens("hello world") > 0
    cleaned = speakable_text("**粗体** 和 `代码` 以及 ```块```")
    for bad in ("**", "`"):
        assert bad not in cleaned, bad
    parts = split_sentences("第一句。第二句！第三句？")
    assert len(parts) >= 1
    messages = [Message(role=Role.SYSTEM, content="系统"), Message(role=Role.USER, content="你好")]
    assert estimate_messages_tokens(messages) > 0


# ---------------------------------------------------------------------------- registry
import plugs  # noqa: E402,F401  (registers built-ins)


@check("注册表：内置插件都已注册")
def _registry() -> None:
    expected = {
        KIND_LLM: {"mock", "llamacpp", "ollama", "openai", "vllm"},
        KIND_EMBEDDER: {"hash", "llamacpp"},
        KIND_RERANKER: {"heuristic", "none"},
        KIND_VECTOR_STORE: {"sqlite", "memory"},
        KIND_STT: {"mock"},
    }
    for kind, names in expected.items():
        available = set(registry.names(kind))
        missing = names - available
        assert not missing, f"{kind} 缺少 {missing}（已注册：{sorted(available)}）"


@check("注册表：未知插件给出可读错误")
def _registry_error() -> None:
    try:
        registry.create(KIND_LLM, "nope")
    except KeyError as exc:
        assert "nope" in str(exc)
        return
    raise AssertionError("未知插件未报错")


# ---------------------------------------------------------------------------- memory
from memory.db import Database  # noqa: E402
from memory.engine import MemoryEngine  # noqa: E402
from memory.extract import heuristic_extract, parse_json_loose  # noqa: E402
from memory.store import MemoryStore  # noqa: E402
from memory.vector import SqliteVectorStore  # noqa: E402
from llm.embed import HashingEmbedder, HeuristicReranker  # noqa: E402
from llm.mock import MockLLMBackend  # noqa: E402

DB = Database(TMP / "data" / "smoke.sqlite3")
DB.init()
STORE = MemoryStore(DB)
EMBEDDER = HashingEmbedder(dim=256)
VECTORS = SqliteVectorStore(DB)
LLM = MockLLMBackend(delay_ms=0)


@check("存储：schema 与 FTS 可用性")
def _schema() -> None:
    tables = {
        row[0] for row in DB.query("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }
    for table in (
        "conversations",
        "messages",
        "memory_items",
        "entities",
        "relations",
        "profiles",
        "memory_vectors",
        "turn_log",
    ):
        assert table in tables, f"缺少表 {table}"
    print(f"         FTS5: {'启用' if DB.fts_enabled else '不可用（已降级）'}")


@acheck("存储：消息、记忆、会话读写")
async def _store_roundtrip() -> None:
    await STORE.ensure_conversation("c1", title="冒烟测试", persona_id="sweet_companion")
    await STORE.add_message("c1", Message(role=Role.USER, content="我喜欢喝不加糖的美式咖啡"))
    await STORE.add_message("c1", Message(role=Role.ASSISTANT, content="记住啦～"))
    messages = await STORE.get_messages("c1")
    assert len(messages) == 2
    first = await STORE.add_memory(content="用户喜欢美式咖啡", kind=MemoryKind.PREFERENCE, subject="用户")
    again = await STORE.add_memory(content="用户喜欢美式咖啡", kind=MemoryKind.PREFERENCE, subject="用户")
    assert first and again is None, "重复记忆应被去重"
    stats = await STORE.stats()
    assert stats["memories"] == 1 and stats["messages"] == 2


@acheck("存储：中文全文检索")
async def _fts() -> None:
    hits = await STORE.search_messages("咖啡")
    assert hits, "FTS/LIKE 应能命中中文子串"
    assert "咖啡" in hits[0]["content"]


@acheck("存储：取代链关闭旧记忆有效期")
async def _supersede() -> None:
    # Identical text on purpose: an explicit supersede must create a new row even when
    # a content-hash dedupe would otherwise fold it away.
    old = await STORE.add_memory(
        content="用户住在北京", kind=MemoryKind.PROFILE, subject="用户", predicate="所在城市", object="北京"
    )
    new = await STORE.add_memory(
        content="用户住在北京",
        kind=MemoryKind.PROFILE,
        subject="用户",
        predicate="所在城市",
        object="上海",
        supersedes=old,
    )
    assert new is not None, "显式取代不能被内容去重吞掉"
    old_item = await STORE.get_memory(old)
    # Scope the assertion to the two rows this check owns: earlier checks in this
    # script share the store, and a global "everything active" assertion would fail
    # for reasons that have nothing to do with supersession.
    active = await STORE.list_memories(kinds=[MemoryKind.PROFILE])
    active_ids = {item.id for item in active}
    assert old_item.superseded_by == new
    assert old_item.valid_to is not None, "被取代的记忆必须关闭有效期窗口"
    assert new in active_ids, "新记忆必须处于生效状态"
    assert old not in active_ids, "被取代的旧记忆不得再出现在生效集合中"


@acheck("向量：检索排序与删除")
async def _vectors() -> None:
    await VECTORS.upsert(["a", "b", "c"], [[1.0, 0, 0], [0.9, 0.1, 0], [0, 0, 1.0]], [{}, {}, {}])
    results = await VECTORS.search([1.0, 0.0, 0.0], top_k=3)
    assert [item[0] for item in results][:2] == ["a", "b"], results
    assert results[0][1] > results[2][1]
    await VECTORS.delete(["a"])
    assert await VECTORS.count() == 2


@check("抽取：容错 JSON 解析")
def _json_loose() -> None:
    assert parse_json_loose('```json\n{"facts": []}\n```') == {"facts": []}
    assert parse_json_loose('前缀 {"facts":[{"content":"x"}]} 后缀')["facts"][0]["content"] == "x"
    assert parse_json_loose("不是 JSON") is None


@check("抽取：规则回退只抓明确陈述")
def _heuristic() -> None:
    payload = heuristic_extract("我叫小明，我喜欢喝美式咖啡")
    contents = " ".join(fact["content"] for fact in payload["facts"])
    assert "小明" in contents
    assert heuristic_extract("今天天气不错")["facts"] == []


@acheck("抽取：写入路径产出事实与图谱节点")
async def _extraction() -> None:
    engine = await MemoryEngine.create(
        AppConfig(paths=Paths(root=TMP)), llm=LLM, db=DB, embedder=EMBEDDER, vectors=VECTORS,
        reranker=HeuristicReranker(),
    )
    await engine.store.ensure_conversation("c2")
    outcome = await engine.remember_turn(
        session_id="c2",
        user_message=Message(role=Role.USER, content="我叫小明，喜欢美式咖啡，在做本地聊天机器人项目"),
        assistant_message=Message(role=Role.ASSISTANT, content="好的"),
        wait=True,
    )
    assert outcome.created >= 1, outcome.to_public()
    entities = await engine.store.list_entities()
    assert entities, "应建立图谱节点"
    profile = await engine.store.get_profile()
    assert profile, "应产出档案字段"
    await engine.close()


@acheck("检索：混合召回返回可解释的评分")
async def _retrieval() -> None:
    engine = await MemoryEngine.create(
        AppConfig(paths=Paths(root=TMP)), llm=LLM, db=DB, embedder=EMBEDDER, vectors=VECTORS,
        reranker=HeuristicReranker(),
    )
    context = await engine.build_context("我平时喝什么咖啡？", session_id="s1")
    assert context.rendered, "应渲染出记忆块"
    assert "咖啡" in context.rendered, context.rendered[:200]
    assert context.hits and context.hits[0].parts, "命中应带评分拆解"
    await engine.close()


@acheck("检索：token 预算被严格遵守")
async def _budget() -> None:
    engine = await MemoryEngine.create(
        AppConfig(paths=Paths(root=TMP)), llm=LLM, db=DB, embedder=EMBEDDER, vectors=VECTORS,
        reranker=HeuristicReranker(),
    )
    for index in range(40):
        memory_id = await engine.store.add_memory(
            content=f"用户的第 {index} 条关于咖啡与茶的琐碎偏好，刻意写长以占用预算" * 3,
            kind=MemoryKind.FACT,
            subject="用户",
            importance=0.5,
        )
        vector = await engine.embedder.embed_one(f"咖啡茶饮 {index}")
        await engine.vectors.upsert([memory_id], [vector], [{}])
    context = await engine.build_context("咖啡", session_id="s1", token_budget=100)
    assert context.tokens_used <= 100, context.tokens_used
    await engine.close()


# ---------------------------------------------------------------------------- engine
from core.chat import ChatEngine  # noqa: E402
from persona.manager import PersonaManager, build_system_prompt  # noqa: E402

CONFIG = AppConfig(paths=Paths(root=TMP))


@check("人设：内置人设具备提示词与音色")
def _personas() -> None:
    manager = PersonaManager(CONFIG)
    personas = manager.list()
    assert {p.id for p in personas} >= {"sakura_cat", "mint_bunny", "luna_witch"}
    for persona in personas:
        assert persona.system_prompt.strip()
        assert persona.voice.voice_id
    assert manager.get(None).id == CONFIG.default_persona
    assert manager.get("不存在的人设").id == CONFIG.default_persona


@check("人设：系统提示词装配包含记忆与语音约束")
def _prompt_assembly() -> None:
    manager = PersonaManager(CONFIG)
    prompt = build_system_prompt(
        manager.get("sweet_companion"), memory_block="【长期记忆】\n- 用户喜欢美式咖啡", voice_mode=True
    )
    assert "美式咖啡" in prompt
    assert "当前时间" in prompt
    assert "语音朗读" in prompt


@acheck("引擎：一轮对话产出完整事件序列")
async def _engine_turn() -> None:
    engine = await MemoryEngine.create(
        AppConfig(paths=Paths(root=TMP)), llm=LLM, db=DB, embedder=EMBEDDER, vectors=VECTORS,
        reranker=HeuristicReranker(),
    )
    chat = ChatEngine(
        config=CONFIG, llm=LLM, personas=PersonaManager(CONFIG), memory=engine, sessions=SessionManager(engine.store)
    )
    kinds: list[str] = []
    text = ""
    async for event in chat.stream_turn(session_id=None, user_text="你好呀"):
        kinds.append(event["kind"])
        if event["kind"] == "text":
            text += event["text"]
    assert kinds[0] == "session", kinds
    assert "start" in kinds and "text" in kinds and kinds[-1] == "done", kinds
    assert "error" not in kinds, kinds
    assert text.strip()
    await engine.close()


@acheck("引擎：跨会话记忆召回")
async def _cross_session() -> None:
    engine = await MemoryEngine.create(
        AppConfig(paths=Paths(root=TMP)), llm=LLM, db=DB, embedder=EMBEDDER, vectors=VECTORS,
        reranker=HeuristicReranker(),
    )
    chat = ChatEngine(
        config=CONFIG, llm=LLM, personas=PersonaManager(CONFIG), memory=engine, sessions=SessionManager(engine.store)
    )
    result = await chat.complete_turn(session_id=None, user_text="我叫小明，我喜欢不加糖的美式咖啡")
    session_id = None
    for event in result.events:
        if event["kind"] == "done":
            session_id = event["session_id"]
    assert session_id, "应返回会话 id"
    session = await chat.sessions.get(session_id)
    await engine._run_extraction(
        {
            "session_id": session_id,
            "user_message": session.messages[-2],
            "assistant_message": session.messages[-1],
            "persona_id": "sweet_companion",
            "scope": "global",
        }
    )
    context = await engine.build_context("我喜欢喝什么咖啡？", session_id="另一个会话")
    assert "咖啡" in context.rendered, context.rendered[:300]
    await engine.close()


@acheck("引擎：历史裁剪不超上下文预算")
async def _trimming() -> None:
    from core.session import DEFAULT_WINDOW

    config = AppConfig(paths=Paths(root=TMP))
    config.llm.context_tokens = 4000
    config.llm.reserve_tokens = 800
    engine = await MemoryEngine.create(
        config, llm=LLM, db=DB, embedder=EMBEDDER, vectors=VECTORS, reranker=HeuristicReranker()
    )
    chat = ChatEngine(
        config=config, llm=LLM, personas=PersonaManager(config), memory=engine, sessions=SessionManager(engine.store)
    )
    session = await chat.sessions.create(persona_id="sweet_companion")
    # Build history directly: driving real turns would make the token total depend
    # on how verbose the mock model happens to be, which is not what is under test.
    for index in range(40):
        session.append(Message(role=Role.USER, content=f"第 {index} 条很长的测试消息" * 8))
        session.append(Message(role=Role.ASSISTANT, content=f"这是第 {index} 条回复" * 8))
    assert len(session.messages) == DEFAULT_WINDOW, "会话窗口应裁剪到 DEFAULT_WINDOW"
    prompt = await chat.build_prompt(session, "总结一下")
    total = estimate_messages_tokens(prompt["messages"])
    assert total <= config.llm.context_tokens, f"{total} tokens 超过 {config.llm.context_tokens}"
    assert prompt["messages"][0].role == Role.SYSTEM
    assert prompt["messages"][-1].content == "总结一下"
    assert any("未展示" in message.content for message in prompt["messages"]), "应留下裁剪提示"
    await engine.close()


@acheck("引擎：后端故障转为可读错误事件")
async def _backend_failure() -> None:
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

    chat = ChatEngine(
        config=CONFIG, llm=FailingLLM(), personas=PersonaManager(CONFIG), memory=None, sessions=SessionManager(None)
    )
    kinds = [event["kind"] async for event in chat.stream_turn(session_id=None, user_text="在吗")]
    assert "error" in kinds, kinds


# ---------------------------------------------------------------------------- memory hygiene
from core.utils import truncate_to_tokens  # noqa: E402


@acheck("健壮性：嵌入服务挂掉后记忆仍可用")
async def _degraded_embedder() -> None:
    class BrokenEmbedder:
        name = "broken"
        dim = 256

        async def embed(self, texts, **kwargs):
            raise RuntimeError("嵌入服务不可用")

        async def embed_one(self, text, **kwargs):
            raise RuntimeError("嵌入服务不可用")

        async def health(self):
            return {"ok": False, "error": "down"}

    db = Database(TMP / "data" / "degraded.sqlite3")
    db.init()
    store = MemoryStore(db)
    engine = await MemoryEngine.create(
        AppConfig(paths=Paths(root=TMP)),
        llm=LLM,
        db=db,
        embedder=BrokenEmbedder(),
        vectors=SqliteVectorStore(db),
        reranker=HeuristicReranker(),
    )
    await store.add_memory(content="用户喜欢喝咖啡", kind=MemoryKind.PREFERENCE, subject="用户")
    context = await engine.build_context("咖啡", session_id="s1")
    assert "咖啡" in context.rendered, "词法通道应在嵌入不可用时兜住"
    await engine.close()
    db.close()


@check("健壮性：超长文本截断到 token 预算")
def _truncate() -> None:
    text = "很长的一段中文测试文本。" * 200
    clipped = truncate_to_tokens(text, 50)
    assert estimate_tokens(clipped) <= 50
    assert len(clipped) < len(text)


# ---------------------------------------------------------------------------- summary
# Close the SQLite handles before deleting the scratch tree.
#
# Windows keeps the database file locked while a connection is open, so
# ``shutil.rmtree(..., ignore_errors=True)`` used to fail *silently*: every run left a
# ``chatbot-smoke-*`` directory behind in %TEMP% with a stray memory.sqlite3 inside.
# The retry loop is the belt to that braces — if a handle survives anyway, the script
# says so instead of leaking quietly.
DB.close()
gc.collect()
for _ in range(10):
    shutil.rmtree(TMP, ignore_errors=True)
    if not TMP.exists():
        break
    time.sleep(0.3)
if TMP.exists():
    print(f"  提示：临时目录未能删除（{TMP}），可能有数据库连接未关闭")

print("")
print("=" * 64)
print(f" 通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
if FAILED:
    for name, error in FAILED:
        print(f"   [FAIL] {name}: {error}")
print("=" * 64)
sys.exit(1 if FAILED else 0)
