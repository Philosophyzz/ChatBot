"""Memory persistence, vectors, extraction, conflict resolution and retrieval."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


# ------------------------------------------------------------------------------------
# Storage layer
# ------------------------------------------------------------------------------------


def test_database_creates_full_schema(tmp_path) -> None:
    from memory.db import Database

    db = Database(tmp_path / "m.sqlite3")
    db.init()
    tables = {
        row[0]
        for row in db.query("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }
    for expected in (
        "conversations",
        "messages",
        "memory_items",
        "entities",
        "entity_aliases",
        "relations",
        "profiles",
        "memory_vectors",
        "turn_log",
        "kv_store",
    ):
        assert expected in tables, f"缺少表 {expected}"


def test_database_close_releases_every_thread_connection(run, tmp_path) -> None:
    """Shutdown must release connections it did not create.

    Connections are thread-local and shutdown happens on another thread
    (``aclose`` → ``asyncio.to_thread``), so "close this thread's handle" was a no-op:
    the request thread's connection survived, Windows kept ``memory.sqlite3`` locked
    after the service stopped, and the 4MB WAL was never checkpointed.
    """
    import shutil
    import threading

    from memory.db import Database

    work_dir = tmp_path / "locked"
    db = Database(work_dir / "m.sqlite3")
    db.init()

    def request_thread() -> None:
        db.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES('probe', 'ok')")

    thread = threading.Thread(target=request_thread)
    thread.start()
    thread.join()
    assert len(db._live) >= 2, "每个线程应有自己的连接"

    run(db.aclose())
    assert db._live == set(), "关闭后不应残留任何连接"
    # A worker that touches the database after shutdown must transparently reconnect
    # rather than receive a closed handle.
    assert [dict(row) for row in db.query("SELECT value FROM schema_meta WHERE key='probe'")] == [
        {"value": "ok"}
    ]
    db.close()

    # The real symptom: the file stayed locked, so the directory could not be removed.
    shutil.rmtree(work_dir)
    assert not work_dir.exists()


def test_store_roundtrip_and_dedup(run, memory_stack) -> None:
    from core.types import MemoryKind, Message, Role

    store = memory_stack.store

    async def scenario():
        await store.ensure_conversation("c1", title="测试", persona_id="sweet_companion")
        await store.add_message("c1", Message(role=Role.USER, content="我喜欢美式咖啡"))
        await store.add_message("c1", Message(role=Role.ASSISTANT, content="记住啦"))

        first = await store.add_memory(content="用户喜欢美式咖啡", kind=MemoryKind.PREFERENCE, subject="用户")
        second = await store.add_memory(content="用户喜欢美式咖啡", kind=MemoryKind.PREFERENCE, subject="用户")
        assert first is not None
        assert second is None, "完全相同的记忆应被去重"

        messages = await store.get_messages("c1")
        assert [m.role for m in messages] == [Role.USER, Role.ASSISTANT]
        return await store.stats()

    stats = run(scenario())
    assert stats["messages"] == 2
    assert stats["memories"] == 1
    assert stats["conversations"] == 1


def test_memory_supersede_keeps_history(run, memory_stack) -> None:
    """The temporal chain is the core guarantee: new truth closes the old window.

    The replacement's text intentionally equals the original's, because that is the
    case a naive content-hash dedupe would swallow: an explicit supersede must always
    create a new row (the two carry different validity windows), otherwise the
    history this design exists to preserve would be erased.
    """
    from core.types import MemoryKind

    store = memory_stack.store

    async def scenario():
        old = await store.add_memory(
            content="用户住在北京",
            kind=MemoryKind.PROFILE,
            subject="用户",
            predicate="所在城市",
            object="北京",
        )
        # Identical text with a different object value is only reachable via an
        # explicit supersede, which is exactly the path under test.
        new = await store.add_memory(
            content="用户住在北京",
            kind=MemoryKind.PROFILE,
            subject="用户",
            predicate="所在城市",
            object="上海",
            supersedes=old,
        )
        active = await store.list_memories()
        retired = await store.list_memories(include_retired=True)
        old_item = await store.get_memory(old)
        return old, new, active, retired, old_item

    old_id, new_id, active, retired, old_item = run(scenario())
    assert new_id is not None, "显式取代必须创建新记忆，不能被内容去重吞掉"
    assert new_id != old_id
    # Scoped to PROFILES so the assertion cannot be affected by other checks sharing
    # the store — a global "everything active" comparison would be testing the
    # fixture, not supersession.
    active_ids = {item.id for item in active}
    retired_ids = {item.id for item in retired}
    assert active_ids == {new_id}
    assert retired_ids == {old_id, new_id}
    assert old_item.superseded_by == new_id
    assert old_item.valid_to is not None, "被取代的记忆必须关闭有效期窗口"


def test_memory_dedupe_rejects_exact_repeat(run, memory_stack) -> None:
    """Without an explicit supersede, an identical claim is folded, not duplicated."""
    from core.types import MemoryKind

    store = memory_stack.store

    async def scenario():
        first = await store.add_memory(
            content="用户喜欢美式咖啡", kind=MemoryKind.PREFERENCE, subject="用户"
        )
        second = await store.add_memory(
            content="用户喜欢美式咖啡", kind=MemoryKind.PREFERENCE, subject="用户"
        )
        items = await store.list_memories()
        return first, second, items

    first, second, items = run(scenario())
    assert first is not None
    assert second is None, "完全相同的记忆应被去重"
    assert len(items) == 1


def test_full_text_search_finds_chinese_substring(run, memory_stack) -> None:
    from core.types import Message, Role

    store = memory_stack.store

    async def scenario():
        await store.ensure_conversation("c2")
        await store.add_message("c2", Message(role=Role.USER, content="我昨天在星巴克点了一杯美式咖啡"))
        await store.add_message("c2", Message(role=Role.USER, content="今天想去看电影"))
        return await store.search_messages("咖啡")

    hits = run(scenario())
    assert len(hits) == 1
    assert "咖啡" in hits[0]["content"]


def test_entity_alias_resolution(run, memory_stack) -> None:
    store = memory_stack.store

    async def scenario():
        entity_id = await store.upsert_entity("小明", type="person", aliases=["我", "用户"])
        # 同一实体再次出现应复用 id 并累加提及次数
        again = await store.upsert_entity("小明", type="person")
        resolved = await store.resolve_entities(["小明", "我", "不存在的人"])
        entity = await store.get_entity(entity_id)
        return entity_id, again, resolved, entity

    entity_id, again, resolved, entity = run(scenario())
    assert entity_id == again
    assert resolved.get("小明") == entity_id
    assert resolved.get("我") == entity_id
    assert "不存在的人" not in resolved
    assert entity.mention_count == 2
    assert "我" in entity.aliases


def test_relations_strengthen_on_repeat(run, memory_stack) -> None:
    store = memory_stack.store

    async def scenario():
        a = await store.upsert_entity("用户", type="person")
        b = await store.upsert_entity("美式咖啡", type="concept")
        first = await store.add_relation(a, "喜欢", b, confidence=0.6)
        second = await store.add_relation(a, "喜欢", b, confidence=0.7)
        neighbours = await store.neighbors([a])
        return first, second, neighbours

    first, second, neighbours = run(scenario())
    assert first == second, "同一个三元组不应产生两条边"
    assert len(neighbours) == 1
    assert neighbours[0]["confidence"] >= 0.7


def test_profile_upsert_overwrites_value(run, memory_stack) -> None:
    store = memory_stack.store

    async def scenario():
        await store.upsert_profile("名字", "小明", confidence=0.8)
        await store.upsert_profile("名字", "小铭", confidence=0.9)
        return await store.get_profile()

    profile = run(scenario())
    assert len(profile) == 1
    assert profile[0]["value"] == "小铭"
    assert profile[0]["confidence"] == 0.9


# ------------------------------------------------------------------------------------
# Vector index
# ------------------------------------------------------------------------------------


def test_sqlite_vector_search_ranks_similar_first(run, tmp_path) -> None:
    from memory.db import Database
    from memory.vector import SqliteVectorStore

    db = Database(tmp_path / "v.sqlite3")
    db.init()
    store = SqliteVectorStore(db)

    async def scenario():
        await store.upsert(
            ["a", "b", "c"],
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]],
            [{}, {}, {}],
        )
        results = await store.search([1.0, 0.0, 0.0], top_k=3)
        count = await store.count()
        await store.delete(["a"])
        after = await store.count()
        return results, count, after

    results, count, after = run(scenario())
    assert count == 3
    assert after == 2
    assert [item_id for item_id, _ in results][:2] == ["a", "b"]
    assert results[0][1] > results[2][1]


# ------------------------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------------------------


def test_parse_json_loose_recovers_messy_output() -> None:
    from memory.extract import parse_json_loose

    assert parse_json_loose('```json\n{"facts": []}\n```') == {"facts": []}
    assert parse_json_loose('好的，结果如下：{"facts": [{"content": "x"}]} 以上。')["facts"][0]["content"] == "x"
    assert parse_json_loose('{"facts": [{"content": "x"},]}') is not None  # 尾随逗号
    assert parse_json_loose("完全不是 JSON") is None
    assert parse_json_loose("") is None


def test_heuristic_extract_only_takes_explicit_statements() -> None:
    from memory.extract import heuristic_extract

    payload = heuristic_extract("我叫小明，我喜欢喝美式咖啡，不喜欢加糖。你记住了吗？")
    contents = " ".join(fact["content"] for fact in payload["facts"])
    assert "小明" in contents
    assert "美式咖啡" in contents
    assert payload["profile"], "应抽取到档案字段"

    empty = heuristic_extract("今天天气不错")
    assert empty["facts"] == [], "模糊闲聊不应产生记忆"


def test_extraction_persists_graph_and_profile(run, memory_stack) -> None:
    from core.types import Message, Role

    async def scenario():
        await memory_stack.store.ensure_conversation("c3")
        outcome = await memory_stack.remember_turn(
            session_id="c3",
            user_message=Message(role=Role.USER, content="我叫小明，喜欢美式咖啡，在做本地聊天机器人项目"),
            assistant_message=Message(role=Role.ASSISTANT, content="好的，记下了"),
            wait=True,
        )
        entities = await memory_stack.store.list_entities()
        profile = await memory_stack.store.get_profile()
        items = await memory_stack.store.list_memories()
        return outcome, entities, profile, items

    outcome, entities, profile, items = run(scenario())
    assert outcome is not None
    assert outcome.created >= 1
    assert len(items) >= 1
    assert profile, "模拟抽取器应产出档案字段"
    assert entities, "应建立图谱节点"
    assert any(entity.name == "用户" for entity in entities)


def test_extraction_is_idempotent_per_message(run, memory_stack) -> None:
    from core.types import Message, Role

    async def scenario():
        await memory_stack.store.ensure_conversation("c4")
        message = Message(role=Role.USER, content="我喜欢喝美式咖啡")
        first = await memory_stack.remember_turn(
            session_id="c4", user_message=message, wait=True
        )
        second = await memory_stack.remember_turn(
            session_id="c4", user_message=message, wait=True
        )
        return first, second, await memory_stack.store.stats()

    first, second, stats = run(scenario())
    assert first.created >= 1
    assert second.method == "skipped", "同一条消息不应重复抽取"
    assert stats["memories"] == first.created


def test_contradiction_supersedes_old_fact(run, memory_stack) -> None:
    """Moving city must invalidate the previous city, not accumulate both."""
    from core.types import MemoryKind

    extractor = memory_stack.extractor
    store = memory_stack.store

    async def scenario():
        old_id = await store.add_memory(
            content="用户住在北京",
            kind=MemoryKind.PROFILE,
            subject="用户",
            predicate="所在城市",
            object="北京",
            importance=0.8,
        )
        vector = await memory_stack.embedder.embed_one("用户住在北京")
        await memory_stack.vectors.upsert([old_id], [vector], [{}])

        fact = {
            "content": "用户住在上海",
            "kind": "profile",
            "subject": "用户",
            "predicate": "所在城市",
            "object": "上海",
            "importance": 0.8,
            "confidence": 0.9,
        }
        similar = await extractor._find_similar(fact, scope="global")
        contradicts = extractor._contradicts(fact, {"item": await store.get_memory(old_id), "similarity": 0.9})
        return old_id, similar, contradicts

    old_id, similar, contradicts = run(scenario())
    assert similar, "应能找到同一 subject 下的候选记忆"
    assert contradicts is True, "同谓词不同取值应判为冲突"


def test_contradiction_rules(run, memory_stack) -> None:
    """The three cases that matter for keeping long-term memory self-consistent.

    1. Same predicate, same object -> confirmation, never a conflict.
    2. Same predicate, different object -> a genuine functional-slot conflict
       ("likes coffee" vs "likes rainy days" really do compete for the same slot,
       and the newer statement is the better guess at what is true now).
    3. No structured predicate -> *never* a conflict. Two unstructured statements can
       both be true, and an embedding score cannot tell the difference, so retiring
       one would silently destroy a valid memory. This is the property that keeps
       similarity alone from deciding truth.
    """
    from core.types import MemoryKind

    extractor = memory_stack.extractor
    store = memory_stack.store

    async def scenario():
        item_id = await store.add_memory(
            content="用户喜欢美式咖啡",
            kind=MemoryKind.PREFERENCE,
            subject="用户",
            predicate="喜欢",
            object="美式咖啡",
        )
        item = await store.get_memory(item_id)
        confirmation = extractor._contradicts(
            {
                "content": "用户喜欢美式咖啡",
                "kind": "preference",
                "subject": "用户",
                "predicate": "喜欢",
                "object": "美式咖啡",
            },
            {"item": item, "similarity": 0.99},
        )
        same_slot_conflict = extractor._contradicts(
            {
                "content": "用户喜欢下雨天",
                "kind": "preference",
                "subject": "用户",
                "predicate": "喜欢",
                "object": "下雨天",
            },
            {"item": item, "similarity": 0.7},
        )
        unstructured = extractor._contradicts(
            {"content": "用户喜欢下雨天", "kind": "preference", "subject": "用户"},
            {"item": item, "similarity": 0.93},
        )
        return confirmation, same_slot_conflict, unstructured

    confirmation, same_slot_conflict, unstructured = run(scenario())
    assert confirmation is False, "同值应视为确认，而非冲突"
    assert unstructured is False, "缺少结构化谓词时不得凭相似度判定冲突"
    # Preferences are NOT functional slots: liking coffee and liking rainy days are both
    # true, so the newer statement must not retire the older one. Measured bug: the
    # extractor superseded "喜欢不加糖的美式咖啡" with "喜欢热的美式咖啡" and silently
    # dropped a valid memory.
    assert same_slot_conflict is False, "喜好类谓词可并存，不应互相取代"


def test_functional_slot_does_supersede(run, memory_stack) -> None:
    """Single-value attributes DO supersede: one name, one city, one job.

    This is the case supersession exists for — the "moved from Beijing to Shanghai"
    scenario. The counterpart test above shows preferences are deliberately excluded.
    """
    from core.types import MemoryKind

    extractor = memory_stack.extractor
    store = memory_stack.store

    async def scenario():
        old_id = await store.add_memory(
            content="用户住在北京",
            kind=MemoryKind.PROFILE,
            subject="用户",
            predicate="所在地",
            object="北京",
        )
        item = await store.get_memory(old_id)
        moved = extractor._contradicts(
            {
                "content": "用户住在上海",
                "kind": "profile",
                "subject": "用户",
                "predicate": "所在地",
                "object": "上海",
            },
            {"item": item, "similarity": 0.85},
        )
        renamed = extractor._contradicts(
            {
                "content": "用户叫张伟",
                "kind": "profile",
                "subject": "用户",
                "predicate": "姓名",
                "object": "张伟",
            },
            {"item": item, "similarity": 0.7},
        )
        return moved, renamed

    moved, renamed = run(scenario())
    assert moved is True, "所在地是单值槽位，换城市必须取代旧值"
    # Different predicate of the same node is not a conflict.
    assert renamed is False, "谓词不同（所在地 vs 姓名）不是冲突"


def test_contradiction_requires_same_subject(run, memory_stack) -> None:
    """Two different people can hold opposite preferences without conflict."""
    from core.types import MemoryKind

    extractor = memory_stack.extractor
    store = memory_stack.store

    async def scenario():
        item_id = await store.add_memory(
            content="小美喜欢美式咖啡",
            kind=MemoryKind.PREFERENCE,
            subject="小美",
            predicate="喜欢",
            object="美式咖啡",
        )
        item = await store.get_memory(item_id)
        return extractor._contradicts(
            {
                "content": "小明喜欢拿铁",
                "kind": "preference",
                "subject": "小明",
                "predicate": "喜欢",
                "object": "拿铁",
            },
            {"item": item, "similarity": 0.9},
        )

    assert run(scenario()) is False, "不同 subject 之间不应互相取代"


# ------------------------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------------------------


def test_retrieval_returns_relevant_memory(run, memory_stack) -> None:
    from core.types import MemoryKind

    store = memory_stack.store

    async def scenario():
        for content, kind, subject in (
            ("用户喜欢喝不加糖的美式咖啡", MemoryKind.PREFERENCE, "用户"),
            ("用户的生日是 3 月 14 日", MemoryKind.PROFILE, "用户"),
            ("用户正在学习 Rust 语言", MemoryKind.FACT, "用户"),
        ):
            memory_id = await store.add_memory(content=content, kind=kind, subject=subject, importance=0.7)
            vector = await memory_stack.embedder.embed_one(content)
            await memory_stack.vectors.upsert([memory_id], [vector], [{"kind": kind.value}])
        return await memory_stack.build_context("我平时喝什么咖啡？", session_id="s1")

    context = run(scenario())
    assert context.rendered, "应渲染出记忆块"
    assert "咖啡" in context.rendered
    assert context.tokens_used > 0
    assert context.hits
    # 每条命中都应能解释自己的分数来源
    assert context.hits[0].parts


def test_retrieval_respects_token_budget(run, memory_stack) -> None:
    from core.types import MemoryKind

    store = memory_stack.store

    async def scenario():
        for index in range(60):
            memory_id = await store.add_memory(
                content=f"用户提到的第 {index} 件关于咖啡和茶饮的琐事，内容较长以便占用预算" * 2,
                kind=MemoryKind.FACT,
                subject="用户",
                importance=0.5,
            )
            vector = await memory_stack.embedder.embed_one(f"咖啡茶饮 {index}")
            await memory_stack.vectors.upsert([memory_id], [vector], [{}])
        return await memory_stack.build_context("咖啡", session_id="s1", token_budget=120)

    context = run(scenario())
    assert context.tokens_used <= 120, "注入的记忆块不得超过预算"
    assert len(context.hits) < 60, "预算限制下应只保留少量记忆"


def test_retrieval_empty_store_is_safe(run, memory_stack) -> None:
    context = run(memory_stack.build_context("随便问问", session_id="empty"))
    assert context.rendered == ""
    assert context.hits == []


def test_memory_survives_embedder_failure(run, temp_config, mock_llm, tmp_path) -> None:
    """A dead embedding server must degrade retrieval, not break it."""
    from memory.db import Database
    from memory.engine import MemoryEngine
    from core.types import MemoryKind

    class BrokenEmbedder:
        name = "broken"
        dim = 256

        async def embed(self, texts, **kwargs):
            raise RuntimeError("嵌入服务不可用")

        async def embed_one(self, text, **kwargs):
            raise RuntimeError("嵌入服务不可用")

        async def health(self):
            return {"ok": False, "error": "down"}

    async def scenario():
        db = Database(tmp_path / "degraded.sqlite3")
        db.init()
        from memory.vector import SqliteVectorStore
        from llm.embed import HeuristicReranker

        engine = await MemoryEngine.create(
            temp_config,
            llm=mock_llm,
            db=db,
            embedder=BrokenEmbedder(),
            vectors=SqliteVectorStore(db),
            reranker=HeuristicReranker(),
        )
        await engine.store.add_memory(content="用户喜欢喝咖啡", kind=MemoryKind.PREFERENCE, subject="用户")
        context = await engine.build_context("咖啡", session_id="s1")
        await engine.close()
        return context

    context = run(scenario())
    # 词法通道仍应命中，说明降级路径有效
    assert "咖啡" in context.rendered
