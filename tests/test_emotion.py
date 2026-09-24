"""Emotion contracts: evidence, uncertainty, separation and deletion behavior."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from core.emotion import EmotionService, explicit_emotion, parse_assessment
from core.types import Message, Role
from memory.db import Database
from memory.store import MemoryStore


@pytest.mark.parametrize("text,label", [("我很难过", "sad"), ("我现在不难过了", "unknown"),
    ("我今天好累", "tired"), ("她今天很难过", None), ('他说“我很难过”', None),
    ("我昨天很开心", None), ("我不是很开心", None), ("我想知道难过是什么意思", None)])
def test_explicit_statements_only(text, label):
    assert (explicit_emotion(text) or {}).get("label") == label


@pytest.mark.parametrize("score", [True, -0.1, 1.1, float('nan'), "0.8"])
def test_rejects_invalid_model_scores(score):
    assert parse_assessment(json.dumps({"user": {"label": "sad", "confidence": score}, "pet": None})) is None


def test_null_is_unknown_not_zero():
    parsed = parse_assessment('{"user":{"label":"sad","intensity":null,"confidence":null},"pet":null}')
    assert parsed["user"]["intensity"] is None
    assert parse_assessment('broken response') is None


def test_state_and_snapshots_survive_restart_and_honor_deletion(temp_config, run):
    async def scenario():
        temp_config.extra['emotion'] = {'enabled': True, 'model_analysis': False}
        db = Database(temp_config.paths.db_path); db.init()
        store = MemoryStore(db)
        await store.ensure_conversation('test', persona_id='sakura_cat')
        service = EmotionService(temp_config, db)
        first = Message(Role.USER, '我很难过')
        await store.add_message('test', first, persona_id='sakura_cat')
        service.capture(first, 'sakura_cat', 'test')
        state = await service.assess(first, 'sakura_cat', 'test', [], SimpleNamespace(name='mock'))
        assert state['user']['label'] == 'sad'
        assert state['pet']['label'] == 'caring'
        assert state['pet']['source'] == 'companion_rule'
        assert service.view('mint_bunny')['user'] is None
        reply = Message(Role.ASSISTANT, '我陪着你。')
        await store.add_message('test', reply, persona_id='sakura_cat')
        service.capture(reply, 'sakura_cat', 'test')
        second = Message(Role.USER, '我现在不难过了')
        await store.add_message('test', second, persona_id='sakura_cat')
        await service.assess(second, 'sakura_cat', 'test', [], SimpleNamespace(name='mock'))
        service = EmotionService(temp_config, db)
        assert service.view('sakura_cat')['user']['label'] == 'unknown'
        snapshots = service.view('sakura_cat')['snapshots']
        assert next(s for s in snapshots if s['message_id'] == reply.id)['user']['label'] == 'sad'
        await store.delete_conversation('test')
        assert service.view('sakura_cat')['user'] is None
        assert service.view('sakura_cat')['snapshots'] == []
        db.close()
    run(scenario())


def test_model_context_timeout_and_deleted_source(temp_config, run):
    async def scenario():
        temp_config.extra['emotion'] = {'enabled': True, 'timeout_s': .1}
        db = Database(temp_config.paths.db_path); db.init()
        store = MemoryStore(db)
        await store.ensure_conversation('test', persona_id='sakura_cat')
        service = EmotionService(temp_config, db)
        current = Message(Role.USER, '又被否决了，好像怎么做都不对')
        await store.add_message('test', current, persona_id='sakura_cat')
        calls = []
        class LLM:
            name = 'test'
            async def complete(self, request):
                calls.append(request)
                return SimpleNamespace(text='{"user":{"label":"sad","confidence":0.8},"pet":{"label":"caring"}}')
        state = await service.assess(current, 'sakura_cat', 'test', [], LLM())
        assert state['user']['label'] == 'sad' and calls
        assert current.content in calls[0].messages[1].content
        # A corrected source invalidates the derived state.
        db.execute('UPDATE messages SET content=? WHERE id=?', ('这是电影台词', current.id))
        assert service.view('sakura_cat')['user'] is None
        next_message = Message(Role.USER, '今天没有什么想说的')
        await store.add_message('test', next_message, persona_id='sakura_cat')
        class Slow:
            name = 'test'
            async def complete(self, request):
                await asyncio.sleep(10)
        assert (await service.assess(next_message, 'sakura_cat', 'test', [], Slow()))['user'] is None
        late_message = Message(Role.USER, '这是什么情况')
        await store.add_message('test', late_message, persona_id='sakura_cat')
        class Deleted(LLM):
            async def complete(self, request):
                await store.delete_conversation('test')
                return await super().complete(request)
        assert (await service.assess(late_message, 'sakura_cat', 'test', [], Deleted()))['user'] is None
        assert db.scalar('SELECT COUNT(*) FROM emotion_observations') == 0
        db.close()
    run(scenario())


def test_chat_emits_emotion_and_voice_adapts_without_changing_identity(temp_config, run):
    from core.app import ChatBotApp
    async def scenario():
        temp_config.extra['emotion'] = {'enabled': True, 'model_analysis': False}
        app = await ChatBotApp(temp_config).start(mock=True)
        try:
            result = await app.engine.complete_turn(session_id=None, user_text='我很难过', persona_id='sakura_cat')
            event = next(e for e in result.events if e['kind'] == 'emotion')
            assert event['emotion']['user']['label'] == 'sad'
            persona = app.personas.get('sakura_cat')
            voice = app.engine.emotions.voice(persona)
            assert voice.voice_id == persona.voice.voice_id
            assert voice.speed < persona.voice.speed
            assert '用户可能难过' in app.engine.emotions.prompt('sakura_cat')
            assert len(app.engine.emotions.view('sakura_cat')['snapshots']) == 2
        finally:
            await app.stop()
    run(scenario())


def test_emotion_and_transcript_work_when_memory_engine_unavailable(temp_config, run, monkeypatch):
    from core.app import ChatBotApp
    from core.session import SessionManager
    async def unavailable(*args, **kwargs):
        raise RuntimeError('memory component unavailable')
    monkeypatch.setattr('core.app.MemoryEngine.create', unavailable)
    async def scenario():
        temp_config.extra['emotion'] = {'enabled': True, 'model_analysis': False}
        app = await ChatBotApp(temp_config).start(mock=True)
        store = app.sessions.store
        try:
            assert app.memory is None and store is not None
            result = await app.engine.complete_turn(session_id=None, user_text='我很难过', persona_id='sakura_cat')
            session_id = next(e['session']['id'] for e in result.events if e['kind'] == 'session')
            restored = await SessionManager(store).get(session_id)
            assert [m.role for m in restored.messages] == [Role.USER, Role.ASSISTANT]
            assert len(app.engine.emotions.view('sakura_cat')['snapshots']) == 2
        finally:
            await app.stop()
            store.db.close()
    run(scenario())
