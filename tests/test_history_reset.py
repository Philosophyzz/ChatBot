"""A privacy reset must cover transcripts, derived data, emotion, FTS and backups."""
import importlib.util
from pathlib import Path


def test_reset_clears_primary_and_backup_without_new_copy(temp_config, run):
    from core.app import ChatBotApp
    import sqlite3
    async def scenario():
        temp_config.extra['emotion'] = {'enabled': True, 'model_analysis': False}
        app = await ChatBotApp(temp_config).start(mock=True)
        await app.engine.complete_turn(session_id=None, user_text='我很难过', persona_id='sakura_cat')
        await app.stop()
    run(scenario())
    database = temp_config.paths.db_path
    backup = database.with_name('memory.backup-test.sqlite3')
    with sqlite3.connect(database) as source, sqlite3.connect(backup) as dest:
        source.backup(dest)
    path = Path(__file__).resolve().parents[1] / 'tools/clear_history.py'
    spec = importlib.util.spec_from_file_location('clear_history', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for db in (database, backup):
        result = module.purge_database(db)
        assert result['before']['messages'] == 2
        assert result['before']['emotion_observations'] == 1
        assert not any(result['after'].values())
        with sqlite3.connect(db) as conn:
            assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH '难过'").fetchone()[0] == 0
