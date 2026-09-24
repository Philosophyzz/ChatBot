"""Clear personal ChatBot history, including old database backups, while services are stopped.

Run with --apply after stopping the desktop and API. No new personal-data backup is made.
Model files, voices, character introductions and application settings are preserved.
"""
from __future__ import annotations

import argparse
import json
import socket
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ("emotion_snapshots", "emotion_observations", "messages_fts", "memory_fts",
          "memory_vectors", "entity_vectors", "profiles", "relations", "entity_aliases",
          "entities", "memory_items", "turn_log", "messages", "conversations", "kv_store")


def purge_database(path: Path):
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.execute("PRAGMA secure_delete=ON")
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        tables = [t for t in TABLES if t in names]
        before = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
        with con:
            for table in tables:
                con.execute(f'DELETE FROM "{table}"')
        con.execute("VACUUM")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        after = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
        if any(after.values()):
            raise RuntimeError(f"Database was not fully cleared: {path}")
        return {"path": str(path), "before": before, "after": after}
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    from core.config import load_config
    config = load_config(ROOT)
    with socket.socket() as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(('127.0.0.1', config.server.port)) == 0:
            raise SystemExit('Stop the ChatBot API and pet before clearing history.')
    data = config.paths.data_dir.resolve()
    databases = sorted(data.glob('memory*.sqlite3'))
    if config.paths.sessions_db_path.exists() and config.paths.sessions_db_path not in databases:
        databases.append(config.paths.sessions_db_path)
    for path in databases:
        if path.resolve().parent != data or path.is_symlink():
            raise SystemExit(f'Unexpected database path: {path}')
    if not args.apply:
        print(json.dumps([str(p) for p in databases], indent=2))
        return 0
    report = [purge_database(path) for path in databases]
    # These runtime logs can contain copied messages or extraction diagnostics.
    for directory in (config.paths.logs_dir, ROOT / 'dist' / 'logs'):
        for pattern in ('chatbot.jsonl*', 'api*.log', 'llama-chat*.log',
                        'desktop-start.log', 'pet.log*', 'pet-crash.log*'):
            for log in directory.glob(pattern):
                if log.is_file() and log.resolve().parent == directory.resolve():
                    log.write_bytes(b'')
    config.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.logs_dir / 'history-reset.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Cleared {len(report)} databases, including backups. All personal-data table counts are zero.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
