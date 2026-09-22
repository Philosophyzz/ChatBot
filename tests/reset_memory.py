"""Reset the local memory store (maintenance tool, not a test).

Everything in ``data/memory.sqlite3`` is *derived*: memories, profile slots,
entities/relations and the transcripts themselves are all produced from
conversations. While developing, that store fills up with synthetic turns — and the
waste is not only cosmetic. A leftover test fact ("用户养了一只叫团子的猫") is
retrieved on the next real question, and the model dutifully tells a stranger about
their own cat. Cleaning the store is how you get back to a truthful assistant.

Usage (stop the web service first, so nothing re-writes the DB while it is emptied):

    python tests/reset_memory.py                     # report only, changes nothing
    python tests/reset_memory.py --apply             # backup + wipe memories/profile/graph
    python tests/reset_memory.py --apply --sessions  # also wipe conversations + messages

``--apply`` always writes a timestamped copy next to the database before touching it,
so a wrong call is recoverable by copying the backup back over ``memory.sqlite3``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Tables holding things the assistant "remembers". Order matters only for reporting.
MEMORY_TABLES = (
    "memory_items",
    "memory_fts",
    "memory_vectors",
    "profiles",
    "entities",
    "entity_aliases",
    "entity_vectors",
    "relations",
)

#: Tables holding the transcripts those memories were derived from.
SESSION_TABLES = (
    "turn_log",
    "messages",
    "messages_fts",
    "conversations",
)


def default_db_path() -> Path:
    return PROJECT_ROOT / "data" / "memory.sqlite3"


def table_counts(db: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    con = sqlite3.connect(str(db))
    try:
        names = {row[0] for row in con.execute("select name from sqlite_master where type='table'")}
        for table in MEMORY_TABLES + SESSION_TABLES:
            if table in names:
                counts[table] = con.execute(f'select count(*) from "{table}"').fetchone()[0]
    finally:
        con.close()
    return counts


def api_is_listening(port: int = 8077) -> bool:
    """True when the web service is up, i.e. it holds the DB open and caches state."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def backup(db: Path) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = db.with_name(f"memory.backup-{stamp}.sqlite3")
    # Fold the WAL back into the main file first: copying only the .sqlite3 while a
    # multi-megabyte -wal sits beside it would produce a backup missing recent writes.
    con = sqlite3.connect(str(db))
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
    finally:
        con.close()
    shutil.copy2(db, target)
    prune_backups(db, keep=5)
    return target


def prune_backups(db: Path, keep: int = 5) -> List[Path]:
    """Keep the newest ``keep`` backups next to the database.

    A reset is a rare, deliberate act; without a limit every one of them leaves a
    multi-hundred-KB copy behind forever.
    """
    backups = sorted(
        db.parent.glob("memory.backup-*.sqlite3"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    removed: List[Path] = []
    for stale in backups[keep:]:
        try:
            stale.unlink()
            removed.append(stale)
        except OSError:
            pass
    return removed


def purge(db: Path, tables: Tuple[str, ...]) -> List[Tuple[str, int]]:
    removed: List[Tuple[str, int]] = []
    con = sqlite3.connect(str(db))
    try:
        names = {row[0] for row in con.execute("select name from sqlite_master where type='table'")}
        for table in tables:
            if table not in names:
                continue
            before = con.execute(f'select count(*) from "{table}"').fetchone()[0]
            if before:
                con.execute(f'delete from "{table}"')
            removed.append((table, before))
        con.commit()
        con.execute("VACUUM")
    finally:
        con.close()
    return removed


def _report(counts: Dict[str, int], title: str) -> None:
    print(title)
    for table in MEMORY_TABLES + SESSION_TABLES:
        if table in counts:
            print(f"  {table:18} {counts[table]}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清空本地记忆库（会先自动备份）")
    parser.add_argument("--db", default=str(default_db_path()), help="memory.sqlite3 路径")
    parser.add_argument("--apply", action="store_true", help="真正执行删除；缺省只统计")
    parser.add_argument("--sessions", action="store_true", help="连同对话记录一起清空")
    parser.add_argument("--force", action="store_true", help="网页服务在运行时也强制继续")
    args = parser.parse_args(argv)

    db = Path(args.db)
    if not db.exists():
        print(f"找不到数据库：{db}")
        return 2

    if api_is_listening() and not args.force:
        print("网页服务（8077）正在运行：请先执行 scripts\\stop.ps1 再清空，")
        print("否则进程里缓存的档案与记忆计数不会刷新。确实要继续就加 --force。")
        return 1

    tables = MEMORY_TABLES + (SESSION_TABLES if args.sessions else ())
    _report(table_counts(db), f"当前状态（{db}）")

    if not args.apply:
        print("\n这是预演，没有改动任何数据。加 --apply 才会真正清空。")
        return 0

    saved = backup(db)
    print(f"\n已备份到 {saved}")
    pruned = prune_backups(db, keep=5)
    if pruned:
        print(f"（已清理 {len(pruned)} 个更早的备份，只保留最近 5 个）")
    removed = purge(db, tables)
    print("已清空：")
    for table, count in removed:
        print(f"  {table:18} -{count}")
    _report(table_counts(db), "\n清空后：")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
