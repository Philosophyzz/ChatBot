"""One-off repair of profile rows written before key canonicalisation existed.

Background: the extractor used to store whatever key the language model produced, so one
attribute accumulated several competing rows. Measured in a real database::

    名字=小明        (confidence 0.95)
    姓名=张伟        (confidence 0.90)
    name=小明        (confidence 0.90)

All three describe the user's name. The UI then shows a profile that contradicts itself,
which is exactly what makes a memory system feel unreliable.

This script merges such rows onto their canonical key, keeping the highest-confidence
value and resolving ties by recency (the most recently stated value wins). It also
repairs entity fragmentation: referring expressions for the user ("我", "用户", the
user's own given name) are collapsed onto a single graph node, with the other names kept
as aliases so nothing is lost.

Safe by default: ``--dry-run`` (the default) only prints what it would do. Pass
``--apply`` to write. A backup copy of the database is taken before any write.

Usage::

    python tests/repair_memory.py                 # show planned changes
    python tests/repair_memory.py --apply         # write them
    python tests/repair_memory.py --apply --no-entity-merge
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sqlite3
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory.extract import is_self_reference, normalize_profile_key  # noqa: E402

DB = ROOT / "data" / "memory.sqlite3"


def plan_profile_merge(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Group profile rows by canonical key and pick a winner per group."""
    rows = conn.execute(
        "SELECT id, key, value, confidence, importance, updated_at, scope, subject FROM profiles"
    ).fetchall()
    groups: Dict[Tuple[str, str, str], List[sqlite3.Row]] = {}
    for row in rows:
        canonical = normalize_profile_key(row["key"])
        if not canonical:
            continue
        groups.setdefault((row["scope"], row["subject"], canonical), []).append(row)

    plan: List[Dict[str, Any]] = []
    for (scope, subject, canonical), members in groups.items():
        # Highest confidence wins; ties break on the most recent statement.
        ranked = sorted(members, key=lambda r: (float(r["confidence"] or 0), float(r["updated_at"] or 0)), reverse=True)
        winner = ranked[0]
        losers = ranked[1:]
        needs_rename = winner["key"] != canonical
        if not losers and not needs_rename:
            continue
        plan.append(
            {
                "scope": scope,
                "subject": subject,
                "canonical": canonical,
                "winner": winner,
                "losers": losers,
                "rename": needs_rename,
            }
        )
    return plan


def apply_profile_merge(conn: sqlite3.Connection, plan: List[Dict[str, Any]]) -> int:
    changed = 0
    for entry in plan:
        winner = entry["winner"]
        # Move the winner onto the canonical key. A row with that key may already exist
        # (that is one of the losers), so delete it first to avoid a unique clash.
        for loser in entry["losers"]:
            conn.execute("DELETE FROM profiles WHERE id=?", (loser["id"],))
            changed += 1
        if entry["rename"]:
            conn.execute(
                "UPDATE profiles SET key=?, updated_at=? WHERE id=?",
                (entry["canonical"], time.time(), winner["id"]),
            )
            changed += 1
    return changed


def plan_entity_merge(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Find every entity node that denotes the same person as some other node.

    Naive grouping by name or alias identity is not enough, and the real data proves it:
    ``小明`` and ``用户`` had each registered the *other* as an alias, so both resolved
    for every mention and accumulated ~36 mentions apiece, while a third node ``张伟``
    also carried the alias ``小明``. Three nodes, one person, relationships split across
    them (including the nonsense edge ``张伟 --称呼--> 小明``, i.e. a person being their
    own nickname).

    The fix is a union-find over the name/alias graph: any two nodes that share a name
    are the same identity, transitively. A cluster containing a self-reference
    ("我"/"用户") is the user's own node, and that node is renamed to the canonical
    ``用户`` so future extraction converges on it.
    """
    rows = conn.execute(
        "SELECT id, name, type, importance, mention_count, scope FROM entities"
    ).fetchall()

    plan: List[Dict[str, Any]] = []
    for scope in sorted({row["scope"] for row in rows}):
        members = [row for row in rows if row["scope"] == scope]
        if not members:
            continue

        # ---- union-find over "same name or shares an alias" --------------------------
        parent: Dict[str, str] = {row["id"]: row["id"] for row in members}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            root_l, root_r = find(left), find(right)
            if root_l != root_r:
                parent[root_r] = root_l

        # name (lowercased) -> first node that claims it
        name_owner: Dict[str, str] = {}
        for row in members:
            for token in [row["name"]] + [
                r["alias"]
                for r in conn.execute(
                    "SELECT alias FROM entity_aliases WHERE entity_id=?", (row["id"],)
                ).fetchall()
            ]:
                key = (token or "").strip().lower()
                if not key:
                    continue
                if key in name_owner:
                    union(name_owner[key], row["id"])
                else:
                    name_owner[key] = row["id"]

        clusters: Dict[str, List[sqlite3.Row]] = {}
        for row in members:
            clusters.setdefault(find(row["id"]), []).append(row)

        for cluster in clusters.values():
            if len(cluster) < 2:
                continue
            self_nodes = [row for row in cluster if is_self_reference(row["name"])]
            # Rank: a self-reference wins the canonical name; then mentions; then importance.
            ranked = sorted(
                cluster,
                key=lambda r: (
                    1 if is_self_reference(r["name"]) else 0,
                    int(r["mention_count"] or 0),
                    float(r["importance"] or 0),
                ),
                reverse=True,
            )
            plan.append(
                {
                    "scope": scope,
                    "keeper": ranked[0],
                    "merged": ranked[1:],
                    "rename_to": "用户" if self_nodes or len(cluster) > 1 else None,
                }
            )
    return plan


def apply_entity_merge(conn: sqlite3.Connection, plan: List[Dict[str, Any]]) -> int:
    """Fold duplicate nodes into the keeper, repointing relations and keeping aliases."""
    changed = 0
    for entry in plan:
        keeper = entry["keeper"]
        # Collect every name in the cluster so all of them remain resolvable afterwards.
        cluster_names: List[str] = []
        for node in [keeper, *entry["merged"]]:
            cluster_names.append(node["name"])
            cluster_names.extend(
                row["alias"]
                for row in conn.execute(
                    "SELECT alias FROM entity_aliases WHERE entity_id=?", (node["id"],)
                ).fetchall()
            )

        for node in entry["merged"]:
            # Repoint relations, then drop the self-loops and duplicate triples the merge
            # can create (e.g. 张伟 --称呼--> 小明 becomes a node pointing at itself).
            for column in ("subject_id", "object_id"):
                conn.execute(
                    f"UPDATE OR IGNORE relations SET {column}=? WHERE {column}=?",
                    (keeper["id"], node["id"]),
                )
            conn.execute("DELETE FROM relations WHERE subject_id=object_id")
            conn.execute(
                "DELETE FROM relations WHERE id NOT IN "
                "(SELECT MIN(id) FROM relations GROUP BY scope, subject_id, predicate, object_id)"
            )
            # Re-home memory subjects onto the canonical name.
            conn.execute(
                "UPDATE memory_items SET subject=? WHERE subject=?",
                (entry.get("rename_to") or keeper["name"], node["name"]),
            )
            conn.execute("DELETE FROM entities WHERE id=?", (node["id"],))
            conn.execute(
                "UPDATE entities SET mention_count=mention_count+?, importance=MAX(importance,?) WHERE id=?",
                (int(node["mention_count"] or 0), float(node["importance"] or 0), keeper["id"]),
            )
            changed += 1

        # Rename the keeper to the canonical self name and register every alias.
        canonical = entry.get("rename_to") or keeper["name"]
        conn.execute("UPDATE entities SET name=? WHERE id=?", (canonical, keeper["id"]))
        for name in cluster_names + [canonical]:
            if name and name.strip():
                conn.execute(
                    "INSERT INTO entity_aliases(entity_id, alias, weight) VALUES(?,?,1.0) "
                    "ON CONFLICT(entity_id, alias) DO NOTHING",
                    (keeper["id"], name.strip()),
                )
        conn.execute(
            "UPDATE memory_items SET subject=? WHERE subject=?",
            (canonical, keeper["name"]),
        )
    return changed


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="修复历史档案/实体记录的质量问题")
    parser.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
    parser.add_argument("--db", default=str(DB))
    parser.add_argument("--no-entity-merge", action="store_true", help="跳过实体归并")
    args = parser.parse_args(argv)

    db_path = pathlib.Path(args.db)
    if not db_path.exists():
        print(f"找不到数据库：{db_path}")
        return 2

    print(f"数据库: {db_path}")
    print(f"模式: {'写入' if args.apply else '预览（加 --apply 才写入）'}")
    print("")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        before = {
            "profiles": conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0],
            "entities": conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
            "relations": conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
            "memories": conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0],
        }
        print(f"修复前: 档案 {before['profiles']} 条 / 实体 {before['entities']} 个 / 关系 {before['relations']} 条 / 记忆 {before['memories']} 条")
        print("")

        p_plan = plan_profile_merge(conn)
        print(f"=== 档案键归并：{len(p_plan)} 组 ===")
        for entry in p_plan:
            winner = entry["winner"]
            print(f"  [{entry['scope']}] {entry['canonical']}  <-  保留 {winner['value']!r} (置信度 {winner['confidence']}, 原键 {winner['key']!r})")
            for loser in entry["losers"]:
                print(f"        删除重复: {loser['key']!r} = {loser['value']!r} (置信度 {loser['confidence']})")
        print("")

        e_plan: List[Dict[str, Any]] = []
        if not args.no_entity_merge:
            e_plan = plan_entity_merge(conn)
            print(f"=== 实体归并：{len(e_plan)} 组 ===")
            for entry in e_plan:
                keeper = entry["keeper"]
                others = ", ".join(f"{r['name']}(提及{r['mention_count']})" for r in entry["merged"])
                print(f"  [{entry['scope']}] 保留 {keeper['name']}(提及{keeper['mention_count']})  <-  合并 {others}")
            print("")

        if not args.apply:
            print("预览结束。确认无误后加 --apply 执行。")
            return 0

        backup = db_path.with_suffix(f".backup-{time.strftime('%Y%m%d-%H%M%S')}.sqlite3")
        shutil.copy2(db_path, backup)
        print(f"已备份到 {backup.name}")
        print("")

        conn.execute("BEGIN IMMEDIATE")
        try:
            n_profile = apply_profile_merge(conn, p_plan)
            n_entity = apply_entity_merge(conn, e_plan) if e_plan else 0
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

        after = {
            "profiles": conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0],
            "entities": conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
            "relations": conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
            "memories": conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0],
        }
        print(f"改动: 档案 {n_profile} 处，实体归并 {n_entity} 个")
        print(f"修复后: 档案 {after['profiles']} 条 / 实体 {after['entities']} 个 / 关系 {after['relations']} 条 / 记忆 {after['memories']} 条")
        print("")
        print("=== 修复后的档案 ===")
        for row in conn.execute("SELECT key, value, confidence FROM profiles ORDER BY importance DESC, updated_at DESC").fetchall():
            print(f"  {row['key']}: {row['value']}  (置信度 {row['confidence']})")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
