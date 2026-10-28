"""清理残留文档：先出清单，再删。

为什么单独写一个：
    分段落库会把失败窗口留下一条 status=failed、0 块的 Document 记录。
    它不占正文，但会让「文档数」和实际书目对不上，也会在列表页出现一堆 0 块的空壳。
    直接 DELETE 是不行的——向量在 Milvus 里，删了 MySQL 的块记录就再也查不到该删哪些向量。

安全约定（LFM G1）：
    默认只打印清单，不删。加 --apply 才真删。
    删除顺序：先删向量（靠 doc_id），再删块，最后删文档。
    doc_id 一条都没有时退出码 1，避免「以为删了其实没匹配」。

用法：
    # 看有多少失败空壳
    .venv/Scripts/python.exe scripts/clean_failed_docs.py --failed-empty

    # 确认后执行
    .venv/Scripts/python.exe scripts/clean_failed_docs.py --failed-empty --apply

    # 删指定文档（例如冒烟测试留下的那条）
    .venv/Scripts/python.exe scripts/clean_failed_docs.py --doc-ids df386d78... --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.clients.milvus_client import MilvusIndex  # noqa: E402
from app.db.base import SessionLocal  # noqa: E402

LIST_FAILED_EMPTY = """
SELECT d.id, d.filename, d.kb_name_if_any, d.status, d.created_at
FROM (
    SELECT d.*, (SELECT k.name FROM psy_knowledge_bases k WHERE k.id = d.kb_id) AS kb_name_if_any
    FROM psy_documents d
    WHERE d.status = 'failed'
      AND NOT EXISTS (SELECT 1 FROM psy_chunks c WHERE c.doc_id = d.id)
) d
ORDER BY d.created_at
"""

LIST_BY_IDS = """
SELECT d.id, d.filename, d.status, d.created_at,
       (SELECT COUNT(*) FROM psy_chunks c WHERE c.doc_id = d.id) AS chunks
FROM psy_documents d
WHERE d.id IN :ids
ORDER BY d.created_at
"""


async def main() -> int:
    ap = argparse.ArgumentParser(description="清理残留文档（默认只出清单）")
    ap.add_argument("--failed-empty", action="store_true", help="所有 status=failed 且 0 块的文档")
    ap.add_argument("--doc-ids", nargs="*", default=[], help="指定 doc_id（支持前缀匹配）")
    ap.add_argument("--apply", action="store_true", help="真的删除")
    args = ap.parse_args()

    if not args.failed_empty and not args.doc_ids:
        print("要么 --failed-empty，要么 --doc-ids，没有目标不动手")
        return 1

    async with SessionLocal() as session:
        targets: list[tuple[str, str, str, str, int]] = []
        if args.failed_empty:
            rows = (await session.execute(text(LIST_FAILED_EMPTY))).all()
            for r in rows:
                targets.append((r[0], r[1], r[3], str(r[4]), 0))
        if args.doc_ids:
            for prefix in args.doc_ids:
                rows = (
                    await session.execute(
                        text(
                            """
                            SELECT d.id, d.filename, d.status, d.created_at,
                                   (SELECT COUNT(*) FROM psy_chunks c WHERE c.doc_id = d.id)
                            FROM psy_documents d
                            WHERE d.id LIKE :pat
                            """
                        ),
                        {"pat": f"{prefix}%"},
                    )
                ).all()
                for r in rows:
                    targets.append((r[0], r[1], r[2], str(r[3]), int(r[4])))

        # 去重（两条路径可能命中同一条）
        seen: set[str] = set()
        uniq = []
        for t in targets:
            if t[0] in seen:
                continue
            seen.add(t[0])
            uniq.append(t)

        if not uniq:
            print("没有匹配的文档")
            return 1

        print(f"命中 {len(uniq)} 条：")
        for doc_id, name, status, created, chunks in uniq:
            print(f"  {doc_id[:8]}  {name[:52]:54} {status:8} 块 {chunks:5d}  {created}")

        if not args.apply:
            print("\n[dry-run] 只是清单。确认无误后加 --apply")
            return 0

    index = MilvusIndex()
    for doc_id, name, *_ in uniq:
        await index.delete_by_doc(doc_id)
        async with SessionLocal() as session:
            await session.execute(text("DELETE FROM psy_chunks WHERE doc_id = :d"), {"d": doc_id})
            await session.execute(text("DELETE FROM psy_documents WHERE id = :d"), {"d": doc_id})
            await session.commit()
        print(f"  已删除 {doc_id[:8]} {name[:40]}")

    print(f"\n完成：{len(uniq)} 条文档及其向量已清理")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
