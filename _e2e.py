import asyncio, sys
from types import SimpleNamespace
sys.path.insert(0, '.')
from sqlalchemy import text
from app.db.base import SessionLocal
from app.clients.milvus_client import MilvusIndex

MARK = "【重建自测标记ZQX17】"

async def main():
    idx = MilvusIndex()
    async with SessionLocal() as s:
        row = (await s.execute(text("""
            SELECT c.id, c.doc_id, c.content, d.filename
            FROM psy_chunks c JOIN psy_documents d ON d.id=c.doc_id
            WHERE d.filename LIKE '%论文写作规范%' AND CHAR_LENGTH(c.content) > 200
            LIMIT 1"""))).first()
        chunk_id, doc_id, original, fname = row
        owner = (await s.execute(text("SELECT owner_id FROM psy_knowledge_bases LIMIT 1"))).scalar()
    print(f"目标块 {chunk_id}  文档 {fname[:20]}  owner={owner}")
    print(f"原文前 40 字：{original[:40]}")

    before_vec = await idx.get_vector(chunk_id)
    print(f"改前向量前3位：{[round(x,4) for x in (before_vec or [])[:3]]}")

    # 1) 模拟用户编辑：改正文 + 打标记（等同 PATCH /chunks/{id}）
    async with SessionLocal() as s:
        await s.execute(text(
            "UPDATE psy_chunks SET content = :c, char_count = CHAR_LENGTH(:c), "
            "meta = JSON_SET(COALESCE(meta,'{}'), '$.needs_reindex', true) WHERE id = :i"
        ), {"c": original + "\n" + MARK, "i": chunk_id})
        await s.commit()
    print("已改正文并打 needs_reindex 标记")

    # 2) 调接口（真实代码路径，只绕过 HTTP 鉴权）
    from app.api.chunks import reindex_document
    user = SimpleNamespace(id=int(owner))
    async with SessionLocal() as s:
        result = await reindex_document(doc_id, user, s)
    print("重建返回：", result)

    # 3) 验 Milvus 里的正文与向量
    fetched = await idx.fetch_chunks([chunk_id])
    content_now = fetched[chunk_id]["content"]
    after_vec = await idx.get_vector(chunk_id)
    print(f"Milvus 正文含标记：{MARK in content_now}")
    print(f"改后向量前3位：{[round(x,4) for x in (after_vec or [])[:3]]}")
    print(f"向量是否变化：{before_vec != after_vec}")

    # 4) 恢复原文并重建回去（不留痕）
    async with SessionLocal() as s:
        await s.execute(text(
            "UPDATE psy_chunks SET content = :c, char_count = CHAR_LENGTH(:c), "
            "meta = JSON_SET(COALESCE(meta,'{}'), '$.needs_reindex', true) WHERE id = :i"
        ), {"c": original, "i": chunk_id})
        await s.commit()
    async with SessionLocal() as s:
        await reindex_document(doc_id, SimpleNamespace(id=int(owner)), s)

    fetched2 = await idx.fetch_chunks([chunk_id])
    rest_vec = await idx.get_vector(chunk_id)
    async with SessionLocal() as s:
        meta = (await s.execute(text("SELECT meta, content FROM psy_chunks WHERE id=:i"), {"i": chunk_id})).first()
    print("\n--- 恢复校验 ---")
    print(f"Milvus 正文已还原：{fetched2[chunk_id]['content'] == original[:8000]}")
    print(f"MySQL 正文已还原：{meta[1] == original}")
    print(f"标记已清除：{'needs_reindex' not in (__import__('json').loads(meta[0]) if meta[0] else {})}")
    print(f"恢复后向量与原始一致：{rest_vec == before_vec}")

asyncio.run(main())
