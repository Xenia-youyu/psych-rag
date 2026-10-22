"""给截图脚本签一个真实 token。

不是测试工具：前端验收需要"已登录且有数据"的视角，而这个视角要求
借知识库所有者的身份。顺手把第一个有文档的 kb_id 带出来。
"""

from __future__ import annotations

import asyncio
import json

from sqlalchemy import select

from app.core.security import create_access_token
from app.db.base import SessionLocal
from app.db.models import Document, KnowledgeBase, User


async def main() -> None:
    async with SessionLocal() as session:
        kb_id = await session.scalar(
            select(Document.kb_id).order_by(Document.created_at.desc()).limit(1)
        )
        if kb_id is None:
            raise SystemExit("库里没有文档记录")
        kb = await session.get(KnowledgeBase, kb_id)
        owner = await session.get(User, kb.owner_id)
        if kb is None or owner is None:
            raise SystemExit("知识库或所有者不存在")

        token, expires_in = create_access_token(owner.id, owner.username)
        doc = await session.scalar(
            select(Document).where(Document.kb_id == kb_id).order_by(Document.created_at.desc())
        )
        print(
            json.dumps(
                {
                    "token": token,
                    "expires_in": expires_in,
                    "user": {"id": owner.id, "username": owner.username},
                    "kb_id": kb.id,
                    "kb_name": kb.name,
                    "doc_id": doc.id if doc else "",
                    "doc_name": doc.filename if doc else "",
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
