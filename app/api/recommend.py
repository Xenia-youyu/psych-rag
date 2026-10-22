"""每日知识推荐。

三个设计决定：

1. **当天固定，不是每次随机。**
   种子 = 日期 + 用户 + 知识库。刷新页面不该换一批——用户看到一条想细看，
   回头找不到是最败好感的行为。想看别的给 `variation` 参数。

2. **推荐用子块，详情给父块。**
   卡片要短（子块 300 字），但「详细介绍」必须是父块全文（约 1200 字）。
   子块常是半句假设半句结论，拿它当详情等于让用户看一段没头没尾的话。

3. **关联内容用向量召回，不调大模型。**
   关联靠库里已有的向量（`get_vector` 直接取，不重新编码），
   详情接口 0.5 秒内返回。要「讲解这段」就走问答链路，不在推荐里另造一套生成。
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.chat import get_runtime
from app.api.kb import get_owned_kb
from app.core.deps import CurrentUser, SessionDep
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.db.models import Chunk, Document
from app.schemas.recommend import (
    RecommendDetail,
    RecommendItem,
    RecommendList,
    RelatedItem,
)

router = APIRouter(prefix="/api/recommend", tags=["每日推荐"])
logger = get_logger(__name__)

MIN_CHARS = 80
# 最多取这么多 id 参与排序。再大的库也不需要候选全量——日常推荐只看前几十个
MAX_CANDIDATES = 600
# 排序后只看前这么多条的内容。质量过滤要读正文，逐条查全表太慢
CHECK_WINDOW = 40

TOC_PATTERN = re.compile(r"\.{3,}\s*\d+")
HEADING_PREFIX = re.compile(r"^[0-9０-９一二三四五六七八九十（()）、\.\s]+")
# 像小标题/条目的开头。切块是按长度硬切的，块首经常落在句子中间
HEADING_START = re.compile(r"^([0-9０-９]+\s*[、\.．]|[一二三四五六七八九十]+、|[（(][一二三四五六七八九十0-9]+[)）]|第[一二三四五六七八九十0-9]+[章节条])")


def _seed(user_id: int, kb_id: str, variation: int) -> str:
    return f"{date.today().isoformat()}|{user_id}|{kb_id}|{variation}"


def _order_key(seed: str, chunk_id: str) -> str:
    return hashlib.blake2b(f"{seed}:{chunk_id}".encode(), digest_size=8).hexdigest()


def _usable(content: str) -> bool:
    text = (content or "").strip()
    if len(text) < MIN_CHARS:
        return False
    # 目录页特征：大量「.....12」的引导点 + 页码。那不是知识点
    if len(TOC_PATTERN.findall(text)) >= 3:
        return False
    # 一个句号都没有，多半是标题串或表格残片
    if "。" not in text and "？" not in text:
        return False
    return True


def _title(content: str) -> str:
    """从正文里取一个像标题的东西。

    不能只看第一行：切块是按长度硬切的，第一行经常只剩个页码「1」，
    去掉编号就空了。所以要往后看几行，取第一个有实义的行。

    如果块首明显落在句子中间（没有编号，且一上来就是长句），
    前面加省略号——让用户一眼看出这是节选而不是完整段落。
    """
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    if not lines:
        return ""

    continued = len(lines[0]) > 24 and not HEADING_START.match(lines[0])
    prefix = "…" if continued else ""

    for line in lines[:3]:
        cleaned = HEADING_PREFIX.sub("", line).strip()
        if len(cleaned) >= 4:
            return prefix + cleaned[:36] + ("…" if len(cleaned) > 36 else "")

    flat = " ".join(lines)
    return prefix + flat[:36] + ("…" if len(flat) > 36 else "")


def _preview(content: str, limit: int = 96) -> str:
    text = " ".join((content or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


async def _doc_titles(session, doc_ids: list[str]) -> dict[str, str]:
    ids = {doc_id for doc_id in doc_ids if doc_id}
    if not ids:
        return {}
    rows = (await session.scalars(select(Document).where(Document.id.in_(ids)))).all()
    return {row.id: Path(row.filename).stem for row in rows}


@router.get("/daily", response_model=RecommendList, summary="今日知识推荐")
async def daily(
    kb_id: str,
    user: CurrentUser,
    session: SessionDep,
    limit: int = Query(default=3, ge=1, le=6),
    variation: int = Query(default=0, ge=0, le=99),
) -> RecommendList:
    await get_owned_kb(kb_id, user.id, session)

    ids = (
        await session.scalars(
            select(Chunk.id)
            .where(
                Chunk.kb_id == kb_id,
                Chunk.enabled.is_(True),
                Chunk.char_count >= MIN_CHARS,
            )
            .limit(MAX_CANDIDATES)
        )
    ).all()

    if not ids:
        return RecommendList(
            date=date.today().isoformat(), variation=variation, total_candidates=0
        )

    seed = _seed(user.id, kb_id, variation)
    ranked = sorted(ids, key=lambda chunk_id: _order_key(seed, chunk_id))

    window = ranked[:CHECK_WINDOW]
    rows = (await session.scalars(select(Chunk).where(Chunk.id.in_(window)))).all()
    by_id = {row.id: row for row in rows}
    ordered = [by_id[chunk_id] for chunk_id in window if chunk_id in by_id]
    picked = [row for row in ordered if _usable(row.content)][:limit]

    doc_map = await _doc_titles(session, [row.doc_id for row in picked])
    items = [
        RecommendItem(
            chunk_id=row.id,
            title=_title(row.content),
            preview=_preview(row.content),
            doc_id=row.doc_id,
            doc_title=doc_map.get(row.doc_id, ""),
            page=row.page_start,
            char_count=row.char_count,
        )
        for row in picked
    ]

    logger.info(
        "recommend.daily",
        extra={
            "extra_fields": {
                "kb_id": kb_id,
                "candidates": len(ids),
                "picked": len(items),
                "variation": variation,
            }
        },
    )
    return RecommendList(
        date=date.today().isoformat(),
        variation=variation,
        total_candidates=len(ids),
        items=items,
    )


async def _related(session, index, chunk: Chunk, limit: int = 3) -> list[RelatedItem]:
    """找内容相近的其他块。

    用库里存的向量做单路稠密检索：比重新编码快 0.4 秒，
    而且关联推荐只需要"意思靠近"，不需要混合检索那套精度。
    """
    vector = await index.get_vector(chunk.id)
    if vector is None:
        return []

    try:
        hits = await index.search(kb_ids=[chunk.kb_id], dense=vector, topk=14)
    except Exception as exc:  # noqa: BLE001 - 关联失败不该让详情页打不开
        logger.warning(
            "recommend.related_failed",
            extra={"extra_fields": {"chunk_id": chunk.id, "error": repr(exc)}},
        )
        return []

    candidates = [
        hit
        for hit in hits
        if hit.get("chunk_id") != chunk.id and hit.get("parent_id") != chunk.parent_id
    ]

    # 优先放一条来自别的书的关联。全是同一本书的相邻段落，等于把"关联"
    # 做成了"上文下接"，用户看不到这个知识点在别的教材里怎么讲。
    cross = [hit for hit in candidates if hit.get("doc_id") != chunk.doc_id]
    same = [hit for hit in candidates if hit.get("doc_id") == chunk.doc_id]
    picked = (cross[:1] + same[: limit - 1]) if cross else same[:limit]
    if len(picked) < limit:
        chosen = {hit["chunk_id"] for hit in picked}
        picked += [hit for hit in candidates if hit["chunk_id"] not in chosen][
            : limit - len(picked)
        ]

    doc_map = await _doc_titles(session, [str(hit.get("doc_id", "")) for hit in picked])
    return [
        RelatedItem(
            chunk_id=str(hit["chunk_id"]),
            doc_id=str(hit.get("doc_id", "")),
            doc_title=doc_map.get(str(hit.get("doc_id", "")), ""),
            page=int(hit.get("page_start") or 0),
            preview=_preview(str(hit.get("content", ""))),
        )
        for hit in picked
    ]


@router.get(
    "/chunks/{chunk_id}", response_model=RecommendDetail, summary="知识点详情与关联内容"
)
async def detail(chunk_id: str, user: CurrentUser, session: SessionDep) -> RecommendDetail:
    chunk = await session.get(Chunk, chunk_id)
    if chunk is None:
        raise AppError(ErrorCode.NOT_FOUND, "这条内容已不存在，可能文档被删除了")
    # 越权一律 404：403 等于告诉对方"这个 ID 存在"
    await get_owned_kb(chunk.kb_id, user.id, session)

    doc_map = await _doc_titles(session, [chunk.doc_id])
    runtime = get_runtime()

    content = chunk.content
    is_full_parent = False
    if chunk.parent_id:
        parents = await runtime.index.fetch_chunks([chunk.parent_id])
        parent_content = parents.get(chunk.parent_id, {}).get("content", "")
        if parent_content:
            content = parent_content
            is_full_parent = True

    related = await _related(session, runtime.index, chunk)

    return RecommendDetail(
        chunk_id=chunk.id,
        title=_title(chunk.content),
        preview=_preview(chunk.content),
        doc_id=chunk.doc_id,
        doc_title=doc_map.get(chunk.doc_id, ""),
        page=chunk.page_start,
        char_count=len(content),
        content=content,
        is_full_parent=is_full_parent,
        related=related,
    )
