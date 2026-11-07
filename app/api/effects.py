"""每日知识：心理学效应卡片。

三个设计决定：

1. **当天固定，不是每次打开都换。**
   种子 = 日期 + variation。用户上午看到「光环效应」想细看，下午打开还是它；
   想换一批就给 `variation=1`。纯随机会让「每日知识」变成「每次刷新都换知识」，
   连自己昨天看过什么都记不住。

2. **效应不复用教材语料。**
   从 psy_chunks 里随机抽块做过一版，效果很差：教材块是按长度硬切的论述文，
   抽出来常常没头没尾（"……这一点在实验中得到了验证。"），既不是完整知识点，
   也没有可读性。效应条目是结构化的独立短文，才撑得住「每日一读」。

3. **内容进库，但不要求登录。**
   效应是公共知识，跟用户语料无关。接口仍然要求登录（整个应用都要），
   但查询不碰 kb_id，因此新用户第一天就能看到内容，不必先建库、先传书。
"""

from __future__ import annotations

import hashlib
from datetime import date

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.core.deps import CurrentUser, SessionDep
from app.core.errors import AppError, ErrorCode
from app.db.models import Effect
from app.schemas.effect import EffectBrief, EffectDaily, EffectDetail, EffectList

router = APIRouter(prefix="/api/effects", tags=["每日知识"])


def _brief(row: Effect) -> EffectBrief:
    return EffectBrief(
        slug=row.slug,
        name=row.name,
        name_en=row.name_en,
        category=row.category,
        founder=row.founder,
        year=row.year,
        summary=row.summary,
    )


def _detail(row: Effect, related: dict[str, Effect]) -> EffectDetail:
    brief = _brief(row)
    # 数据里有些关联指向还没写进内容池的 slug，解析不到就跳过，
    # 不能因此让整张卡片渲染失败
    related_items = [_brief(related[s]) for s in (row.related or []) if s in related]
    return EffectDetail(
        **brief.model_dump(),
        detail=row.detail,
        experiment=row.experiment,
        example=row.example,
        source=row.source,
        related=related_items,
    )


def _order_key(seed: str, slug: str) -> str:
    return hashlib.blake2b(f"{seed}:{slug}".encode(), digest_size=8).hexdigest()


@router.get("/daily", response_model=EffectDaily, summary="今日心理学效应")
async def daily(
    user: CurrentUser,
    session: SessionDep,
    count: int = Query(default=1, ge=1, le=3),
    variation: int = Query(default=0, ge=0, le=99),
) -> EffectDaily:
    rows = (await session.scalars(select(Effect))).all()
    if not rows:
        return EffectDaily(date=date.today().isoformat(), variation=variation, total=0)

    seed = f"{date.today().isoformat()}|{variation}"
    ranked = sorted(rows, key=lambda row: _order_key(seed, row.slug))
    picked = ranked[:count]

    index = {row.slug: row for row in rows}
    return EffectDaily(
        date=date.today().isoformat(),
        variation=variation,
        total=len(rows),
        items=[_detail(row, index) for row in picked],
    )


@router.get("", response_model=EffectList, summary="效应总览")
async def list_effects(
    user: CurrentUser,
    session: SessionDep,
    category: str = Query(default="", max_length=24),
    q: str = Query(default="", max_length=60),
) -> EffectList:
    rows = (await session.scalars(select(Effect))).all()
    cats = sorted({row.category for row in rows if row.category})

    items = rows
    if category:
        items = [row for row in items if row.category == category]
    if q.strip():
        kw = q.strip()
        items = [
            row
            for row in items
            if kw in row.name or kw in row.name_en or kw in row.summary
        ]

    return EffectList(
        total=len(items), categories=cats, items=[_brief(row) for row in items]
    )


@router.get("/{slug}", response_model=EffectDetail, summary="效应详情")
async def get_effect(slug: str, user: CurrentUser, session: SessionDep) -> EffectDetail:
    row = await session.scalar(select(Effect).where(Effect.slug == slug))
    if row is None:
        raise AppError(ErrorCode.NOT_FOUND, "这个效应条目不存在")
    rows = (await session.scalars(select(Effect))).all()
    index = {r.slug: r for r in rows}
    return _detail(row, index)
