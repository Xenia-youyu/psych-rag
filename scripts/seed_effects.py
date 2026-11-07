"""把 data/effects.json 灌进 psy_effects 表。

内容维护在 JSON 里，代码只管同步：
- 新增条目：按 slug 插入
- 已有条目：按 slug 更新（改成什么就以 JSON 为准）
- 不删除库里多出来的（避免误删手工加的条目，要删自己写 SQL）

用法：
    python scripts/seed_effects.py            # 同步
    python scripts/seed_effects.py --dry-run  # 只看会改什么
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.base import SessionLocal, init_models  # noqa: E402
from app.db.models import Effect  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "effects.json"


async def main(apply: bool) -> None:
    items = json.loads(DATA.read_text(encoding="utf-8"))
    await init_models()

    created = updated = 0
    async with SessionLocal() as session:
        existing = {row.slug: row for row in (await session.scalars(select(Effect))).all()}
        for order, item in enumerate(items):
            row = existing.get(item["slug"])
            if row is None:
                session.add(
                    Effect(
                        slug=item["slug"],
                        name=item["name"],
                        name_en=item.get("name_en", ""),
                        category=item.get("category", "社会认知"),
                        founder=item.get("founder", ""),
                        year=item.get("year", ""),
                        summary=item["summary"],
                        detail=item["detail"],
                        experiment=item.get("experiment", ""),
                        example=item.get("example", ""),
                        related=item.get("related", []),
                        source=item.get("source", ""),
                        sort_order=order,
                    )
                )
                created += 1
            else:
                row.name = item["name"]
                row.name_en = item.get("name_en", "")
                row.category = item.get("category", "社会认知")
                row.founder = item.get("founder", "")
                row.year = item.get("year", "")
                row.summary = item["summary"]
                row.detail = item["detail"]
                row.experiment = item.get("experiment", "")
                row.example = item.get("example", "")
                row.related = item.get("related", [])
                row.source = item.get("source", "")
                row.sort_order = order
                updated += 1

        if apply:
            await session.commit()
        else:
            await session.rollback()

    print(f"{'已写入' if apply else '预演（未写入）'}：新增 {created} 条，更新 {updated} 条，源文件 {len(items)} 条")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main(apply=not args.dry_run))
