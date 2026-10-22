"""文档上传与管理。

上传接口只做三件事：校验 → 落盘 → 建记录（status=uploaded）。
解析与向量化走后台任务（M2 接 MinerU 后实现），接口立即返回，
前端靠轮询文档列表看 status 变化。

为什么不让上传接口同步解析：
一本 300 页扫描版教材走 vlm backend 要十几分钟，HTTP 早超时了。
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from sqlalchemy import delete, func, select

from app.api.kb import get_owned_kb
from app.config import BASE_DIR, settings
from app.core.deps import CurrentUser, SessionDep
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.db.models import Chunk, Document
from app.ingest.detect import SUPPORTED_EXTENSIONS, plan_for
from app.schemas.kb import DocumentOut, MessageOut, ParseParams

router = APIRouter(prefix="/api", tags=["文档"])
logger = get_logger(__name__)

UPLOAD_CHUNK = 1 << 20  # 1MB


def _upload_root() -> Path:
    root = Path(settings.app.upload_dir)
    return root if root.is_absolute() else BASE_DIR / root


@router.post(
    "/kb/{kb_id}/documents",
    response_model=DocumentOut,
    summary="上传文档（异步解析）",
)
async def upload_document(
    background: BackgroundTasks,
    kb_id: str,
    user: CurrentUser,
    session: SessionDep,
    file: UploadFile = File(...),
    chunk_size: int = Form(800),
    overlap_ratio: float = Form(0.10),
    hybrid: bool = Form(True),
    mode: str = Form("single"),
    parent_size: int = Form(1200),
    child_size: int = Form(320),
) -> DocumentOut:
    kb = await get_owned_kb(kb_id, user.id, session)

    filename = (file.filename or "").strip()
    if not filename:
        raise AppError(ErrorCode.VALIDATION, "缺少文件名")

    # 扩展名校验用统一的 detect 模块，避免各处各写一份白名单
    plan = plan_for(filename)

    limit_bytes = settings.app.max_upload_mb * 1024 * 1024
    params = ParseParams(
        chunk_size=chunk_size,
        overlap_ratio=overlap_ratio,
        hybrid=hybrid,
        mode="parent_child" if mode == "parent_child" else "single",
        parent_size=parent_size,
        child_size=child_size,
    )

    doc_id = uuid.uuid4().hex[:16]
    target_dir = _upload_root() / kb_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{doc_id}{plan.extension}"

    written = 0
    digest = hashlib.sha256()
    with target_path.open("wb") as sink:
        while True:
            block = await file.read(UPLOAD_CHUNK)
            if not block:
                break
            written += len(block)
            digest.update(block)
            if written > limit_bytes:
                sink.close()
                # 清理失败不能盖过真正的错误原因：用户需要看到"文件太大"，
                # 而不是一个 unlink 的 OSError
                try:
                    target_path.unlink(missing_ok=True)
                except OSError as exc:  # noqa: BLE001
                    logger.warning(
                        "doc.cleanup_failed",
                        extra={"extra_fields": {"path": str(target_path), "error": repr(exc)}},
                    )
                raise AppError(
                    ErrorCode.FILE_TOO_LARGE,
                    f"文件超过 {settings.app.max_upload_mb}MB 上限",
                )
            sink.write(block)

    # 内容哈希去重：同一本书重复上传会产生两份一模一样的向量，
    # 检索时用户会看到两条相同内容，且清理起来很麻烦（要按 doc_id 逐条删）。
    # 判重范围限定在同一个知识库内——同一本书进两个库是合理需求。
    content_hash = digest.hexdigest()
    duplicate = await session.scalar(
        select(Document).where(
            Document.kb_id == kb_id,
            Document.content_sha256 == content_hash,
            Document.status != "failed",  # 上次失败的允许重传
        )
    )
    if duplicate is not None:
        try:
            target_path.unlink(missing_ok=True)
        except OSError as exc:  # noqa: BLE001
            logger.warning(
                "doc.cleanup_failed",
                extra={"extra_fields": {"path": str(target_path), "error": repr(exc)}},
            )
        logger.info(
            "doc.duplicate_skipped",
            extra={
                "extra_fields": {
                    "kb_id": kb_id,
                    "existing_doc": duplicate.id,
                    "sha256": content_hash[:12],
                }
            },
        )
        existing = DocumentOut.model_validate(duplicate)
        existing.duplicated = True
        return existing

    document = Document(
        id=doc_id,
        kb_id=kb_id,
        filename=filename,
        file_type=plan.extension.lstrip("."),
        size_bytes=written,
        status="uploaded",
        last_stage="uploaded",
        parse_params=params.model_dump(),
        storage_path=str(target_path),
        content_sha256=content_hash,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    # doc_count 必须等文档真正落库后再统计，否则会把自己多算一次
    kb.doc_count = int(
        await session.scalar(
            select(func.count(Document.id)).where(Document.kb_id == kb_id)
        )
        or 0
    )
    await session.commit()

    logger.info(
        "doc.uploaded",
        extra={
            "extra_fields": {
                "doc_id": doc_id,
                "kb_id": kb_id,
                "filename": filename,
                "size": written,
            }
        },
    )

    # 解析与向量化丢到后台。注意 FastAPI 的 BackgroundTasks 是「响应返回后在同一个
    # worker 里继续跑」，一本 300 页扫描件会占住这个 worker 十几分钟。
    # 单机自用够了；真要多用户并发，得换 Celery / RQ 这类真队列。
    background.add_task(_run_ingest, doc_id)

    return DocumentOut.model_validate(document)


async def _run_ingest(doc_id: str) -> None:
    """后台入库。异常已在 pipeline 内部兜住并写进 error_msg，这里只记日志。"""
    from app.ingest.pipeline import IngestPipeline

    report = await IngestPipeline().run(doc_id)
    if report.ok:
        logger.info(
            "doc.ingest_ok",
            extra={
                "extra_fields": {
                    "doc_id": doc_id,
                    "pages": report.pages,
                    "children": report.children,
                    "vectors": report.vectors,
                    "elapsed_ms": report.elapsed_ms,
                }
            },
        )
    else:
        logger.warning(
            "doc.ingest_failed",
            extra={"extra_fields": {"doc_id": doc_id, "error": report.error}},
        )


@router.get(
    "/kb/{kb_id}/documents",
    response_model=list[DocumentOut],
    summary="文档列表",
)
async def list_documents(kb_id: str, user: CurrentUser, session: SessionDep) -> list[DocumentOut]:
    await get_owned_kb(kb_id, user.id, session)
    rows = await session.scalars(
        select(Document).where(Document.kb_id == kb_id).order_by(Document.created_at.desc())
    )
    return [DocumentOut.model_validate(row) for row in rows]


@router.get("/documents/{doc_id}", response_model=DocumentOut, summary="文档详情")
async def get_document(doc_id: str, user: CurrentUser, session: SessionDep) -> DocumentOut:
    document = await _get_owned_doc(doc_id, user.id, session)
    return DocumentOut.model_validate(document)


@router.delete("/documents/{doc_id}", response_model=MessageOut, summary="删除文档")
async def delete_document(doc_id: str, user: CurrentUser, session: SessionDep) -> MessageOut:
    document = await _get_owned_doc(doc_id, user.id, session)
    kb_id = document.kb_id
    storage_path = document.storage_path

    await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
    await session.delete(document)
    await session.commit()

    # 落盘文件同步清理，避免存储垃圾
    path = Path(storage_path) if storage_path else None
    if path and path.exists():
        try:
            path.unlink()
        except OSError as exc:  # noqa: BLE001
            logger.warning(
                "doc.unlink_failed",
                extra={"extra_fields": {"doc_id": doc_id, "error": repr(exc)}},
            )

    logger.info("doc.deleted", extra={"extra_fields": {"doc_id": doc_id, "kb": kb_id}})
    return MessageOut(message="文档已删除")


async def _get_owned_doc(doc_id: str, user_id: int, session: SessionDep) -> Document:
    """取文档并沿 kb → owner 校验归属。越权一律 404。"""
    document = await session.scalar(select(Document).where(Document.id == doc_id))
    if document is None:
        raise AppError(ErrorCode.DOC_NOT_FOUND)

    from app.db.models import KnowledgeBase

    owner_id = await session.scalar(
        select(KnowledgeBase.owner_id).where(KnowledgeBase.id == document.kb_id)
    )
    if owner_id != user_id:
        raise AppError(ErrorCode.DOC_NOT_FOUND)
    return document


@router.get("/supported-types", summary="支持的文档类型")
async def supported_types() -> dict:
    return {"extensions": SUPPORTED_EXTENSIONS, "max_upload_mb": settings.app.max_upload_mb}
