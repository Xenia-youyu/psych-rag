"""本地推理运行时的统一设置。

只做一件事：把 torch 的 CPU 线程数开到位。

为什么需要：torch 默认取 CPU 线程数的一半左右（这台 8 逻辑核只开 4），
对交叉编码这种长时间单次推理影响不小。实测开满后重排快约 10%。

放在单独模块而不是塞进某个 client：embedder 和 reranker 都要用，
谁先加载谁调一次，重复调用是无害的（set_num_threads 幂等）。
"""

from __future__ import annotations

import os

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_configured = False


def configure_torch_threads() -> int:
    """设置 torch 线程数并返回实际生效值。CPU-only 环境下调用。"""
    global _configured

    import torch

    if settings.embedding.device != "cpu":
        return torch.get_num_threads()

    target = settings.embedding.num_threads or (os.cpu_count() or 4)
    if not _configured:
        torch.set_num_threads(target)
        _configured = True
        logger.info(
            "runtime.threads",
            extra={
                "extra_fields": {
                    "threads": torch.get_num_threads(),
                    "cpu_count": os.cpu_count(),
                }
            },
        )
    return torch.get_num_threads()
