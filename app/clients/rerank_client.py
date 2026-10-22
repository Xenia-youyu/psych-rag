"""bge-reranker-v2-m3 重排客户端。

用途：对粗排后的候选做精排，取前 5~8 条。
重排是 RAG 里性价比最高的一个环节——一次交叉编码，通常能把 Top-5 命中率拉十几个点。

为什么用 sentence_transformers.CrossEncoder 而不是 FlagEmbedding.FlagReranker：
    - CrossEncoder 是通用 API，换任何 BERT 类重排模型都不用改代码；
      FlagReranker 与 BGE 系绑定，且不同大版本间签名常变。
    - 参考项目 Itcast_qa_system 实测跑通的也是这条路。
    两者底层是同一个模型、同一套打分，差别只在 API 稳定性。

max_length 刻意设成 512：
    不设的话会按模型配置吃满 8192 token。父块 1200 字的正文远超 512 token 的
    有效判别区间，多出来的计算不提升排序质量，只让 CPU 上单对耗时从 0.2 秒涨到 2 秒。
"""

from __future__ import annotations

import asyncio
import math

from app.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger

logger = get_logger(__name__)


def _sigmoid(value: float) -> float:
    """数值稳定的 sigmoid：大负数直接返回 0，避免 exp 溢出。"""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


class BGEReranker:
    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str | None = None,
        max_length: int | None = None,
        quantize: bool | None = None,
    ) -> None:
        conf = settings.embedding
        self.model_path = model_path or conf.bge_reranker_path
        self.device = device or conf.device
        self.max_length = max_length or settings.retrieval.rerank_max_length
        self.quantize = (
            settings.retrieval.rerank_quantize if quantize is None else quantize
        )
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        if not self.model_path:
            raise AppError(
                ErrorCode.INTERNAL,
                "未配置 [embedding] bge_reranker_path，重排不可用",
            )

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise AppError(
                ErrorCode.INTERNAL,
                "缺少 sentence-transformers，请先安装：uv pip install sentence-transformers",
            ) from exc

        from app.clients.runtime import configure_torch_threads

        configure_torch_threads()

        logger.info(
            "reranker.loading",
            extra={
                "extra_fields": {
                    "path": self.model_path,
                    "device": self.device,
                    "max_length": self.max_length,
                    "quantize": self.quantize,
                }
            },
        )
        model = CrossEncoder(
            self.model_path,
            device=self.device,
            max_length=self.max_length,
        )

        if self.quantize:
            # 实测（scripts/quantize_probe.py，300 字文档 / 256 token）：
            #   fp32 1329 ms/对 → int8 470 ms/对，2.83 倍
            #   正确文档两版都排第一，分数 0.9924 → 0.9863，对判定无影响
            # 568M 的 XLM-R large 在 CPU 上不量化基本不可用——8 个候选要 10 秒。
            # inplace=True 就地替换，避免 deepcopy 把内存翻倍到 4 GB。
            try:
                import torch

                torch.quantization.quantize_dynamic(
                    model.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
                )
                logger.info("reranker.quantized", extra={"extra_fields": {"dtype": "qint8"}})
            except Exception as exc:  # noqa: BLE001
                # 量化失败退回 fp32：只是慢，不影响正确性
                logger.warning(
                    "reranker.quantize_failed",
                    extra={"extra_fields": {"error": repr(exc)}},
                )

        self._model = model
        return self._model

    def _score_sync(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        model = self._load()
        pairs = [[query, document] for document in documents]
        raw = model.predict(pairs)

        # predict 返回的是**原始 logits**（可正可负，量级 -10~10），不是概率。
        # 不做 sigmoid 的话，config 里 rerank_score_threshold=0.15 这个门槛毫无意义：
        # 大多数正确命中的 logit 是正数，会被当成"通过"；而若按 0.15 筛，
        # 一条 logit=-3 的好结果和 logit=0.1 的垃圾结果表现一样。
        # FlagReranker(compute_score(normalize=True)) 内部就是 sigmoid，
        # 这里显式补上，保证换实现后阈值语义不变。
        try:
            values = [float(item) for item in raw]
        except TypeError:
            values = [float(raw)]
        return [_sigmoid(value) for value in values]

    async def score(self, query: str, documents: list[str]) -> list[float]:
        return await asyncio.to_thread(self._score_sync, query, documents)
