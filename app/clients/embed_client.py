"""bge-m3 向量化客户端。

一次编码同时产出稠密向量与稀疏权重，这是选 bge-m3 的唯一理由：
换成两个模型分别做 dense / sparse，要么语义空间不一致，要么成本翻倍。

懒加载：模型很大（约 2.2 GB），导入即加载会把启动时间拖到无法接受。
真正的模型加载放在首次调用，且只加载一次。

未安装 FlagEmbedding 时给出明确提示，不静默降级成"假向量"——
静默降级会让检索结果看起来正常但实际全错，比直接报错危险得多。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Embedding:
    dense: list[float] = field(default_factory=list)
    sparse: dict[str, float] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.dense


class BGEM3Embedder:
    """稠密 + 稀疏双表征。"""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        conf = settings.embedding
        self.model_path = model_path or conf.bge_m3_path
        self.device = device or conf.device
        self.batch_size = batch_size or conf.batch_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        if not self.model_path:
            raise AppError(
                ErrorCode.INTERNAL,
                "未配置 [embedding] bge_m3_path，向量化不可用",
            )

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise AppError(
                ErrorCode.INTERNAL,
                "缺少 FlagEmbedding，请先安装：pip install FlagEmbedding",
            ) from exc

        from app.clients.runtime import configure_torch_threads

        configure_torch_threads()

        logger.info(
            "embedder.loading",
            extra={"extra_fields": {"path": self.model_path, "device": self.device}},
        )
        self._model = BGEM3FlagModel(
            self.model_path,
            use_fp16=self.device != "cpu",
            device=self.device,
        )
        return self._model

    def _encode_sync(self, texts: list[str]) -> list[Embedding]:
        model = self._load()
        output = model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        # 不能写 `dense_vecs or []`：dense_vecs 是 numpy 数组，
        # `or` 会触发 "truth value of an array is ambiguous"。
        # 这个 bug 只有接真模型才暴露——替身测试里它是 list，永远走不到那一行。
        dense_vectors = output.get("dense_vecs")
        sparse_weights = output.get("lexical_weights")
        if dense_vectors is None:
            dense_vectors = []
        if sparse_weights is None:
            sparse_weights = []

        results: list[Embedding] = []
        for index in range(len(texts)):
            dense = (
                [float(x) for x in dense_vectors[index]]
                if index < len(dense_vectors)
                else []
            )
            sparse_raw = sparse_weights[index] if index < len(sparse_weights) else {}
            # lexical_weights 的 key 可能是 token id（int），Milvus 需要字符串
            sparse = {str(key): float(value) for key, value in dict(sparse_raw).items()}
            results.append(Embedding(dense=dense, sparse=sparse))
        return results

    async def encode(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        # 模型推理是同步阻塞的，放线程池，避免卡住事件循环
        return await asyncio.to_thread(self._encode_sync, texts)

    async def encode_one(self, text: str) -> Embedding:
        result = await self.encode([text])
        return result[0] if result else Embedding()

    @property
    def dim(self) -> int:
        return 1024
