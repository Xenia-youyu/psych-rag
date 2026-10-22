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
import os
from dataclasses import dataclass, field

import httpx

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


class DashScopeEmbedder:
    """百炼 text-embedding-v4（OpenAI 兼容接口）。

    只有稠密向量，没有 sparse / colbert，这是一个明确的取舍：

        本机 4 物理核跑 bge-m3 → 3.7 秒/向量，4.8 万页要 257 小时
        走 API → 约 0.25 秒/条，0.5 元/百万 token，全量约 15 元

    代价是混合检索退化成纯稠密检索，术语精确匹配会弱一些（重排能补一部分）。

    失败重试放在这里而不是调用方：批处理跑到第 3 万条时因为一次 429 整体失败，
    重来的成本远高于这里退避三次。
    """

    max_retries = 3

    def __init__(self) -> None:
        conf = settings.embedding
        self.base_url = conf.dashscope_base_url.rstrip("/")
        self.model = conf.dashscope_model
        self._dim = conf.dashscope_dim
        self.batch_size = max(1, min(conf.dashscope_batch_size, 10))
        self.concurrency = max(1, conf.dashscope_concurrency)
        self.api_key = os.getenv(conf.dashscope_api_key_env, "")
        self.total_tokens = 0

    def _ensure_key(self) -> None:
        if not self.api_key:
            raise AppError(
                ErrorCode.INTERNAL,
                f"缺少环境变量 {settings.embedding.dashscope_api_key_env}，"
                "向量化不可用（[embedding] provider=dashscope）",
            )

    async def _post(self, client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self._dim,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(
                    f"{self.base_url}/embeddings", headers=headers, json=payload
                )
            except httpx.HTTPError as exc:
                last_error = repr(exc)
            else:
                if response.status_code < 400:
                    data = response.json()
                    usage = data.get("usage") or {}
                    self.total_tokens += int(usage.get("total_tokens", 0) or 0)
                    # 接口不保证顺序，按 index 排回来，否则向量会和块错位
                    items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
                    return [[float(x) for x in item["embedding"]] for item in items]

                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                # 4xx 除了限流都不值得重试
                if response.status_code < 500 and response.status_code != 429:
                    break

            if attempt < self.max_retries:
                await asyncio.sleep(0.8 * (2**attempt))

        logger.error("embed.dashscope_failed", extra={"extra_fields": {"error": last_error}})
        raise AppError(ErrorCode.LLM_FAILED, f"向量化失败：{last_error[:120]}")

    async def encode(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        self._ensure_key()

        batches = [
            texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)
        ]
        semaphore = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient(timeout=60) as client:

            async def run(batch: list[str]) -> list[list[float]]:
                async with semaphore:
                    return await self._post(client, batch)

            groups = await asyncio.gather(*(run(batch) for batch in batches))

        results: list[Embedding] = []
        for group in groups:
            results.extend(Embedding(dense=vector, sparse={}) for vector in group)
        return results

    async def encode_one(self, text: str) -> Embedding:
        result = await self.encode([text])
        return result[0] if result else Embedding()

    @property
    def dim(self) -> int:
        return self._dim


def build_embedder() -> BGEM3Embedder | DashScopeEmbedder:
    """按配置选向量化实现。调用方不关心是本地还是 API。"""
    if settings.embedding.provider == "dashscope":
        return DashScopeEmbedder()
    return BGEM3Embedder()
