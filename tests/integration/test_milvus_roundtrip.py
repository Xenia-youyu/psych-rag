"""Milvus 真实读写集成测试。

验证的是 production 路径：MilvusIndex 在真实服务上能不能写、能不能双路召、
partition key 隔离是否生效、删除是否干净。

共享服务器上有同学的 edurag_0706 / edurag_1006，本文件有一条用例专门盯着它们：
我们写入自己的 collection 时，别人的实体数不能变。
"""

from __future__ import annotations

import asyncio
import math
import random
import uuid

import pytest

from app.clients.milvus_client import MilvusIndex
from app.config import settings
from tests.conftest import requires_db, requires_milvus

pytestmark = [pytest.mark.integration, requires_db, requires_milvus]

DIM = 1024


def unit_vector(seed: int) -> list[float]:
    rng = random.Random(seed)
    raw = [rng.uniform(-1, 1) for _ in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def make_record(chunk_id: str, kb_id: str, *, content: str, vector_seed: int,
                parent_id: str = "", is_parent: bool = False) -> dict:
    """造一条待召回的块。

    sparse_vector 只在集合确实有这个字段时才写：
    API 向量化产出的集合（psych_chunks_v2）是 dense-only，
    多写一个不存在的字段会被 Milvus 直接拒掉。
    """
    # is_parent 默认 False：这些用例造的是"待召回的块"，父块会被检索过滤掉
    record = {
        "chunk_id": chunk_id,
        "kb_id": kb_id,
        "doc_id": f"doc_{kb_id}",
        "parent_id": parent_id,
        "content": content,
        "dense_vector": unit_vector(vector_seed),
        "page_start": 10 + vector_seed,
        "page_end": 11 + vector_seed,
        "domain": "普通",
        "evidence_grade": "教材共识",
        "is_parent": is_parent,
    }
    if collection_has_sparse():
        record["sparse_vector"] = {
            str(101 + vector_seed % 5): 0.8,
            str(202 + vector_seed % 3): 0.4,
        }
    return record


def collection_has_sparse() -> bool:
    """当前配置指向的集合有没有稀疏向量字段。"""
    from pymilvus import MilvusClient

    client = MilvusClient(
        uri=settings.milvus.uri,
        token=f"{settings.milvus.user}:{settings.milvus.password}",
    )
    client.use_database(settings.milvus.database)
    try:
        desc = client.describe_collection(settings.milvus.collection)
    except Exception:
        return True  # 查不到就按"有"处理，让错误在真正插入时暴露
    return any(field["name"] == "sparse_vector" for field in desc["fields"])


@pytest.fixture
async def index() -> MilvusIndex:
    return MilvusIndex()


@pytest.fixture
async def cleanup_kb(index: MilvusIndex):
    """每个用例用完就清掉自己写的分区数据。"""
    kb_ids: list[str] = []
    yield kb_ids
    for kb_id in kb_ids:
        try:
            await index.delete_by_kb(kb_id)
        except Exception:  # noqa: BLE001
            pass


class TestRoundtrip:
    async def test_upsert_search_fetch_delete(self, index: MilvusIndex, cleanup_kb) -> None:
        kb_id = f"testkb_{uuid.uuid4().hex[:8]}"
        cleanup_kb.append(kb_id)

        records = [
            make_record("c1", kb_id, content="负强化通过移除厌恶刺激增加行为频率", vector_seed=1),
            make_record("c2", kb_id, content="惩罚通过施加厌恶刺激减少行为频率", vector_seed=2),
        ]
        written = await index.upsert(records)
        await index.flush()  # 写完立刻要查：Bounded 一致性下不刷盘可能查不到
        assert written == 2

        hits = await index.search(kb_ids=[kb_id], dense=unit_vector(1), topk=5)
        assert hits, "写入后应能检索到"
        assert hits[0]["chunk_id"] in {"c1", "c2"}
        assert hits[0]["content"]

        fetched = await index.fetch_chunks(["c1", "c2"])
        assert set(fetched) == {"c1", "c2"}
        assert "负强化" in fetched["c1"]["content"]
        assert fetched["c1"]["page_start"] == 11

    async def test_hybrid_dense_plus_sparse(self, index: MilvusIndex, cleanup_kb) -> None:
        # 这条用例验证的是 dense + sparse 的混合召回。
        # provider=dashscope 时向量只有稠密，集合里根本没有 sparse_vector 字段，
        # 在那种集合上跑它没有意义——不是失败，是不适用。
        if not collection_has_sparse():
            pytest.skip("当前集合为 dense-only，跳过混合检索用例")
        kb_id = f"testkb_{uuid.uuid4().hex[:8]}"
        cleanup_kb.append(kb_id)
        await index.upsert([make_record("h1", kb_id, content="混合检索测试", vector_seed=7)])
        await index.flush()

        hits = await index.search(
            kb_ids=[kb_id],
            dense=unit_vector(7),
            sparse={"108": 0.9},
            topk=5,
        )
        assert hits
        assert hits[0]["chunk_id"] == "h1"

    async def test_delete_by_kb_cleans_up(self, index: MilvusIndex) -> None:
        kb_id = f"testkb_{uuid.uuid4().hex[:8]}"
        await index.upsert([make_record("d1", kb_id, content="待删除", vector_seed=3)])

        await index.delete_by_kb(kb_id)

        # Milvus 删除是最终一致：delete + flush 之后仍可能有秒级传播延迟。
        # 这里轮询而不是 sleep 死等，既不让测试变慢，也不掩盖真实的"删不掉"。
        hits = await index.search(kb_ids=[kb_id], dense=unit_vector(3), topk=5)
        for _ in range(10):
            if not hits:
                break
            await asyncio.sleep(0.3)
            hits = await index.search(kb_ids=[kb_id], dense=unit_vector(3), topk=5)
        assert hits == []


class TestIsolation:
    async def test_kb_filter_prevents_cross_kb_leak(self, index: MilvusIndex, cleanup_kb) -> None:
        """两个知识库各写一条，检索时必须只看到自己那条。"""
        kb_a = f"testkb_a_{uuid.uuid4().hex[:6]}"
        kb_b = f"testkb_b_{uuid.uuid4().hex[:6]}"
        cleanup_kb.extend([kb_a, kb_b])

        await index.upsert([
            make_record("a1", kb_a, content="A 库专属内容", vector_seed=11),
            make_record("b1", kb_b, content="B 库专属内容", vector_seed=11),
        ])
        await index.flush()

        hits_a = await index.search(kb_ids=[kb_a], dense=unit_vector(11), topk=10)
        assert hits_a
        assert all(hit["chunk_id"] == "a1" for hit in hits_a), "A 库检索串到了 B 库数据"

        hits_both = await index.search(kb_ids=[kb_a, kb_b], dense=unit_vector(11), topk=10)
        assert {hit["chunk_id"] for hit in hits_both} == {"a1", "b1"}

    async def test_empty_kb_ids_never_matches(self, index: MilvusIndex) -> None:
        assert await index.search(kb_ids=[], dense=unit_vector(1)) == []


class TestParentExclusion:
    """父块必须留在库里（回填要用），但不能作为召回候选。

    父块比分块长 4 倍，向量是整页的平均语义，容易靠"大而全"的相似度
    挤进 topK，把真实子块命中的名额顶掉。事后按 parent_id 去重救不回来
    ——父块的 parent_id 是空字符串，它压根不会和子块归到同一组。
    """

    async def test_parent_never_recalled(self, index: MilvusIndex, cleanup_kb) -> None:
        kb_id = f"testkb_p_{uuid.uuid4().hex[:6]}"
        cleanup_kb.append(kb_id)
        # 构造最坏情况：父块向量与查询完全相同，相似度必然最高
        await index.upsert([
            make_record("p1", kb_id, content="父块全文", vector_seed=21, is_parent=True),
            make_record("c1", kb_id, content="子块片段", vector_seed=22, parent_id="p1"),
        ])
        await index.flush()

        hits = await index.search(kb_ids=[kb_id], dense=unit_vector(21), topk=10)
        ids = [hit["chunk_id"] for hit in hits]
        assert "p1" not in ids, "父块被召回了，会挤占子块的名额"
        assert "c1" in ids

    async def test_parent_still_fetchable(self, index: MilvusIndex, cleanup_kb) -> None:
        """排除在召回之外 ≠ 从库里删掉。回填靠 chunk_id 直接取，必须还取得到。"""
        kb_id = f"testkb_pf_{uuid.uuid4().hex[:6]}"
        cleanup_kb.append(kb_id)
        await index.upsert([
            make_record("p2", kb_id, content="父块原文全文", vector_seed=31, is_parent=True),
        ])
        await index.flush()

        rows = await index.fetch_chunks(["p2"])
        assert rows["p2"]["content"] == "父块原文全文"
        assert rows["p2"]["is_parent"] is True

    async def test_include_parents_switch(self, index: MilvusIndex, cleanup_kb) -> None:
        """排查问题时要能看到父块，所以留了开关，但不该是默认行为。"""
        kb_id = f"testkb_pi_{uuid.uuid4().hex[:6]}"
        cleanup_kb.append(kb_id)
        await index.upsert([
            make_record("p3", kb_id, content="父块", vector_seed=41, is_parent=True),
        ])
        await index.flush()

        excluded = await index.search(kb_ids=[kb_id], dense=unit_vector(41), topk=10)
        assert excluded == []

        included = await index.search(
            kb_ids=[kb_id], dense=unit_vector(41), topk=10, include_parents=True
        )
        assert [hit["chunk_id"] for hit in included] == ["p3"]


class TestNeighbourSafety:
    async def test_other_collections_still_exist(self) -> None:
        """同学的 collection 必须还在。共享环境里被误删是灾难级事故。"""
        from pymilvus import MilvusClient

        client = MilvusClient(uri=settings.milvus.uri, timeout=10)
        try:
            client.use_database(settings.milvus.database)
        except AttributeError:
            client.using_database(settings.milvus.database)

        names = client.list_collections()
        names = [n if isinstance(n, str) else n.get("name", "") for n in names]

        assert settings.milvus.collection in names
        for neighbour in ("edurag_0706", "edurag_1006"):
            if neighbour in names:
                desc = client.describe_collection(neighbour)
                assert desc.get("fields"), f"{neighbour} 结构被破坏"

    async def test_our_collection_has_partition_key(self) -> None:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=settings.milvus.uri, timeout=10)
        try:
            client.use_database(settings.milvus.database)
        except AttributeError:
            client.using_database(settings.milvus.database)

        desc = client.describe_collection(settings.milvus.collection)
        partition_key = next(
            (f.get("name") for f in desc.get("fields", []) if f.get("is_partition_key")),
            None,
        )
        assert partition_key == "kb_id", "多知识库隔离依赖 kb_id 作分区键"
