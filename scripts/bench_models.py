"""模型实测脚本：验证 bge-m3 与 reranker 能真正加载，并测出 CPU 上的速度。

为什么要单独测：
    单元测试全部用替身，跑得再绿也证明不了"模型能加载、能出双路向量"。
    而 CPU 上的真实速度决定了要不要调 rerank_max_candidates / rerank_topk——
    这两个参数拍脑袋定是没意义的，必须实测。

用法：
    .venv/Scripts/python.exe scripts/bench_models.py
    .venv/Scripts/python.exe scripts/bench_models.py --encode-only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.embed_client import BGEM3Embedder  # noqa: E402
from app.clients.rerank_client import BGEReranker  # noqa: E402
from app.config import settings  # noqa: E402

QUERIES = [
    "负强化和惩罚的区别是什么",
    "操作性条件作用和经典条件作用怎么区分",
    "最近发展区是谁提出来的",
    "皮亚杰认知发展阶段有哪几个",
]

DOCS = [
    "负强化是通过撤销厌恶刺激来增加行为发生的概率；惩罚是通过施加厌恶刺激或撤销愉快刺激来减少行为。"
    "两者方向相反：负强化让行为变多，惩罚让行为变少。这是普通心理学教材中最容易被学生混淆的一对概念。",
    "经典条件作用由巴甫洛夫提出，强调中性刺激与无条件刺激的联结，反应是应答性的、不随意的；"
    "操作性条件作用由斯金纳系统阐述，强调行为的结果对行为的塑造，反应是操作性的、主动发出的。",
    "最近发展区由维果茨基提出，指儿童现有发展水平与在成人指导下可能达到的水平之间的差距。"
    "教学应当走在发展的前面。",
    "皮亚杰把认知发展分为感知运动阶段、前运算阶段、具体运算阶段和形式运算阶段四个阶段，"
    "分别对应 0-2 岁、2-7 岁、7-11 岁和 11 岁以后。",
    "心理测量学中，信度指测量结果的稳定性与一致性，效度指测量到所要测量特质的程度。"
    "信度是效度的必要而非充分条件。",
    "自我决定理论认为人有自主、胜任和归属三种基本心理需要，满足这三种需要能促进内在动机。",
]


async def main() -> int:
    parser = argparse.ArgumentParser(description="实测 bge-m3 与 reranker")
    parser.add_argument("--encode-only", action="store_true", help="只测向量化，不加载重排")
    args = parser.parse_args()

    conf = settings.embedding
    print("=" * 66)
    print(f"bge-m3     : {conf.bge_m3_path}")
    print(f"reranker   : {conf.bge_reranker_path}")
    print(f"device     : {conf.device} | batch_size: {conf.batch_size}")
    print("=" * 66)

    # ---- 1. 向量化 ----
    embedder = BGEM3Embedder()
    t0 = time.perf_counter()
    vecs = await embedder.encode(QUERIES[:1])
    load_and_first = time.perf_counter() - t0
    print(f"\n[向量化] 首次调用（含加载模型）: {load_and_first:.1f}s")

    first = vecs[0]
    print(f"  dense 维度      : {len(first.dense)}")
    print(f"  sparse 非零项   : {len(first.sparse)}")
    print(f"  dense 前 3 值   : {[round(x, 4) for x in first.dense[:3]]}")
    sample = list(first.sparse.items())[:3]
    print(f"  sparse 样例     : {[(k, round(v, 3)) for k, v in sample]}")
    if len(first.dense) != 1024:
        print("  ✗ dense 维度不是 1024，与 Milvus schema 不符")
        return 1
    if not first.sparse:
        print("  ✗ 稀疏权重为空，双路召回会退化成单路")
        return 1

    # 批量吞吐
    batch = QUERIES * 2
    t0 = time.perf_counter()
    await embedder.encode(batch)
    dt = time.perf_counter() - t0
    print(f"\n[向量化] 批量 {len(batch)} 条: {dt:.2f}s → {len(batch)/dt:.1f} 条/秒")

    if args.encode_only:
        return 0

    # ---- 2. 重排 ----
    reranker = BGEReranker()
    t0 = time.perf_counter()
    scores = reranker._score_sync(QUERIES[0], [DOCS[0]])
    load_pair = time.perf_counter() - t0
    print(f"\n[重排] 首次调用（含加载模型）: {load_pair:.1f}s")
    print(f"  单对分数        : {round(scores[0], 4)}（sigmoid 后的 0~1 概率）")

    t0 = time.perf_counter()
    all_scores = reranker._score_sync(QUERIES[0], DOCS)
    dt = time.perf_counter() - t0
    per_pair = dt / len(DOCS)
    print(f"  批量 {len(DOCS)} 对      : {dt:.2f}s → {per_pair*1000:.0f} ms/对")

    ranked = sorted(zip(all_scores, DOCS), key=lambda x: -x[0])
    print("\n  排序结果（第一条应命中负强化/惩罚）:")
    for score, doc in ranked[:3]:
        print(f"    {score:.4f}  {doc[:36]}…")

    # ---- 3. 反推线上延迟预算 ----
    cap = settings.retrieval.rerank_max_candidates
    print("\n" + "=" * 66)
    print(f"按 rerank_max_candidates={cap} 估算线上重排耗时: {cap * per_pair:.1f}s")
    if cap * per_pair > 3:
        print("  ⚠ 超过 3 秒目标，建议把 rerank_max_candidates 往下调")
    else:
        print("  ✓ 在 3 秒目标内")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
