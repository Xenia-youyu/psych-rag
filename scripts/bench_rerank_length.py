"""按文档长度测重排耗时：搞清楚线上真实成本。

为什么单独测：
    bench_models.py 里用的是 60 字短句，编码只有几十个 token，
    测出来 116 ms/对——漂亮但没有参考价值。
    线上喂的是 1200 字的父块（被 max_length 截到 256 token），
    成本完全不是一回事。参数决策必须按真实长度来。

用法：
    .venv/Scripts/python.exe scripts/bench_rerank_length.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from app.clients.rerank_client import BGEReranker  # noqa: E402
from app.config import settings  # noqa: E402

QUERY = "负强化和惩罚的区别是什么"
BASE = (
    "负强化通过撤销厌恶刺激来增加行为发生的概率；惩罚通过施加厌恶刺激或撤销愉快刺激来减少行为。"
    "两者方向相反：负强化让行为变多，惩罚让行为变少，这是学生最容易混淆的一对概念。"
)


def pad(text: str, target: int) -> str:
    return (text * (target // len(text) + 1))[:target]


def main() -> int:
    conf = settings.retrieval
    reranker = BGEReranker()
    reranker._load()  # 显式加载，把加载耗时排除在测量之外

    print(f"max_length={conf.rerank_max_length} | quantize={conf.rerank_quantize} "
          f"| threads={torch.get_num_threads()} | cpu={os.cpu_count()}")
    print("=" * 60)
    print(f"{'文档长度':>8} {'ms/对':>9} {'8 候选':>9} {'6 候选':>9}")
    print("-" * 60)

    rows = []
    for length in (60, 300, 800, 1200):
        docs = [pad(BASE, length) for _ in range(4)]
        reranker._score_sync(QUERY, docs[:1])  # 预热
        t0 = time.perf_counter()
        reranker._score_sync(QUERY, docs)
        dt = time.perf_counter() - t0
        per_pair = dt / len(docs) * 1000
        rows.append((length, per_pair))
        print(f"{length:>8} {per_pair:>9.0f} {per_pair*8/1000:>8.1f}s {per_pair*6/1000:>8.1f}s")

    print("=" * 60)
    parent_cost = dict(rows)[1200]
    print(f"线上真实成本（1200 字父块，即 chunk.parent_size）: {parent_cost:.0f} ms/对")
    print(f"  rerank_max_candidates=8 → {parent_cost*8/1000:.1f}s")
    print(f"  rerank_max_candidates=6 → {parent_cost*6/1000:.1f}s")
    print(f"  rerank_max_candidates=4 → {parent_cost*4/1000:.1f}s")
    print("\n注意：60 字与 1200 字在 max_length=256 下都会截到 256 token，")
    print("但实测仍有差异，说明短文本没有把 batch 填满，padding 变少 → 更快。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
