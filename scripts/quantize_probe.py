"""对比 fp32 与 int8 动态量化在 CPU 上的重排速度与排序质量。

为什么值得做：
    bge-reranker-v2-m3 是 568M 的 XLM-R large，CPU 上 fp32 单对约 1.1 秒（256 token）。
    8 个候选就是 9 秒，直接违背端到端 3 秒目标。
    Transformer 编码器的瓶颈在 Linear 层，int8 动态量化通常能拿 2~3 倍，
    但**必须验证排序质量不掉**——量化把正确答案排到第二，那就是本末倒置。

用法：
    .venv/Scripts/python.exe scripts/quantize_probe.py
"""

from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

from app.config import settings  # noqa: E402

QUERY = "负强化和惩罚的区别是什么"
GOLD = "负强化通过撤销厌恶刺激来增加行为发生的概率；惩罚通过施加厌恶刺激来减少行为，两者方向相反。"
OTHERS = [
    "经典条件作用强调中性刺激与无条件刺激的联结，反应是应答性的。",
    "最近发展区指现有水平与在成人指导下可能达到的水平之间的差距。",
    "信度是效度的必要条件而非充分条件，效度指测到目标特质的程度。",
    "自我决定理论认为人有自主、胜任和归属三种基本心理需要。",
    "皮亚杰把认知发展分为四个阶段，形式运算阶段始于十一岁以后。",
    "工作记忆容量有限，组块化能提高单位时间内的信息保持量。",
    "注意的分配需要自动化加工支撑，否则难以同时处理两个任务。",
    "艾宾浩斯遗忘曲线显示遗忘先快后慢，复习应安排在初期。",
]


def pad(text: str, target: int = 300) -> str:
    """子块量级（300 字）。"""
    return (text * (target // len(text) + 1))[:target]


def bench(model: CrossEncoder, pairs: list, label: str) -> tuple[float, list[float]]:
    model.predict(pairs)  # 预热
    t0 = time.perf_counter()
    scores = model.predict(pairs)
    dt = time.perf_counter() - t0
    per_pair = dt / len(pairs) * 1000
    print(f"  {label:8} {per_pair:7.0f} ms/对   8 候选 {per_pair*8/1000:5.1f}s")
    return per_pair, [float(s) for s in scores]


def main() -> int:
    path = settings.embedding.bge_reranker_path
    docs = [pad(GOLD)] + [pad(text) for text in OTHERS]
    pairs = [[QUERY, doc] for doc in docs]
    print(f"样本：1 条正确 + {len(docs)-1} 条干扰，每条 {len(docs[0])} 字")
    print(f"torch 线程：{torch.get_num_threads()}（逻辑核 {os.cpu_count()}）")
    print("=" * 58)

    torch.set_num_threads(int(os.cpu_count() or 4))
    fp32 = CrossEncoder(path, device="cpu", max_length=256)
    print("fp32:")
    base_ms, base_scores = bench(fp32, pairs, "baseline")
    gold_rank_fp32 = sorted(base_scores, reverse=True).index(base_scores[0]) + 1

    print("int8:")
    quantized = copy.deepcopy(fp32)
    torch.quantization.quantize_dynamic(
        quantized.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
    )
    q_ms, q_scores = bench(quantized, pairs, "int8")
    gold_rank_q = sorted(q_scores, reverse=True).index(q_scores[0]) + 1

    print("=" * 58)
    print(f"加速比        : {base_ms / q_ms:.2f}x")
    print(f"正确答案排名  : fp32 第 {gold_rank_fp32} / int8 第 {gold_rank_q}")
    print(f"分数对比      : fp32 {base_scores[0]:.4f} / int8 {q_scores[0]:.4f}")

    order_fp32 = sorted(range(len(base_scores)), key=lambda i: -base_scores[i])
    order_q = sorted(range(len(q_scores)), key=lambda i: -q_scores[i])
    same = order_fp32[:3] == order_q[:3]
    print(f"Top-3 顺序一致: {'是' if same else '否 —— 量化改变了排序，不能用'}")
    return 0 if (gold_rank_q == 1 and same) else 1


if __name__ == "__main__":
    sys.exit(main())
