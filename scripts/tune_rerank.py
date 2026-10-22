"""重排参数扫描：找 CPU 上的「质量不掉、速度可接受」组合。

背景：
    实测 bge-reranker-v2-m3 在 CPU 上 361 ms/对（max_length=512、torch 4 线程），
    rerank_max_candidates=15 → 5.4 秒，超出 3 秒目标。
    能调的两个旋钮：torch 线程数、交叉编码的 token 上限。
    但截断长度会影响排序质量——所以每个组合都要检查"正确答案是否仍排第一"。

用法：
    .venv/Scripts/python.exe scripts/tune_rerank.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

from app.config import settings  # noqa: E402

QUERY = "负强化和惩罚的区别是什么"

# 一条正确答案 + 三条干扰项，长度贴近父块（约 1200 字）
GOLD = (
    "负强化是通过撤销厌恶刺激来增加行为发生的概率；惩罚是通过施加厌恶刺激或撤销愉快刺激来减少行为。"
    "两者方向相反：负强化让行为变多，惩罚让行为变少。这是普通心理学教材中最容易被学生混淆的一对概念。"
)
DISTRACTORS = [
    "经典条件作用由巴甫洛夫提出，强调中性刺激与无条件刺激的联结，反应是应答性的、不随意的。",
    "最近发展区由维果茨基提出，指儿童现有发展水平与在成人指导下可能达到的水平之间的差距。",
    "信度指测量结果的稳定性与一致性，效度指测量到所要测量特质的程度，信度是效度的必要条件。",
]


def pad(text: str, target: int = 1200) -> str:
    """把文本补到父块量级，避免用短文本测出虚高的速度。"""
    return (text * (target // len(text) + 1))[:target]


def run(max_length: int, threads: int, docs: list[str]) -> tuple[float, bool]:
    torch.set_num_threads(threads)
    model = CrossEncoder(
        settings.embedding.bge_reranker_path, device="cpu", max_length=max_length
    )
    model.predict([[QUERY, docs[0]]])  # 预热，排除首次编译/缓存开销

    t0 = time.perf_counter()
    scores = model.predict([[QUERY, doc] for doc in docs])
    dt = time.perf_counter() - t0

    gold_first = float(scores[0]) == max(float(s) for s in scores)
    return dt / len(docs) * 1000, gold_first


def main() -> int:
    docs = [pad(GOLD)] + [pad(text) for text in DISTRACTORS]
    print(f"样本：1 条正确答案 + {len(docs)-1} 条干扰项，每条约 {len(docs[0])} 字")
    print(f"机器：{torch.get_num_threads()} 线程（默认），逻辑核 {__import__('os').cpu_count()}")
    print("=" * 62)
    print(f"{'线程':>4} {'max_len':>8} {'ms/对':>8} {'15 候选耗时':>12}  正确答案第一")
    print("-" * 62)

    results = []
    for threads in (8, 6, 4):
        for max_length in (512, 384, 256):
            per_pair, gold_first = run(max_length, threads, docs)
            results.append((threads, max_length, per_pair, gold_first))
            flag = "✓" if gold_first else "✗ 质量掉了"
            print(f"{threads:>4} {max_length:>8} {per_pair:>8.0f} {per_pair*15/1000:>11.1f}s  {flag}")

    print("=" * 62)
    ok = [r for r in results if r[3]]
    if not ok:
        print("没有组合保住排序质量，别调截断了，改用更小的重排模型")
        return 1
    best = min(ok, key=lambda r: r[2])
    print(f"最快且质量达标：线程 {best[0]}、max_length {best[1]} → {best[2]:.0f} ms/对")
    print(f"按 15 候选估算：{best[2]*15/1000:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
