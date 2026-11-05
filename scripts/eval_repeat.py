"""多次测量：把检索 / 生成两侧各跑 N 轮，报均值与极差。

为什么需要这个脚本：
    生成侧的要点命中在同一配置下会出现 41/43 与 37/43 的差异（2026-11-03 实测两次），
    检索侧也不是完全确定——Milvus 的 HNSW 是近似检索，segment 结构变化会让 top-k 有
    微小漂移（2026-11-05 观察到 compare 的块级 recall 从 0.955 掉到 0.909，题目未命中
    数却没变）。单次结果不足以判断一个改动有没有效，比较指标必须给区间。

用法：
    # 各跑 3 轮（约 45 分钟：检索 5.5 分钟 + 生成 9 分钟，乘以轮数）
    .venv/Scripts/python.exe scripts/eval_repeat.py --repeat 3

    # 只测检索（约 17 分钟）
    .venv/Scripts/python.exe scripts/eval_repeat.py --repeat 3 --only retrieve

产出：
    data/golden/repeat_<时间戳>.json —— 每轮明细 + 均值 + 极差
    控制台表格 —— 直接能贴进汇报
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
REPORT_DIR = ROOT / "data" / "golden"

# 检索侧表格行：concept       43      0.953    0.806    0.977
RET_LINE = re.compile(r"^(concept|compare|multihop)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
# 生成侧表格行：concept      43    100%       89%     37/43         0     5.0
GEN_LINE = re.compile(r"^(concept|compare|multihop)\s+(\d+)\s+(\d+)%\s+(\d+)%\s+(\d+)/(\d+)")


def run_script(script: str, log: Path) -> str:
    env = {"PYTHONPATH": str(ROOT), "JWT_SECRET": "dev-local-only", "PATH": str(Path(sys.executable).parent)}
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [str(PY), str(ROOT / script)],
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**__import__("os").environ, **env},
        )
    text = log.read_text(encoding="utf-8", errors="ignore")
    print(f"  {script} 退出码 {proc.returncode}，耗时 {time.perf_counter() - started:.0f}s")
    return text


def parse_retrieve(text: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        m = RET_LINE.match(line.strip())
        if m:
            kind, n, recall, mrr, hit = m.groups()
            out[kind] = {"n": int(n), "recall@10": float(recall), "mrr": float(mrr), "hit@5": float(hit)}
    miss = re.search(r"耗时\s*[\d.]+\s*s\s*\|\s*未命中\s*(\d+)\s*条", text)
    if miss:
        out["_misses"] = {"n": int(miss.group(1))}
    return out


def parse_generation(text: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        m = GEN_LINE.match(line.strip())
        if m:
            kind, n, answered, kp_rate, passed, total = m.groups()
            out[kind] = {
                "n": int(n),
                "kp_rate": float(kp_rate) / 100,
                "passed": int(passed),
                "total": int(total),
            }
    return out


def summarize(rounds: list[dict], keys: list[tuple[str, str]]) -> None:
    print(f"\n{'指标':<26}{'各轮':<30}{'均值':>8}{'极差':>10}")
    print("-" * 76)
    for kind, metric in keys:
        vals = []
        for r in rounds:
            v = (r.get(kind) or {}).get(metric)
            if v is not None:
                vals.append(v)
        if not vals:
            continue
        label = f"{kind}.{metric}"
        spread = max(vals) - min(vals)
        shown = " / ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in vals)
        print(f"{label:<26}{shown:<30}{statistics.mean(vals):>8.3f}{spread:>10.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="多次测量取均值")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--only", choices=["retrieve", "generate", "both"], default="both")
    args = ap.parse_args()

    if not PY.is_file():
        print(f"找不到解释器 {PY}")
        return 1

    ret_rounds: list[dict] = []
    gen_rounds: list[dict] = []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for i in range(1, args.repeat + 1):
        print(f"\n===== 第 {i}/{args.repeat} 轮 =====")
        if args.only in ("retrieve", "both"):
            ret_rounds.append(parse_retrieve(run_script("scripts/eval_run.py", ROOT / f"_repeat_ret_{i}.log")))
        if args.only in ("generate", "both"):
            gen_rounds.append(
                parse_generation(run_script("scripts/eval_generation.py", ROOT / f"_repeat_gen_{i}.log"))
            )

    if ret_rounds:
        summarize(ret_rounds, [(k, m) for k in ("concept", "compare", "multihop")
                               for m in ("recall@10", "mrr", "hit@5")])
        misses = [r.get("_misses", {}).get("n") for r in ret_rounds]
        if any(m is not None for m in misses):
            print(f"{'未命中条数':<26}{' / '.join(str(m) for m in misses)}")
    if gen_rounds:
        summarize(gen_rounds, [(k, "kp_rate") for k in ("concept", "compare", "multihop")])
        print()
        for i, r in enumerate(gen_rounds, 1):
            passed = sum((r.get(k) or {}).get("passed", 0) for k in ("concept", "compare", "multihop"))
            total = sum((r.get(k) or {}).get("total", 0) for k in ("concept", "compare", "multihop"))
            print(f"  第 {i} 轮 知识题要点过关 {passed}/{total}" +
                  (f" = {passed/total:.1%}" if total else ""))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"repeat_{stamp}.json"
    out.write_text(
        json.dumps({"repeat": args.repeat, "only": args.only,
                    "retrieve": ret_rounds, "generation": gen_rounds}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n明细已存 → {out}")
    for log in ROOT.glob("_repeat_*.log"):
        log.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
