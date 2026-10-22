"""下载本地模型权重。

本机没有 GPU（nvidia-smi 不存在），所以下的都是能在 CPU 上跑的权重。
bge-m3 编码约 10~30 条/秒，reranker 每对 20~50ms。别期待 GPU 的速度。

用法：
    pip install modelscope
    python -m scripts.download_models                  # 只下 bge 两个（约 4.6 GB）
    python -m scripts.download_models --with-mineru    # 加上 MinerU pipeline（约 +2.0 GB）
    python -m scripts.download_models --source hf      # 走 HuggingFace

为什么默认不下 MinerU 的 VLM 模型：
    那是一个 1.2B 的视觉语言模型，官方要 20GB+ 显存。本机无 GPU，
    纯 CPU 推理单页要几十秒，40,274 页扫描件跑完是几百小时——不可行。
    扫描页的兜底交给 qwen3-max API（config.ini 的 [vl] 段），
    MinerU 只跑 pipeline 后端（版面 + OCR + 公式 + 表格，全小模型）。
    确实想试 VLM 就加 --mineru-mode vlm。
"""

from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

# 这里刻意不 import app.config：
# 那个模块会做 fail-fast 校验（要求 DEEPSEEK_API_KEY 等环境变量齐全），
# 但"下载本地权重"这件事跟 API Key 一点关系都没有。
# 为了下模型先配一堆密钥，属于把校验放错了层。
CONFIG_INI = Path(__file__).resolve().parent.parent / "config.ini"


def model_paths() -> tuple[str, str]:
    parser = configparser.ConfigParser()
    parser.read(CONFIG_INI, encoding="utf-8")
    section = parser["embedding"] if parser.has_section("embedding") else {}
    return (
        section.get("bge_m3_path", "D:/models/bge-m3"),
        section.get("bge_reranker_path", "D:/models/bge-reranker-v2-m3"),
    )


_EMBED_PATH, _RERANK_PATH = model_paths()

MODELS = {
    "bge-m3": ("BAAI/bge-m3", _EMBED_PATH),
    "bge-reranker-v2-m3": ("BAAI/bge-reranker-v2-m3", _RERANK_PATH),
}


def human(path: Path) -> str:
    try:
        return f"{sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1024**3:.2f} GB"
    except OSError:
        return "-"


def download_modelscope(repo_id: str, target: Path) -> None:
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise SystemExit("缺少 modelscope，先执行：pip install modelscope modelscope-hub") from None

    target.mkdir(parents=True, exist_ok=True)
    # 必须排除 onnx 与 demo 素材。
    # bge-m3 仓库里同时放 pytorch_model.bin(2.27GB) 和 onnx/model.onnx_data(2.27GB)，
    # 不排除的话一个模型下 4.6GB，翻倍且没有任何用处——FlagEmbedding 只认 PyTorch 权重。
    snapshot_download(
        repo_id,
        local_dir=str(target),
        ignore_patterns=["onnx/*", "*.onnx", "*.onnx_data", "imgs/*", "*.jpg", "*.png", "*.gif"],
    )


def download_hf(repo_id: str, target: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("缺少 huggingface_hub，先执行：pip install huggingface_hub") from None

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        ignore_patterns=["onnx/*", "*.onnx", "*.onnx_data", "imgs/*", "*.jpg", "*.png", "*.gif"],
    )


def verify(target: Path) -> tuple[bool, str]:
    """模型目录至少要有关键文件，不能是个空壳。"""
    if not target.exists():
        return False, "目录不存在"

    required = ("config.json",)
    missing = [name for name in required if not (target / name).exists()]
    if missing:
        return False, f"缺关键文件 {missing}"

    weights = list(target.glob("*.bin")) + list(target.glob("*.safetensors"))
    if not weights:
        return False, "没有权重文件（*.bin / *.safetensors）"

    return True, f"{len(weights)} 个权重文件 / {human(target)}"


def mineru_model_root() -> Path:
    """MinerU 的模型落在哪。

    MinerU 自己认 MINERU_MODEL_SOURCE 环境变量决定缓存位置，
    但下载完会用 `mineru-models-download` 写一份 ~/mineru.json 指向实际路径，
    所以这里只做「有没有下成功」的判断，不假设路径。
    """
    import os

    for env in ("MINERU_MODEL_PATH", "MODELSCOPE_CACHE"):
        raw = os.environ.get(env)
        if raw:
            return Path(raw)
    return Path.home() / ".cache" / "modelscope" / "hub"


def download_mineru(mode: str) -> bool:
    """调用 MinerU 官方下载器。它自己会按 MINERU_MODEL_SOURCE 选源并写配置。"""
    import os
    import shutil
    import subprocess

    exe = shutil.which("mineru-models-download")
    if not exe:
        print("[FAIL] 找不到 mineru-models-download，先执行：pip install mineru")
        return False

    env = dict(os.environ)
    env.setdefault("MINERU_MODEL_SOURCE", "modelscope")

    print(f"\n[下载] MinerU 模型（mode={mode}，源={env['MINERU_MODEL_SOURCE']}）")
    print("       这一步会拉版面/OCR/公式/表格等小模型，约 2.0 GB")
    result = subprocess.run(
        [exe, "-s", env["MINERU_MODEL_SOURCE"], "-m", mode],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        print(f"[FAIL] MinerU 模型下载返回 {result.returncode}")
        return False

    print("[OK]   MinerU 模型完成，配置已写入 ~/mineru.json")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 bge-m3 / bge-reranker-v2-m3 / MinerU 模型")
    parser.add_argument("--source", choices=["modelscope", "hf"], default="modelscope")
    parser.add_argument("--only", choices=list(MODELS), help="只下一个 bge 模型")
    parser.add_argument("--with-mineru", action="store_true", help="同时下载 MinerU 模型")
    parser.add_argument(
        "--mineru-mode",
        choices=["pipeline", "vlm", "all"],
        default="pipeline",
        help="MinerU 模型类型；无 GPU 机器不要用 vlm",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"下载源：{args.source}")
    print("=" * 60)

    failed = []
    for name, (repo_id, raw_path) in MODELS.items():
        if args.only and name != args.only:
            continue

        target = Path(raw_path)
        ok, detail = verify(target)
        if ok:
            print(f"[SKIP] {name} 已就绪（{detail}）")
            continue

        print(f"\n[下载] {name}  <- {repo_id}")
        print(f"       目标 {target}")
        try:
            if args.source == "modelscope":
                download_modelscope(repo_id, target)
            else:
                download_hf(repo_id, target)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name} 下载失败：{exc!r}")
            failed.append(name)
            continue

        ok, detail = verify(target)
        if ok:
            print(f"[OK]   {name} 完成（{detail}）")
        else:
            print(f"[FAIL] {name} 下载后校验不通过：{detail}")
            failed.append(name)

    if args.with_mineru and not args.only:
        if not download_mineru(args.mineru_mode):
            failed.append("mineru")

    print("\n" + "=" * 60)
    if failed:
        print(f"失败：{failed}")
        print("排查方向：网络能否访问下载源 / 磁盘剩余空间是否 > 8GB")
        return 1

    print("全部就绪。接着执行：python -m app.check_env")
    return 0


if __name__ == "__main__":
    sys.exit(main())
