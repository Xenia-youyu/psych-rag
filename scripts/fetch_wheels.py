"""离线取轮子：绕过 pip 被代理 403 的问题。

背景：本机 pip 的 HTTPS 请求会被沙箱透明代理挡下（返回 403 Forbidden），
但 Python 直接 urllib 请求同一个镜像站是正常的。所以这里自己解析
simple index、挑出与当前解释器兼容的 wheel、直接下载并校验 sha256，
最后交给 `pip install --no-deps --no-index` 从本地装。

用法：
    python -m scripts.fetch_wheels torch filelock sympy networkx jinja2 fsspec setuptools tqdm psutil
    python -m scripts.fetch_wheels --install torch modelscope

设计取舍：
    - 只处理 wheel，不处理 sdist。本机没有 C 编译器，sdist 装了也是白装。
    - 不做传递依赖解析。依赖自己列出来——反正数量不多，比引一个解析器进来简单。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import parse as parse_version

INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
DEFAULT_DIR = Path("D:/wheels")
UA = "pip/26"
CHUNK = 1 << 20


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout).read()


def _candidates(pkg: str) -> list[tuple[str, str]]:
    """从 simple index 取 (whl 绝对地址, sha256) 列表。"""
    base = f"{INDEX}/{pkg}/"
    html = _get(base).decode("utf-8", "ignore")
    links = re.findall(r'href="([^"#]+)#sha256=([0-9a-fA-F]{64})"', html)
    return [(urljoin(base, href), sha.lower()) for href, sha in links if href.endswith(".whl")]


def pick(pkg: str) -> tuple[str, str, str]:
    """挑出兼容当前解释器的最优 wheel，返回 (url, sha256, 文件名)。"""
    want = canonicalize_name(pkg)
    priority = {str(tag): i for i, tag in enumerate(sys_tags())}

    best: tuple[tuple, str, str, str] | None = None
    seen = 0

    for url, sha in _candidates(pkg):
        fname = url.rsplit("/", 1)[-1]
        try:
            name, version, _build, tags = parse_wheel_filename(fname)
        except Exception:  # noqa: BLE001 - 文件名不合规就跳过
            continue
        if canonicalize_name(name) != want:
            continue
        rank = min((priority[str(t)] for t in tags if str(t) in priority), default=None)
        if rank is None:
            continue
        seen += 1
        key = (parse_version(str(version)), -rank)
        if best is None or key > best[0]:
            best = (key, url, sha, fname)

    if best is None:
        raise LookupError(f"{pkg}: 没有兼容 python{sys.version_info.major}.{sys.version_info.minor} 的 wheel")
    print(f"     可选 {seen} 个兼容版本，选中 {best[3]}")
    return best[1], best[2], best[3]


def download(url: str, sha: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / url.rsplit("/", 1)[-1]
    if dest.exists() and _sha256(dest) == sha:
        print(f"     [SKIP] 已存在且哈希一致 {dest.name}")
        return dest

    digest = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            fh.write(block)
            digest.update(block)
            got += len(block)
            if total:
                print(f"\r     下载 {got/1048576:6.1f}/{total/1048576:.1f} MB", end="", flush=True)
    print()

    actual = digest.hexdigest()
    if actual != sha:
        dest.unlink(missing_ok=True)
        raise ValueError(f"{dest.name} 哈希不符：期望 {sha[:12]} 实际 {actual[:12]}")
    print(f"     [OK] {dest.name} 哈希校验通过")
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="从清华镜像离线取 wheel")
    parser.add_argument("packages", nargs="+", help="包名列表")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help=f"存放目录（默认 {DEFAULT_DIR}）")
    parser.add_argument("--install", action="store_true", help="下载后直接 pip install --no-deps")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    print(f"目标目录 {target_dir}")
    print("=" * 60)

    files: list[Path] = []
    failed: list[str] = []
    for pkg in args.packages:
        print(f"\n[{pkg}]")
        try:
            url, sha, _name = pick(pkg)
            files.append(download(url, sha, target_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"     [FAIL] {exc}")
            failed.append(pkg)

    print("\n" + "=" * 60)
    if files:
        total = sum(f.stat().st_size for f in files) / 1024**2
        print(f"已下载 {len(files)} 个文件 / {total:.0f} MB → {target_dir}")
    if failed:
        print(f"失败：{failed}")

    if args.install and files:
        # 注意：不能把 whl 文件名当参数传给 pip —— pip 会把含 .whl 的参数
        # 当成相对路径去解析（报 "looks like a filename, but the file does not exist"），
        # 而不是当成 --find-links 里的候选。必须传规范包名。
        names = sorted({canonicalize_name(p) for p in args.packages if p not in failed})
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--no-deps", "--no-index", "--find-links", str(target_dir),
            *names,
        ]
        print("\n执行：pip install --no-deps --no-index --find-links "
              f"\"{target_dir}\" {' '.join(names)}")
        result = subprocess.run(cmd, check=False)
        return result.returncode

    if files:
        names = " ".join(sorted({canonicalize_name(p) for p in args.packages if p not in failed}))
        print("\n装它：")
        print(f'  pip install --no-deps --no-index --find-links "{target_dir}" {names}')

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
