"""找回数据服务器：扫网段 → 用项目凭据认领 → 可选改写 config.ini。

为什么需要这个脚本：
    共享实验环境是 DHCP 分配，机器一重启 IP 就换。本项目已经换了四次：
    .175 → .52 → .181 → .185 → .101 → .168。每次都要现写一遍扫描脚本，
    而且网段里跑着十几个同学的 MySQL+Milvus+Redis 三件套，**连错库不会报错**，
    只会得到「文档数不对、指标全 0」这类假问题。

    所以认领判据必须是数据，不是 IP：
    `psy_documents` 的行数 + 最新写入时间对得上，才算我们的。

用法：
    # 只看哪台是我们的
    .venv/Scripts/python.exe scripts/find_server.py

    # 确认后直接改 config.ini（会先备份）
    .venv/Scripts/python.exe scripts/find_server.py --apply --yes

    # 库比预期小很多时（比如刚建的库）调低门槛
    .venv/Scripts/python.exe scripts/find_server.py --expect-docs 1
"""

from __future__ import annotations

import argparse
import asyncio
import configparser
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pymysql

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.ini"


def local_ips() -> list[str]:
    out = subprocess.run(
        ["ipconfig"], capture_output=True, text=True, encoding="gbk", errors="ignore"
    ).stdout
    return re.findall(r"IPv4[^\:]*:\s*([0-9\.]+)", out)


async def port_open(host: str, port: int, timeout: float = 0.7) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        return True
    except Exception:
        return False


@dataclass
class Candidate:
    host: str
    docs: int
    chunks: int
    last_write: str


async def scan(subnet: str, mysql_user: str, mysql_pw: str, database: str) -> list[Candidate]:
    hosts = [f"{subnet}.{i}" for i in range(1, 255)]
    sem = asyncio.Semaphore(220)

    async def open_both(h: str) -> bool:
        async with sem:
            return await port_open(h, 3306) and await port_open(h, 19530)

    alive = [h for h, ok in zip(hosts, await asyncio.gather(*(open_both(h) for h in hosts))) if ok]
    print(f"同开 3306+19530 的主机：{len(alive)} 台 → {', '.join(sorted(alive)) or '无'}")

    found: list[Candidate] = []
    for host in sorted(alive, key=lambda x: int(x.split(".")[-1])):
        try:
            conn = pymysql.connect(
                host=host, port=3306, user=mysql_user, password=mysql_pw,
                database=database, connect_timeout=3, read_timeout=4,
            )
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM psy_documents")
                docs = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM psy_chunks")
                chunks = int(cur.fetchone()[0])
                cur.execute("SELECT MAX(created_at) FROM psy_documents")
                last = cur.fetchone()[0]
            conn.close()
            found.append(Candidate(host, docs, chunks, str(last)))
        except Exception:
            pass
    return found


def pick(best: Candidate) -> None:
    cp = configparser.ConfigParser()
    cp.read(CONFIG, encoding="utf-8")
    backup = CONFIG.with_name(f"config.ini.bak-{datetime.now():%Y%m%d-%H%M}")
    shutil.copy2(CONFIG, backup)
    for section in ("mysql", "redis", "milvus"):
        if cp.has_section(section) and cp.has_option(section, "host"):
            cp.set(section, "host", best.host)
    with CONFIG.open("w", encoding="utf-8") as fh:
        cp.write(fh)
    print(f"\n已写入 config.ini（mysql/redis/milvus → {best.host}），备份 {backup.name}")
    print("接着跑：PYTHONPATH=. .venv/Scripts/python.exe app/check_env.py")


def main() -> int:
    ap = argparse.ArgumentParser(description="找回数据服务器并可选改写配置")
    ap.add_argument("--subnet", default="", help="网段前缀，默认取本机 192.168.x")
    ap.add_argument("--expect-docs", type=int, default=130, help="判定为本项目的最小文档数")
    ap.add_argument("--apply", action="store_true", help="改 config.ini")
    ap.add_argument("--yes", action="store_true", help="跳过确认")
    args = ap.parse_args()

    cp = configparser.ConfigParser()
    cp.read(CONFIG, encoding="utf-8")
    user = cp.get("mysql", "user")
    pw = cp.get("mysql", "password")
    db = cp.get("mysql", "database")
    print(f"当前配置指向：{cp.get('mysql', 'host')}（{user}@{db}）")

    subnet = args.subnet
    if not subnet:
        candidates = [ip for ip in local_ips() if ip.startswith("192.168.")]
        if not candidates:
            print("本机没有 192.168 网段地址，用 --subnet 指定")
            return 1
        subnet = ".".join(candidates[0].split(".")[:3])
    print(f"本机地址：{local_ips()}  扫描网段：{subnet}.1-254")

    found = asyncio.run(scan(subnet, user, pw, db))
    if not found:
        print("\n没有能用项目凭据登录的库。可能是：服务器关机 / 密码变了 / 数据被迁走")
        return 1

    print("\n能用项目凭据登录的库：")
    for c in found:
        mark = "  <== 本项目" if c.docs >= args.expect_docs else ""
        print(f"  {c.host:16} 文档 {c.docs:5d}  块 {c.chunks:6d}  最新写入 {c.last_write}{mark}")

    hits = [c for c in found if c.docs >= args.expect_docs]
    if not hits:
        print(f"\n没有文档数 ≥ {args.expect_docs} 的库，可调低 --expect-docs 再试")
        return 2
    best = max(hits, key=lambda c: (c.docs, c.last_write))
    if len(hits) > 1:
        print(f"\n有 {len(hits)} 台都过门槛，取文档数最多的 {best.host}——请人工确认后再 --apply")
        return 3

    print(f"\n认领成功：{best.host}（{best.docs} 文档 / {best.chunks} 块 / 最新 {best.last_write}）")
    if not args.apply:
        print("只报告不改写。确认后加 --apply")
        return 0
    if not args.yes:
        print("改写 config.ini 需要 --yes")
        return 0
    pick(best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
