"""停掉正在运行的 ingest_plan，并确认进程真的没了。

为什么落成文件：命令行里含 'scripts.ingest_plan' 字样，用 `python -c` 时
psutil 按 cmdline 匹配会把自己也算进去，脚本把自己杀掉（exit 15、无输出）。
"""

from __future__ import annotations

import os
import time

import psutil

TARGET = "scripts.ingest_plan"


def pids() -> list[int]:
    me = os.getpid()
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] == me:
            continue
        try:
            argv = p.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        # 只认「-m 后面的模块名」这一项，避免匹配到命令行里出现同名字符串的别的进程
        if any(a.strip() == TARGET for a in argv):
            out.append(p.info["pid"])
    return out


def main() -> None:
    for _ in range(3):
        found = pids()
        if not found:
            break
        for pid in found:
            try:
                psutil.Process(pid).kill()
                print(f"  killed {pid}")
            except psutil.NoSuchProcess:
                pass
        time.sleep(2)
    print("残留:", pids() or "无")


if __name__ == "__main__":
    main()
