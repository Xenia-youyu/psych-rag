"""确认后端只有一个实例：以「谁在监听 8000」为唯一判据。

为什么不用 `python -c` 数进程：命令字符串里带 'uvicorn'/'app.main'，
psutil 按 cmdline 匹配会把自己也算进去，每次都多出 2 个假进程（venv 启动器 + 真解释器）。
判断实例数只看监听端口，一个端口只能被一个进程绑定，这是最硬的判据。
"""

from __future__ import annotations

import json
import time

import psutil
import urllib.request


def listeners(port: int = 8000) -> list[int]:
    pids = []
    for c in psutil.net_connections(kind="tcp"):
        if c.laddr.port == port and c.status == psutil.CONN_LISTEN:
            pids.append(c.pid)
    return sorted(set(p for p in pids if p))


def main() -> None:
    pids = listeners()
    print(f"监听 8000 的进程: {pids or '无'}")

    for pid in pids:
        p = psutil.Process(pid)
        started = time.strftime("%H:%M:%S", time.localtime(p.create_time()))
        try:
            parent = psutil.Process(p.ppid())
            pname = f"{parent.name()}({parent.pid})"
        except psutil.NoSuchProcess:
            pname = "?"
        print(f"  pid={pid} {p.name()} 启动={started} 父进程={pname}")

    if len(pids) == 1:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=8) as r:
            print("健康:", json.loads(r.read()))
    else:
        print("!! 实例数不是 1，先清理")


if __name__ == "__main__":
    main()
