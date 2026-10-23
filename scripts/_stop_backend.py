"""停掉所有 uvicorn app.main 进程，并确认端口真的释放。

为什么单独写成文件：用 `python -c` 时，命令行字符串里含 'uvicorn'，
psutil 匹配 cmdline 会把自己的进程也匹配上，脚本把自己杀掉（exit 15、无输出）。
"""

from __future__ import annotations

import os
import socket
import time

import psutil

TARGET_ARGS = ("uvicorn", "app.main")


def backend_pids() -> list[int]:
    me = os.getpid()
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] == me:
            continue
        try:
            argv = p.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if all(any(t in a for a in argv) for t in TARGET_ARGS):
            out.append(p.info["pid"])
    return out


def port_open(port: int = 8000) -> bool:
    s = socket.socket()
    s.settimeout(1)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def main() -> None:
    for attempt in range(3):
        pids = backend_pids()
        if not pids:
            break
        for pid in pids:
            try:
                psutil.Process(pid).kill()
                print(f"  killed {pid}")
            except psutil.NoSuchProcess:
                pass
        time.sleep(2)

    left = backend_pids()
    print("残留进程:", left or "无")
    print("8000 端口:", "仍被占用" if port_open() else "已释放")


if __name__ == "__main__":
    main()
