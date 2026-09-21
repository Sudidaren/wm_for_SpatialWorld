#!/usr/bin/env python3
"""并行分片上传到新卡。

单路 SFTP 只有 0.99 MB/s（延迟受限），4 路并发实测 2.95 MB/s。
做法：本地 split 成 N 片，N 个连接同时传，卡上 cat 还原。

用法：python3 upload_parallel.py <本地文件> [并发数]
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 56266
PW = "/dgVLk3vXhc1"
REMOTE_DIR = "/root/autodl-tmp"


def put_one(remote: str, local: str, errors: list) -> None:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(HOST, port=PORT, username="root", password=PW, timeout=30,
                  banner_timeout=60)
        s = c.open_sftp()
        s.put(local, remote)
        s.close()
        c.close()
    except Exception as exc:  # noqa: BLE001
        errors.append((local, repr(exc)))


def main() -> int:
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    base = os.path.basename(path)
    stage = os.path.dirname(os.path.abspath(path))
    size_mb = os.path.getsize(path) / 1e6

    # 先按 n 片切开（split -n 按字节均分，不用 -d 也能按 size 切）
    for f in os.listdir(stage):
        if f.startswith(base + ".part"):
            os.remove(os.path.join(stage, f))
    subprocess.run(["split", "-n", str(n), "-d", "-a", "2", path,
                    os.path.join(stage, base + ".part")], check=True)
    parts = sorted(f for f in os.listdir(stage) if f.startswith(base + ".part"))
    print(f"{base}: {size_mb:.0f} MB → {len(parts)} 片，{n} 路并发", flush=True)

    errors: list = []
    t = time.time()
    ths = [threading.Thread(target=put_one,
                            args=(f"{REMOTE_DIR}/{p}", os.path.join(stage, p), errors))
           for p in parts]
    [x.start() for x in ths]
    [x.join() for x in ths]
    dt = time.time() - t
    if errors:
        print("  !! 有分片失败:", errors[:2], flush=True)
        return 1
    # 卡上还原 + 删片
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PW, timeout=30,
              banner_timeout=60)
    _i, o, e = c.exec_command(
        f"cd {REMOTE_DIR} && cat {base}.part* > {base} && rm -f {base}.part* && "
        f"ls -l {base}", timeout=600)
    print("  卡上还原:", o.read().decode().strip(), e.read().decode()[:200], flush=True)
    c.close()
    print(f"  用时 {dt/60:.1f} 分钟，{size_mb*1000/dt:.0f} KB/s", flush=True)
    for p in parts:
        os.remove(os.path.join(stage, p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
