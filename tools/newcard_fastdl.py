#!/usr/bin/env python3
"""在评测卡上并行下载 ModelScope 模型（裸 curl，比 modelscope 自带下载器快 5 倍）。

为什么不用 modelscope.snapshot_download：实测它单文件 2.65 MB/s，而同样的
文件用 curl 单连接 11.9 MB/s、8 并行 14 MB/s。差在它的分块/流式实现上。

用法（在卡上跑）：
    /root/miniconda3/bin/python fastdl.py <repo_id> <dest_dir> [并发数]
例如：
    ... Qwen/Qwen3-VL-30B-A3B-Instruct /root/autodl-tmp/models/qwen3vl-30b-bf16 6

特性：
  * 先拉文件清单（ModelScope API），按大小从大到小排队；
  * 每个文件一个 curl，`-C -` 断点续传，`--retry` 兜网络抖动；
  * 已下完（大小一致）的文件直接跳过 —— 中断后重跑即可续；
  * 每 20 秒打一行总进度与瞬时速率，可直接 tail 日志。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request


def file_list(repo: str):
    api = (f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files"
           "?Revision=master&Recursive=true")
    with urllib.request.urlopen(api, timeout=60) as r:
        data = json.load(r)
    out = []
    for f in data["Data"]["Files"]:
        path = f["Path"]
        if f.get("Type") != "blob" or path.startswith("."):
            continue
        out.append((path, int(f["Size"])))
    out.sort(key=lambda x: -x[1])
    return out


def main() -> int:
    repo, dest = sys.argv[1], sys.argv[2]
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    os.makedirs(dest, exist_ok=True)
    files = file_list(repo)
    total = sum(s for _, s in files)
    print(f'[fastdl] {repo} -> {dest}', flush=True)
    print(f'[fastdl] {len(files)} 个文件，共 {total / 1e9:.2f} GB，并发 {jobs}',
          flush=True)
    base = f'https://www.modelscope.cn/models/{repo}/resolve/master/'

    def have(path, size):
        p = os.path.join(dest, path)
        return os.path.exists(p) and os.path.getsize(p) == size

    done_bytes = sum(s for p, s in files if have(p, s))
    pending = [(p, s) for p, s in files if not have(p, s)]
    print(f'[fastdl] 已完成 {done_bytes / 1e9:.2f} GB，待下 '
          f'{sum(s for _, s in pending) / 1e9:.2f} GB', flush=True)

    running = {}
    t0 = time.time()
    last = t0
    while pending or running:
        while pending and len(running) < jobs:
            path, size = pending.pop(0)
            dst = os.path.join(dest, path)
            os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
            proc = subprocess.Popen(
                ["curl", "-sL", "-C", "-", "--retry", "5", "--retry-delay", "3",
                 "--connect-timeout", "30", "-o", dst, base + path])
            running[proc] = (path, size)
        for proc in list(running):
            if proc.poll() is None:
                continue
            path, size = running.pop(proc)
            ok = have(path, size)
            print(f'[fastdl] {"OK " if ok else "FAIL"} {path} '
                  f'({size / 1e9:.2f} GB)', flush=True)
            if not ok:
                pending.append((path, size))       # 没下全就重排
        now = time.time()
        if now - last >= 20:
            cur = sum(os.path.getsize(os.path.join(dest, p))
                      for p, s in files if os.path.exists(os.path.join(dest, p)))
            rate = (cur - done_bytes) / max(now - last, 1) / 1e6
            eta = (total - cur) / max(rate, 0.01) / 60
            print(f'[fastdl] {cur / 1e9:.2f}/{total / 1e9:.2f} GB  '
                  f'{rate:.1f} MB/s  剩余约 {eta:.0f} 分钟', flush=True)
            last = now
        time.sleep(1)

    bad = [p for p, s in files if not have(p, s)]
    cur = sum(os.path.getsize(os.path.join(dest, p))
              for p, s in files if os.path.exists(os.path.join(dest, p)))
    print(f'[fastdl] 结束：{cur / 1e9:.2f}/{total / 1e9:.2f} GB，'
          f'用时 {(time.time() - t0) / 60:.1f} 分钟，缺 {len(bad)} 个文件',
          flush=True)
    for p in bad:
        print(f'[fastdl]   缺: {p}', flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
