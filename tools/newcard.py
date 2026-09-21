#!/usr/bin/env python3
"""新开的评测卡（westd）上的最小操作工具：跑命令 / 传文件 / 拉文件。

凭据和 card_ssh.py 里的老四张卡放一起管理，但单独一份，避免动到正在用的
那套（老卡的 tag 被 card_ctl / c2c / monitor 引用着）。

    python3 tools/newcard.py run "nvidia-smi"
    python3 tools/newcard.py put <本地> <远端>
    python3 tools/newcard.py get <远端> <本地>
    python3 tools/newcard.py watch "tail -f /root/x.log"     # 流式看输出
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import paramiko

#: 2026-09-20 新开的三张卡：一条线一个模型（30B / 8B / Kimi）。
#: 老的五张（A-E）在 card_ssh.py 里，tag 被 card_ctl / c2c / monitor 引用着，
#: 不要混。
CARDS = {
    "30b": ("connect.westd.seetacloud.com", 25713, "ZkYWHfECEihE"),
    "8b": ("connect.westd.seetacloud.com", 33177, "tFvwsBM9d5a3"),
    "kimi": ("connect.westc.seetacloud.com", 47454, "l35oYyWKBqhy"),
    # 2026-09-20 晚：12GB 卡，专门跑 GPT-5 + WM（GPT-5 是 API 模型，
    # 卡上只跑 WM 的感知头 + Unity，所以不需要大显存，也不需要 vLLM）
    "gpt5": ("connect.westb.seetacloud.com", 35814, "NMdugxH13WDd"),
}
DEFAULT_CARD = os.environ.get("NEWCARD", "30b")
HOST, PORT, PASSWORD = CARDS[DEFAULT_CARD]
USER = "root"


def connect(retries: int = 5, timeout: int = 30) -> paramiko.SSHClient:
    last = None
    for i in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                      timeout=timeout, banner_timeout=60, auth_timeout=60)
            return c
        except Exception as exc:                       # noqa: BLE001
            last = exc
            print(f"[newcard] 连接失败 {i + 1}/{retries}: {exc}", file=sys.stderr)
            time.sleep(3)
    raise SystemExit(f"连不上 {HOST}:{PORT}: {last}")


def run(cmd: str, stream: bool = False) -> int:
    c = connect()
    try:
        _in, out, err = c.exec_command(cmd, get_pty=stream)
        if stream:
            chan = out.channel
            while True:
                if chan.recv_ready():
                    sys.stdout.write(chan.recv(4096).decode("utf-8", "replace"))
                    sys.stdout.flush()
                elif chan.exit_status_ready() and not chan.recv_ready():
                    break
                else:
                    time.sleep(0.2)
            return chan.recv_exit_status()
        sys.stdout.write(out.read().decode("utf-8", "replace"))
        msg = err.read().decode("utf-8", "replace")
        if msg.strip():
            sys.stderr.write(msg)
        return out.channel.recv_exit_status()
    finally:
        c.close()


def put(local: str, remote: str) -> int:
    c = connect(timeout=60)
    try:
        sftp = c.open_sftp()
        size = os.path.getsize(local)
        t0 = time.time()

        def cb(done, total):
            pct = done * 100.0 / max(total, 1)
            if int(pct) % 5 == 0:
                rate = done / max(time.time() - t0, 1e-6) / 1e6
                sys.stdout.write(f"\r  {pct:5.1f}%  {done/1e9:.2f}/{total/1e9:.2f}GB"
                                 f"  {rate:.0f}MB/s   ")
                sys.stdout.flush()

        sys.stdout.write(f"上传 {local} -> {remote}（{size/1e9:.2f}GB）\n")
        sftp.put(local, remote, callback=cb)
        print(f"\n  完成，用时 {time.time() - t0:.0f}s")
        sftp.close()
        return 0
    finally:
        c.close()


def get(remote: str, local: str) -> int:
    c = connect(timeout=60)
    try:
        sftp = c.open_sftp()
        t0 = time.time()
        sftp.get(remote, local)
        print(f"下载 {remote} -> {local}，用时 {time.time() - t0:.0f}s")
        sftp.close()
        return 0
    finally:
        c.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "watch", "put", "get"])
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--card", default=None, choices=sorted(CARDS))
    args = ap.parse_args()
    if args.card:
        global HOST, PORT, PASSWORD
        HOST, PORT, PASSWORD = CARDS[args.card]
    if args.action == "run":
        return run(args.a)
    if args.action == "watch":
        return run(args.a, stream=True)
    if args.action == "put":
        return put(args.a, args.b)
    return get(args.a, args.b)


if __name__ == "__main__":
    raise SystemExit(main())
