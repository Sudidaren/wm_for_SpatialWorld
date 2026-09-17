#!/usr/bin/env python3
"""Stream one tarball to the training box and unpack it there."""

from __future__ import annotations

import argparse
import os
import time

import paramiko

HOST, PORT, PW = "connect.weste.seetacloud.com", 32808, "kQLLhVht3BEB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--extract-to", default="/root/autodl-tmp")
    args = ap.parse_args()

    size = os.path.getsize(args.tar)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PW, timeout=30)
    sftp = c.open_sftp()
    t0 = time.time()
    print(f"[push] {size/1e9:.2f} GB -> {args.remote}", flush=True)
    sftp.put(args.tar, args.remote)
    dt = time.time() - t0
    print(f"[push] done in {dt:.0f}s ({size/1e6/max(1,dt):.1f} MB/s)", flush=True)
    sftp.close()
    _, o, _ = c.exec_command(
        f"cd {args.extract_to} && tar xf {args.remote} && rm -f {args.remote} && "
        f"du -sh lightwm_data_cov2 lightwm_data_procthor2 2>/dev/null", timeout=1800)
    print("[push] extract:", o.read().decode().strip().replace("\n", " | "), flush=True)
    c.close()
    os.remove(args.tar)
    print("[push] local tarball removed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
