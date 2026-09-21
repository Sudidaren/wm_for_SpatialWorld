#!/usr/bin/env python3
"""把云上各卡的结果拉回本地（只拉小的记账文件，不拉成千上万张 png）。

每张卡上：把 run 目录里的 results.csv / summary.json / summary_by_env_category.csv
/ plan.json / state.json 打成一个 tar.gz，再 get 回来解到

    /mnt/d/lightwm_out/cloud_pull_<日期>/<卡tag>_<run>/

    python3 tools/pull_cloud_results.py            # 拉全部四张
    python3 tools/pull_cloud_results.py 30b kimi   # 只拉指定卡
"""

from __future__ import annotations

import os
import sys
import tarfile
import time

import paramiko

from newcard import CARDS

OUT_ROOT = f"/mnt/d/lightwm_out/cloud_pull_{time.strftime('%Y%m%d')}"
ARTIFACTS = ("results.csv", "summary.json", "summary_by_env_category.csv",
             "plan.json", "state.json")


def sh(host: str, port: int, pwd: str, cmd: str, timeout: int = 300) -> str:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username="root", password=pwd, timeout=30,
              banner_timeout=60, auth_timeout=60)
    try:
        _in, out, err = c.exec_command(cmd, timeout=timeout)
        so = out.read().decode("utf-8", "replace")
        se = err.read().decode("utf-8", "replace")
        return so + (("\n[stderr] " + se) if se.strip() else "")
    finally:
        c.close()


def pull_tag(tag: str) -> None:
    host, port, pwd = CARDS[tag]
    run = sh(host, port, pwd,
             "ls -dt /home/sudidaren/spatialworld_eval/runs/wm_* | head -1").strip()
    if not run:
        print(f"{tag}: 找不到 run 目录")
        return
    name = os.path.basename(run)
    files = " ".join(f'"{f}"' for f in ARTIFACTS)
    tar = f"/root/autodl-tmp/pull_{tag}.tgz"
    sh(host, port, pwd, f'cd "{run}" && tar czf {tar} {files} 2>/dev/null; '
                        f'ls -la {tar}')
    local_dir = os.path.join(OUT_ROOT, f"{tag}_{name}")
    os.makedirs(local_dir, exist_ok=True)
    local_tar = os.path.join(local_dir, f"{tag}.tgz")
    transport = paramiko.Transport((host, port))
    transport.connect(username="root", password=pwd)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.get(tar, local_tar)
    sftp.close()
    transport.close()
    with tarfile.open(local_tar) as tf:
        tf.extractall(local_dir)
    os.remove(local_tar)
    print(f"{tag}: {run} -> {local_dir}")


def main() -> None:
    tags = sys.argv[1:] or list(CARDS)
    os.makedirs(OUT_ROOT, exist_ok=True)
    for tag in tags:
        if tag not in CARDS:
            print(f"跳过未知卡 {tag}")
            continue
        try:
            pull_tag(tag)
        except Exception as exc:                       # noqa: BLE001
            print(f"{tag} 拉取失败: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
