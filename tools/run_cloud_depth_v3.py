#!/usr/bin/env python3
"""Drive the v3 depth-head retrain on the cloud box (weste) from here.

The box cannot be powered on from inside the container, so this script does
everything *around* that: it checks the link, uploads the pool the box is
missing, syncs the code, starts training and (optionally) pulls the artefacts
back.  Run one step at a time:

    python3 tools/run_cloud_depth_v3.py status          # is the card up?
    python3 tools/run_cloud_depth_v3.py upload-pool     # classic homes, 2.5 GB
    python3 tools/run_cloud_depth_v3.py sync-code       # repo + splits
    python3 tools/run_cloud_depth_v3.py train           # index + split + v3
    python3 tools/run_cloud_depth_v3.py watch           # tail the log / report
    python3 tools/run_cloud_depth_v3.py pull            # weights -> D:

Everything large stays on the box's /root/autodl-tmp or on the local D: drive.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST, PORT, PW = "connect.weste.seetacloud.com", 32808, "kQLLhVht3BEB"
DATA = "/root/autodl-tmp"
REMOTE_WM = "/root/lightwm_phases"
LOCAL_POOL = Path("/mnt/d/lightwm_data_cov")
TARBALL = Path("/mnt/d/lightwm_cov_upload.tar")
PULL_DIR = Path("/mnt/d/lightwm_out/depth_v3")

#: files the box needs to run v3; the pools and the HF cache are already there
CODE_PATHS = ("phase_b", "shared", "tools", "data/splits_noneval.json",
              "data/eval_rooms.json")


def connect(timeout: int = 30):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PW, timeout=timeout)
    return c


def run(c, cmd: str, timeout: int = 3600) -> str:
    _, out, err = c.exec_command(cmd, timeout=timeout)
    text = out.read().decode(errors="replace")
    etext = err.read().decode(errors="replace")
    return (text + ("\n[stderr] " + etext if etext.strip() else "")).strip()


def step_status(c) -> int:
    print(run(c, "hostname; nvidia-smi --query-gpu=name,memory.used "
                 "--format=csv,noheader 2>/dev/null | head -1; "
                 f"ls -d {DATA}/* 2>/dev/null | head -12; "
                 f"ls {DATA}/lightwm_data_cov/episodes 2>/dev/null | wc -l"))
    return 0


def step_upload_pool(c) -> int:
    have = run(c, f"ls {DATA}/lightwm_data_cov/episodes 2>/dev/null | wc -l").strip()
    if have.isdigit() and int(have) > 0:
        print(f"[pool] already on the box ({have} episodes); nothing to do")
        return 0
    print(f"[pool] packing {LOCAL_POOL} ...")
    with tarfile.open(TARBALL, "w") as tf:
        tf.add(LOCAL_POOL, arcname="lightwm_data_cov", recursive=True)
    size = TARBALL.stat().st_size
    print(f"[pool] packed {size/1e9:.2f} GB; uploading as one stream ...")
    sftp = c.open_sftp()
    t0 = time.time()
    sftp.put(str(TARBALL), f"{DATA}/lightwm_cov_upload.tar")
    dt = time.time() - t0
    sftp.close()
    print(f"[pool] uploaded in {dt:.0f}s ({size/1e6/max(1, dt):.1f} MB/s)")
    print("[pool] extracting:", run(
        c, f"cd {DATA} && tar xf lightwm_cov_upload.tar && "
           f"rm -f lightwm_cov_upload.tar && "
           f"ls lightwm_data_cov/episodes | wc -l", timeout=1800))
    TARBALL.unlink(missing_ok=True)
    print("[pool] local tarball removed")
    return 0


def step_sync_code(c) -> int:
    local_tar = Path("/tmp/lightwm_code.tar")
    with tarfile.open(local_tar, "w") as tf:
        for rel in CODE_PATHS:
            tf.add(ROOT / rel, arcname=rel, recursive=True)
    sftp = c.open_sftp()
    sftp.put(str(local_tar), "/root/lightwm_code.tar")
    sftp.close()
    local_tar.unlink(missing_ok=True)
    print(run(c, f"mkdir -p {REMOTE_WM} && cd {REMOTE_WM} && "
                 f"tar xf /root/lightwm_code.tar && rm -f /root/lightwm_code.tar && "
                 f"ls phase_b/train_depth_v2.py tools/cloud_depth_v3.sh"))
    # the checkpoint the training initialises from must exist on the box
    print("[code] init checkpoint:", run(
        c, f"ls -la {REMOTE_WM}/checkpoints/small_objects_20260910/"
           f"dense_depth_best.pt 2>&1 | tail -1"))
    return 0


def step_train(c) -> int:
    out = run(c, f"cd {REMOTE_WM} && LIGHTWM_ROOT={REMOTE_WM} "
                 f"bash tools/cloud_depth_v3.sh", timeout=1800)
    print(out)
    return 0


def step_watch(c) -> int:
    print(run(c, "tail -n 12 /root/train_depth_v3.log; echo; "
                 "grep -c '^  ep' /root/train_depth_v3.log"))
    print("\n[done?] the training writes its checkpoint at the end of each "
          "epoch that improves validation:")
    print(run(c, f"ls -la {DATA}/depth_v3/ 2>/dev/null"))
    return 0


def step_pull(c) -> int:
    PULL_DIR.mkdir(parents=True, exist_ok=True)
    sftp = c.open_sftp()
    for name in ("dense_depth_v2_best.pt", "dense_depth_v2_best.pt.json",
                 "train_meta.json", "val_split.json"):
        remote = f"{DATA}/depth_v3/{name}"
        try:
            sftp.get(remote, str(PULL_DIR / name))
            print(f"[pull] {name}")
        except FileNotFoundError:
            print(f"[pull] missing on the box: {name}")
    sftp.close()
    print(f"[pull] -> {PULL_DIR}")
    return 0


STEPS = {"status": step_status, "upload-pool": step_upload_pool,
         "sync-code": step_sync_code, "train": step_train,
         "watch": step_watch, "pull": step_pull}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=sorted(STEPS))
    args = ap.parse_args()
    try:
        c = connect()
    except Exception as exc:
        print(f"cannot reach {HOST}:{PORT} ({exc}).\n"
              "The AutoDL container cannot power itself on -- ask the user to "
              "start weste in the console (and check the port, it moves).")
        return 2
    try:
        return STEPS[args.step](c)
    finally:
        c.close()


if __name__ == "__main__":
    raise SystemExit(main())
