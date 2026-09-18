#!/usr/bin/env python3
"""Drive the v3 depth-head retrain on a cloud box from here.

The box cannot be powered on from inside the container, so this script does
everything *around* that: it checks the link, uploads the pool the box is
missing, syncs the code, starts training and (optionally) pulls the artefacts
back.  Run one step at a time:

    python3 tools/run_cloud_depth_v3.py check           # what is on the box?
    python3 tools/run_cloud_depth_v3.py upload-pool     # classic homes, 2.5 GB
    python3 tools/run_cloud_depth_v3.py sync-code       # repo + splits
    python3 tools/run_cloud_depth_v3.py train           # index + split + v3
    python3 tools/run_cloud_depth_v3.py watch           # tail the log / report
    python3 tools/run_cloud_depth_v3.py pull            # weights -> D:

Everything large stays on the box's /root/autodl-tmp or on the local D: drive.

The box is whatever you point it at -- AutoDL reassigns the ssh port on every
boot and the three instances have different contents, so the target comes from
the command line (or from LIGHTWM_CLOUD_HOST / _PORT / _PASSWORD):

    python3 tools/run_cloud_depth_v3.py check \
        --host connect.westd.seetacloud.com --port 28564 --password ...

``check`` prints exactly what a fresh box is missing (GPU visibility, torch,
each data pool, the DINOv2 cache, the repo), so switching cards is a matter of
running it once and filling the gaps it names.
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

#: pool name -> local directory; the box needs all four to reproduce the
#: training index (the classic family is the one that collapsed the scale)
POOLS = {
    "cov": Path("/mnt/d/lightwm_data_cov"),
    "cov2": Path("/mnt/d/lightwm_data_cov2"),
    "procthor2": Path("/mnt/d/lightwm_data_procthor2"),
    "valhouses": Path("/mnt/d/lightwm_data_valhouses"),
}

#: files the box needs to run v3; the pools and the HF cache are already there
CODE_PATHS = ("phase_b", "shared", "tools", "data/splits_noneval.json",
              "data/eval_rooms.json")


def connect(host: str, port: int, password: str, timeout: int = 30):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username="root", password=password,
              timeout=timeout)
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


def step_check(c) -> int:
    """Say exactly what this box has and what it is missing."""
    print("host:", run(c, "hostname"))
    gpu = run(c, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader "
                 "2>&1 | head -2")
    devs = run(c, "ls /dev/nvidia* 2>/dev/null | head -3")
    print("gpu :", gpu.replace("\n", " | "), "| /dev/nvidia*:",
          devs.replace("\n", " ") or "(none)")
    if "Permission denied" in gpu or not devs:
        print("      !! no usable GPU device in this container -- training "
              "cannot run here (AutoDL's no-GPU mode, or the card is busy)")
    pyth = run(c, "for p in python3 /root/miniconda3/bin/python "
                  "/root/miniconda3/envs/*/bin/python; do [ -x $p ] && "
                  "echo -n \"$p=$($p -c 'import torch;print(torch.__version__,"
                  "torch.cuda.is_available())' 2>&1|tail -1) ; \"; done")
    print("torch:", pyth)
    for name in POOLS:
        n = run(c, f"ls {DATA}/lightwm_data_{name}/episodes 2>/dev/null | wc -l").strip()
        ok = "OK " if n.isdigit() and int(n) > 0 else "缺失"
        print(f"pool : lightwm_data_{name:10s} {ok} episodes={n}")
    print("hf   :", run(c, "du -sh /root/.cache/huggingface/hub 2>/dev/null "
                           "|| echo 缺失").strip())
    print("code :", run(c, f"ls -d {REMOTE_WM} 2>/dev/null || echo 不存在").strip())
    print("disk :", run(c, f"df -h {DATA} | tail -1").strip())
    return 0


def step_upload_pool(c, names=None) -> int:
    for name in (names or ["cov"]):
        src = POOLS[name]
        remote = f"{DATA}/lightwm_data_{name}"
        have = run(c, f"ls {remote}/episodes 2>/dev/null | wc -l").strip()
        if have.isdigit() and int(have) > 0:
            print(f"[pool] {name}: already on the box ({have} episodes)")
            continue
        tar = Path(f"/mnt/d/lightwm_{name}_upload.tar")
        print(f"[pool] {name}: packing {src} ...")
        with tarfile.open(tar, "w") as tf:
            tf.add(src, arcname=f"lightwm_data_{name}", recursive=True)
        size = tar.stat().st_size
        print(f"[pool] {name}: packed {size/1e9:.2f} GB; uploading ...")
        sftp = c.open_sftp()
        t0 = time.time()
        sftp.put(str(tar), f"{DATA}/lightwm_upload.tar")
        dt = time.time() - t0
        sftp.close()
        print(f"[pool] {name}: {dt:.0f}s ({size/1e6/max(1, dt):.1f} MB/s)")
        print("[pool] extracted:", run(
            c, f"cd {DATA} && tar xf lightwm_upload.tar && "
               f"rm -f lightwm_upload.tar && "
               f"ls {remote}/episodes | wc -l", timeout=3600))
        tar.unlink(missing_ok=True)
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


STEPS = {"status": step_status, "check": step_check,
         "upload-pool": step_upload_pool, "sync-code": step_sync_code,
         "train": step_train, "watch": step_watch, "pull": step_pull}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=sorted(STEPS))
    ap.add_argument("--host", default=os.environ.get("LIGHTWM_CLOUD_HOST", HOST))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("LIGHTWM_CLOUD_PORT", PORT)))
    ap.add_argument("--password",
                    default=os.environ.get("LIGHTWM_CLOUD_PASSWORD", PW))
    ap.add_argument("--pools", default="cov",
                    help="comma separated, for upload-pool: "
                         + ",".join(POOLS))
    args = ap.parse_args()
    try:
        c = connect(args.host, args.port, args.password)
    except Exception as exc:
        print(f"cannot reach {args.host}:{args.port} ({exc}).\n"
              "The AutoDL container cannot power itself on -- ask the user to "
              "start it in the console; the ssh port changes on every boot, so "
              "pass --port (and --host) for a card other than weste.")
        return 2
    try:
        if args.step == "upload-pool":
            return step_upload_pool(c, [p.strip() for p in args.pools.split(",") if p.strip()])
        return STEPS[args.step](c)
    finally:
        c.close()


if __name__ == "__main__":
    raise SystemExit(main())
