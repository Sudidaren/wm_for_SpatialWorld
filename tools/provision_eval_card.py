#!/usr/bin/env python3
"""Turn a bare AutoDL card into an evaluation box.

The evaluation harness has hard-coded paths (``/home/sudidaren/
spatialworld_eval``, ``/home/sudidaren/SpatialWorld/envs/ai2thor/.venv``), so a
card has to mirror the workstation layout instead of inventing its own.  This
script ships the three tarballs built by ``tools/make_eval_payload.sh`` --
code + perception weights, the evaluation harness, and the SpatialWorld task
data -- and then installs the python packages the harness imports.

GitHub is not reachable from the AutoDL boxes (only hf-mirror and the pypi
mirrors are), which is why the code travels as tarballs rather than a clone.

    python3 tools/provision_eval_card.py --card D --dry-run
    python3 tools/provision_eval_card.py --card D
    python3 tools/provision_eval_card.py --all
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import paramiko

PAYLOAD = Path("/mnt/d/eval_card_payload")
REMOTE_TMP = "/root/autodl-tmp/eval_payload"
LAYOUT = "/home/sudidaren"

#: the five cards the user opened.  Ports change on every boot; keep this table
#: in one place so a re-boot is a one-line edit.
CARDS = {
    "A": ("connect.weste.seetacloud.com", 15868, "Yl3WnjEDllk+"),
    "B": ("connect.westb.seetacloud.com", 52296, "hxeahvC88uMM"),
    "C": ("connect.cqa1.seetacloud.com", 47531, "0yy1HhF4KfYV"),
    "D": ("connect.weste.seetacloud.com", 32808, "kQLLhVht3BEB"),
    "E": ("connect.westb.seetacloud.com", 22416, "cZmriU8VEh+F"),
}

TARS = ("lightwm_phases.tar", "spatialworld_eval.tar", "spatialworld.tar")
#: code-only bundle (29 MB): the perception runtime, the world-model overlay
#: and the evaluation harness, without any weights
CODE_TAR = "code_only.tar"

#: what the harness + perception runtime import.  Installed into the image's
#: conda python, which is the one `run_ablation.sh` will use on the card.
PIP = ("ai2thor==5.0.0", "rfdetr", "supervision", "opencv-python-headless",
       "transformers", "openai", "python-dotenv", "pyyaml")


def sh(c, cmd: str, timeout: int = 3600) -> str:
    _, out, err = c.exec_command(cmd, timeout=timeout)
    text = out.read().decode(errors="replace")
    etext = err.read().decode(errors="replace")
    return (text + ("\n[stderr] " + etext if etext.strip() else "")).strip()


def provision(tag: str, dry: bool) -> int:
    host, port, pw = CARDS[tag]
    print(f"===== card {tag}  {host}:{port} =====", flush=True)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username="root", password=pw, timeout=25)
    try:
        state = sh(c, f"echo -n \"code=$(test -f {LAYOUT}/spatialworld_eval/eval_config.py "
                      f"&& test -f {LAYOUT}/lightwm_phases/phase_b/runtime_factory.py "
                      f"&& test -f {LAYOUT}/SpatialWorld/mllm_base_agent/agent/world_model.py "
                      f"&& echo 1 || echo 0) \"; "
                      f"echo -n \"weights=$(test -d {LAYOUT}/lightwm_phases/checkpoints/rfdetr_small_228094 "
                      f"&& test -d {LAYOUT}/lightwm_phases/checkpoints/small_objects_20260910 "
                      f"&& echo 1 || echo 0) \"; "
                      f"echo -n \"tasks=$(test -d {LAYOUT}/SpatialWorld/data/ai2thor/tasks "
                      f"&& echo 1 || echo 0) \"; "
                      f"echo -n \"da2=$(test -f {LAYOUT}/lightwm_phases/checkpoints/"
                      f"da2_metric_indoor_small/model.safetensors && echo 1 || echo 0)\"")
        print(f"  {state}", flush=True)
        if dry:
            print("  dry run", flush=True)
            return 0
        need = {}
        for kv in state.split():
            k, _, v = kv.partition("=")
            need[k] = (v == "1")
        sh(c, f"mkdir -p {REMOTE_TMP}")
        sftp = c.open_sftp()

        def push(name):
            local = PAYLOAD / name
            t0 = time.time()
            sftp.put(str(local), f"{REMOTE_TMP}/{name}")
            dt = time.time() - t0
            print(f"    {name}: {local.stat().st_size/1e6:.0f} MB in {dt/60:.1f} min",
                  flush=True)

        # code always (it is the thing that changes run to run)
        push(CODE_TAR)
        extract = [CODE_TAR]
        if not need["weights"]:
            push("lightwm_phases.tar")          # code + perception weights
            extract.append("lightwm_phases.tar")
        if not need["tasks"]:
            push("spatialworld.tar")            # task definitions + overlay
            extract.append("spatialworld.tar")
        if not need["code"]:
            push("spatialworld_eval.tar")       # the harness itself
            extract.append("spatialworld_eval.tar")
        sftp.close()
        print("  extracting into " + LAYOUT, flush=True)
        sh(c, f"mkdir -p {LAYOUT} && cd {LAYOUT} && "
              + " && ".join(f"tar xf {REMOTE_TMP}/{t}" for t in extract)
              + f" && rm -f {REMOTE_TMP}/*.tar && ls {LAYOUT}")
        # the DA2 weights are the one thing a card can fetch itself: the model
        # is public and hf-mirror is reachable from AutoDL (github is not)
        if not need["da2"]:
            print("  fetching DA2 from hf-mirror ...", flush=True)
            print(sh(c, f"mkdir -p {LAYOUT}/lightwm_phases/checkpoints/da2_metric_indoor_small && "
                        f"HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 "
                        f"/root/miniconda3/bin/python -c \""
                        f"from huggingface_hub import snapshot_download as d;"
                        f"import shutil,os;"
                        f"p=d('depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf');"
                        f"t='{LAYOUT}/lightwm_phases/checkpoints/da2_metric_indoor_small';"
                        f"[shutil.copy(os.path.join(p,f), os.path.join(t,f)) "
                        f"for f in ('config.json','model.safetensors','preprocessor_config.json')]\" "
                        f"2>&1 | tail -2; ls -la {LAYOUT}/lightwm_phases/checkpoints/"
                        f"da2_metric_indoor_small | tail -3", timeout=1800), flush=True)
        print("  pip install ...", flush=True)
        print(sh(c, f"/root/miniconda3/bin/pip install -q -i "
                    f"https://pypi.tuna.tsinghua.edu.cn/simple {' '.join(PIP)} "
                    f"2>&1 | tail -3", timeout=1800), flush=True)
        print("  verify:", flush=True)
        print(sh(c, "/root/miniconda3/bin/python -c \""
                    "import ai2thor, rfdetr, supervision, cv2, transformers, openai;"
                    "print('imports ok:', ai2thor.__version__)\" 2>&1 | tail -2"),
              flush=True)
        print(sh(c, f"ls {LAYOUT}/lightwm_phases/checkpoints/da2_metric_indoor_small/ "
                    f"2>/dev/null | tr '\\n' ' '; df -h /root/autodl-tmp | tail -1"),
              flush=True)
        return 0
    finally:
        c.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="", help="A|B|C|D|E")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tags = list(CARDS) if args.all else ([args.card] if args.card else [])
    if not tags:
        print("pick --card X or --all")
        return 2
    rc = 0
    for tag in tags:
        try:
            rc |= provision(tag, args.dry_run)
        except Exception as exc:
            print(f"card {tag}: FAILED {type(exc).__name__}: {exc}", flush=True)
            rc |= 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
