#!/usr/bin/env python3
"""Append the Gemini baseline vs WM+Gemini comparison to a log, on a timer.

Both arms are still growing (the WM arm is at a handful of tasks), so the
useful thing is not one number but the trajectory: the same paired statistic
recomputed as tasks land, on the tasks both arms have decided. That way the
question "does the world model help a strong model" is answered by a curve
instead of by whichever snapshot happened to be in front of us.

    python3 tools/watch_gemini_pair.py --interval 600

Log: /mnt/d/lightwm_out/gemini_pair.log
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path("/mnt/d/lightwm_out/gemini_pair.log")

ARMS = [
    # the closed model on its own (existing, 105-task frozen run)
    "T6base=local:/home/sudidaren/spatialworld_eval/runs/gemini31pro_ai2thor_procthor_438_frozen_v1/results.csv",
    # same model + WM, but with the spatial-memory channel off (the accidental
    # default of 2026-09-19; kept as the A5 row)
    "WMnohint=local:/home/sudidaren/spatialworld_eval/runs/wm_gemini31pro_da2_nohint_a311/results.csv",
    # the final WM: memory block on, state check on
    "WMhint=local:/home/sudidaren/spatialworld_eval/runs/wm_gemini31pro_da2_a311/results.csv",
]


def one_pass() -> str:
    cmd = [sys.executable, str(HERE / "compare_arms.py")]
    for a in ARMS:
        cmd += ["--arm", a]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    stamp = time.strftime("%F %T")
    body = p.stdout.strip() or p.stderr.strip()[:400]
    keep = [ln for ln in body.splitlines()
            if ln.strip().startswith(("T6base", "WMnohint", "WMhint", "shared", "  "))]
    keep = [ln for ln in keep if "rows=" in ln or "shared" in ln or "vs" in ln
            or "=" in ln]
    return f"[{stamp}]\n  " + "\n  ".join(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        block = one_pass()
        print(block, flush=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(block + "\n")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
