#!/usr/bin/env python3
"""Track every AI2-THOR arm and report pairings once they have enough tasks.

The question the main table has to answer is a paired one, so this prints:
  * each arm's decided count and success count as it grows,
  * every pair that both arms have decided, with McNemar and a bootstrap CI,
  * a marker when an arm crosses --checkpoint tasks, so a "does it help yet"
    answer does not require scrolling.

    python3 tools/watch_main_table.py --interval 600 --checkpoint 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path("/mnt/d/lightwm_out/main_table.log")

# name -> source:path  (source is a card letter or "local")
ARMS = {
    "T1-base30": "C:/root/autodl-tmp/runs_local/base_q30_a311/results.csv",
    "T2-nohint30": "D:/root/autodl-tmp/runs_local/wm_q30_da2_a311/results.csv",
    "T2-final30": "B:/root/autodl-tmp/runs_local/wm_q30_da2_final_a311/results.csv",
    "A2-noinject": "D:/root/autodl-tmp/runs_local/wm_q30_da2_a2_noinject/results.csv",
    "T5-sameinfo": "C:/root/autodl-tmp/runs_local/base_q30_sameinfo_a311/results.csv",
    "A1-final": "B:/root/autodl-tmp/runs_local/wm_q30_da2_a1_final_a311/results.csv",
    "T3-oldhead": "D:/root/autodl-tmp/runs_local/wm_q30_headv2_final_a311/results.csv",
    "T4-v3head": "B:/root/autodl-tmp/runs_local/wm_q30_headv3_final_a311/results.csv",
    "A4-notri": "D:/root/autodl-tmp/runs_local/wm_q30_da2_a4_final_a311/results.csv",
    "WM+Gem-src": "local:/home/sudidaren/spatialworld_eval/runs/wm_gemini31pro_da2_a311/results.csv",
    "WM+Gem-nohint": "local:/home/sudidaren/spatialworld_eval/runs/wm_gemini31pro_da2_nohint_a311/results.csv",
    "Gemini-base": "local:/home/sudidaren/spatialworld_eval/runs/gemini31pro_ai2thor_procthor_438_frozen_v1/results.csv",
}


def one_pass(checkpoint: int) -> str:
    lines: list[str] = []
    for name, spec in ARMS.items():
        cmd = [sys.executable, str(HERE / "compare_arms.py"), "--arm", f"{name}={spec}"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            lines.append(f"  {name:>14}: unreachable (timeout)")
            continue
        for ln in p.stdout.splitlines():
            if ln.strip().startswith(name):
                lines.append("  " + ln.strip())
                break

    # pairs among the arms that are actually comparable (both past the checkpoint)
    done = {}
    for ln in lines:
        try:
            head, rest = ln.split(":", 1)
            n = int(rest.split("=")[1].split()[0])
            done[head.strip()] = n
        except Exception:
            continue
    ready = [n for n, v in done.items() if v >= checkpoint]
    if len(ready) >= 2:
        cmd = [sys.executable, str(HERE / "compare_arms.py")]
        for n in ready:
            cmd += ["--arm", f"{n}={ARMS[n]}"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        body = [ln for ln in p.stdout.splitlines()
                if "shared" in ln or "vs" in ln or " = " in ln]
        lines.append(f"  --- pairs among arms with >= {checkpoint} decided ---")
        lines += ["  " + ln.strip() for ln in body]

    stamp = time.strftime("%F %T")
    marks = ", ".join(f"{n}={v}" for n, v in sorted(done.items()))
    return f"[{stamp}] decided: {marks}\n" + "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--checkpoint", type=int, default=50)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        block = one_pass(args.checkpoint)
        print(block, flush=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(block + "\n")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
