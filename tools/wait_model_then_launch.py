#!/usr/bin/env python3
"""Launch an arm on a card as soon as its model download finishes.

A 30B BF16 download takes hours and the transfer runs while nobody is
watching.  This watcher turns "the download finished" into "the arm started"
without a human in the loop, and it will not start an arm on a half-written
model: it waits for every shard *and* for the downloader to exit.

    python3 tools/wait_model_then_launch.py --once
    python3 tools/wait_model_then_launch.py          # poll until launched

The arms are the ones the day's plan calls for:
  C -> base_q30_a311        the main-table baseline, so the paired WM arm on D
                            and this one finish inside one download window
  B -> wm_q30_da2_a1_nomem  the memory-view ablation (memory_frames=0), the
                            one experiment that can falsify the WM claim
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from card_ssh import run  # noqa: PLC2701 - sibling tool

HERE = Path(__file__).resolve().parent
MODEL_DIR = "/root/autodl-tmp/models/qwen3vl-30b-bf16"
SHARDS = 13

COMMON = ["--scenes", "ai2thor", "--total", "311"]

PLAN: dict[str, list[str]] = {
    "C": COMMON + [
        "--arm", "qwen3vl-30b|http://127.0.0.1:8000/v1|off|llm|base_q30_a311|10",
    ],
    "B": COMMON + [
        "--depth-source", "da2", "--da2-scale", "1.3816",
        "--env", "LIGHTWM_MEMORY_FRAMES=0",
        "--arm", "qwen3vl-30b|http://127.0.0.1:8000/v1|off|wm|wm_q30_da2_a1_nomem|10",
    ],
}

READY = rf"""
echo -n "shards="; ls {MODEL_DIR}/*.safetensors 2>/dev/null | wc -l
echo -n "index=";  test -f {MODEL_DIR}/model.safetensors.index.json && echo yes || echo no
echo -n "downloader="; pgrep -fc "[s]napshot_download|[m]odelscope" 2>/dev/null || echo 0
echo -n "size=";  du -sh {MODEL_DIR} 2>/dev/null | cut -f1
echo -n "disk=";  df -h /root/autodl-tmp | awk 'NR==2{{print $4}}'
"""


def status(card: str) -> dict[str, str]:
    rc, out, err = run(card, READY, timeout=180)
    vals: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            vals[k.strip()] = v.strip()
    if err.strip():
        vals["stderr"] = err.strip()[:200]
    return vals


def is_ready(v: dict[str, str]) -> tuple[bool, str]:
    try:
        shards = int(v.get("shards", "0"))
    except ValueError:
        shards = 0
    busy = int(v.get("downloader", "0") or 0)
    if shards >= SHARDS and v.get("index") == "yes" and busy == 0:
        return True, f"{shards} shards, index present, downloader exited"
    return False, (f"{shards}/{SHARDS} shards, index={v.get('index')}, "
                   f"downloader={'running' if busy else 'gone'}, {v.get('size', '?')}")


def launch(card: str) -> int:
    cmd = [sys.executable, str(HERE / "card_ctl.py"), "launch", card] + PLAN[card]
    print(f"  launching: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", nargs="+", default=sorted(PLAN))
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true", help="report readiness only")
    ap.add_argument("--max-hours", type=float, default=8.0)
    args = ap.parse_args()

    deadline = time.time() + args.max_hours * 3600
    waiting = set(args.cards)
    while True:
        for card in sorted(waiting):
            try:
                v = status(card)
            except Exception as exc:  # noqa: BLE001
                print(f"[{time.strftime('%H:%M:%S')}] {card}: unreachable "
                      f"({type(exc).__name__})", flush=True)
                continue
            ok, why = is_ready(v)
            print(f"[{time.strftime('%H:%M:%S')}] {card}: "
                  f"{'READY' if ok else 'waiting'} -- {why}", flush=True)
            if ok and not args.once:
                rc = launch(card)
                print(f"  {card}: launch rc={rc}", flush=True)
                if rc == 0:
                    waiting.discard(card)
        if args.once or not waiting:
            return 0
        if time.time() > deadline:
            print(f"gave up after {args.max_hours} h; still waiting on {sorted(waiting)}",
                  flush=True)
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
