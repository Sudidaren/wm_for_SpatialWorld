#!/usr/bin/env python3
"""Failure profile of one evaluation arm, from its own per-step logs.

Answers "what are these MLLMs actually bad at" with numbers instead of
impressions:

  * how many steps a failure takes relative to the golden plan (overrun),
  * how much of the budget goes to navigation vs interaction,
  * how much of it is repetition (the same action, or the same rotate pair),
  * which failure reasons dominate.

    python3 tools/analyze_failures.py --run local:/home/.../runs/<name>
    python3 tools/analyze_failures.py --run D:/root/autodl-tmp/runs_local/<name>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path

from card_ssh import CARDS, run as card_run  # noqa: PLC2701 - sibling tool

NAV = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight",
       "LookUp", "LookDown", "Teleport"}
INT = {"PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn",
       "ToggleObjectOff", "SliceObject", "DropHandObject", "CleanObject",
       "FillObjectWithLiquid", "DirtyObject", "UseUpObject", "BreakObject"}


def sh(source: str, cmd: str, timeout: int = 300) -> str:
    if source == "local":
        return subprocess.run(["bash", "-lc", cmd], capture_output=True,
                              text=True, timeout=timeout).stdout
    rc, out, err = card_run(source, cmd, timeout=timeout)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="SOURCE:PATH (card letter or local)")
    ap.add_argument("--limit-tasks", type=int, default=400)
    args = ap.parse_args()

    source, _, root = args.run.partition(":")
    if source not in ("local", *CARDS):
        raise SystemExit(f"unknown source {source!r}")

    csv_text = sh(source, f"cat {root}/results.csv")
    rows = {r["Task ID"]: r for r in csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))}
    if not rows:
        raise SystemExit("no results.csv rows")

    # Fetch every per-step log in ONE round trip (a `cat` per task is 300 ssh
    # calls for a full arm, which is slower than the token counting itself).
    dump = sh(source,
              f"cd {root} && find . -name steps.jsonl -printf '%T@ %p\\n' 2>/dev/null "
              "| sort -n | awk '{print $2}' | while read f; do "
              "echo \"===FILE $f\"; cat \"$f\"; done", timeout=900)
    files: dict[str, list[str]] = {}
    cur = None
    for line in dump.splitlines():
        if line.startswith("===FILE "):
            path = line[len("===FILE "):].strip()
            m = re.search(r"ai2thor\d+", path)
            cur = m.group(0) if m else None
            if cur:
                files[cur] = []          # later file overwrites: the last attempt wins
        elif cur is not None:
            files[cur].append(line)

    stat_rows = []
    for tid, lines in list(files.items())[:args.limit_tasks]:
        row = rows.get(tid)
        if row is None or str(row.get("Completed")).lower() not in ("true", "false"):
            continue
        acts = []
        for line in lines:
            try:
                s = json.loads(line)
            except Exception:
                continue
            a = (s.get("action_string") or "").strip()
            m = re.match(r"([A-Za-z]+)", a)
            if m:
                acts.append(m.group(1))
        if not acts:
            continue
        # the first step is the harness's own init action, not the model's
        acts = acts[1:] or acts
        nav = sum(1 for a in acts if a in NAV)
        inter = sum(1 for a in acts if a in INT)
        rep = 0
        for i in range(2, len(acts)):
            if acts[i] == acts[i - 1] == acts[i - 2]:
                rep += 1
        pairs = sum(1 for i in range(1, len(acts))
                    if {acts[i - 1], acts[i]} == {"RotateLeft", "RotateRight"})
        golden = int(float(row.get("golden_actions_count") or 0) or 0)
        stat_rows.append({
            "tid": tid,
            "success": str(row.get("Success")).lower() == "true",
            "steps": len(acts),
            "golden": golden,
            "nav": nav / len(acts),
            "inter": inter / len(acts),
            "repeat3": rep / len(acts),
            "rot_pair": pairs / len(acts),
            "reason": str(row.get("failure_reason") or "")[:60],
            "status": str(row.get("Status") or ""),
        })

    if not stat_rows:
        print("no per-step logs parsed")
        return 1

    ok = [r for r in stat_rows if r["success"]]
    bad = [r for r in stat_rows if not r["success"]]
    print(f"run={root}")
    print(f"  解析到 {len(stat_rows)} 个任务: 成功 {len(ok)} ({len(ok)/len(stat_rows):.1%})")
    for label, group in (("成功", ok), ("失败", bad)):
        if not group:
            continue
        st = [r["steps"] for r in group]
        gs = [r["golden"] for r in group if r["golden"]]
        print(f"  [{label}] n={len(group)} 步数中位 {statistics.median(st):.0f}"
              + (f" / 黄金 {statistics.median(gs):.0f} (超支 {statistics.median(st)/max(statistics.median(gs),1):.1f}x)"
                 if gs else "")
              + f" 导航占比 {statistics.median([r['nav'] for r in group]):.0%}"
              + f" 交互占比 {statistics.median([r['inter'] for r in group]):.0%}"
              + f" 连续重复3次 {statistics.median([r['repeat3'] for r in group]):.0%}"
              + f" 原地左右转 {statistics.median([r['rot_pair'] for r in group]):.0%}")
    print("  失败原因 top6:")
    for reason, n in Counter(r["reason"] for r in bad).most_common(6):
        print(f"    {n:4d}  {reason or '(空)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
