#!/usr/bin/env python3
"""How much of the real task target set does the model's own wording cover?

The cheat-free replacement for the task's ground-truth ``target_object_types``
is one extra call per task: the model names the objects the instruction is
about, and ``target_priority`` keeps those.  Whether that is good enough is an
empirical question, so this compares the self-stated names against the task
files and reports coverage.

    python3 tools/measure_selfstated_targets.py --run B:/root/autodl-tmp/runs_local/wm_q30_da2_final_a311
    python3 tools/measure_selfstated_targets.py --run local:/home/.../runs/wm_gemini31pro_da2_a311

Only runs with the memory channel on carry the self-stated names (they appear
in the hint as ``（任务目标…）``), which is exactly the configuration we care
about.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path

from card_ssh import CARDS, run as card_run  # noqa: PLC2701 - sibling tool

TASK_ROOT = Path("/home/sudidaren/SpatialWorld/data/ai2thor/tasks")
STATED_RE = re.compile(r"-\s*([A-Za-z][A-Za-z0-9]*)\s*（任务目标")


def sh(source: str, cmd: str, timeout: int = 900) -> str:
    if source == "local":
        return subprocess.run(["bash", "-lc", cmd], capture_output=True,
                              text=True, timeout=timeout).stdout
    rc, out, err = card_run(source, cmd, timeout=timeout)
    return out


def normalise(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def ground_truth(tid: str) -> set[str]:
    f = TASK_ROOT / tid / "task.json"
    if not f.exists():
        return set()
    d = json.loads(f.read_text())
    raw = d.get("target_object_types")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except Exception:
            raw = [raw]
    return {normalise(str(x)) for x in (raw or [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="SOURCE:PATH (card letter or local)")
    ap.add_argument("--min-hinted-steps", type=int, default=2,
                    help="ignore tasks whose logs are a stub (killed before the "
                         "memory channel ever rendered)")
    args = ap.parse_args()
    source, _, root = args.run.partition(":")
    if source not in ("local", *CARDS):
        raise SystemExit(f"unknown source {source!r}")

    dump = sh(source,
              f"cd {root} && find . -name steps.jsonl | while read f; do "
              "echo \"===FILE $f\"; cat \"$f\"; done")
    stated: dict[str, set[str]] = {}
    hinted: dict[str, int] = {}
    cur = None
    for line in dump.splitlines():
        if line.startswith("===FILE "):
            m = re.search(r"ai2thor\d+", line)
            cur = m.group(0) if m else None
            if cur:
                stated[cur] = set()
                hinted[cur] = 0
            continue
        if cur is None:
            continue
        try:
            s = json.loads(line)
        except Exception:
            continue
        hint = s.get("mem_hint") or ""
        if hint:
            hinted[cur] = hinted.get(cur, 0) + 1
        for name in STATED_RE.findall(hint):
            stated[cur].add(normalise(name))

    rows = []
    for tid, names in sorted(stated.items()):
        if hinted.get(tid, 0) < args.min_hinted_steps:
            continue
        gt = ground_truth(tid)
        if not gt:
            continue
        rows.append({
            "tid": tid,
            "gt": gt,
            "said": names,
            "hit": len(gt & names),
            "extra": len(names - gt),
            "hinted": hinted.get(tid, 0),
        })

    if not rows:
        print("没有可统计的任务（需要开着记忆通道的 run）")
        return 1
    n = len(rows)
    full = sum(1 for r in rows if r["hit"] == len(r["gt"]))
    part = sum(1 for r in rows if r["hit"] > 0)
    recall = statistics.mean(r["hit"] / len(r["gt"]) for r in rows)
    extra = statistics.mean(r["extra"] for r in rows)
    empty = sum(1 for r in rows if not r["said"])
    print(f"run={root}")
    print(f"  任务数 {n}")
    print(f"  目标**全部**被自述覆盖: {full}/{n} = {full/n:.1%}")
    print(f"  至少覆盖一个目标:       {part}/{n} = {part/n:.1%}")
    print(f"  目标级召回率(均值):     {recall:.1%}")
    print(f"  平均多说出的额外物体:   {extra:.2f} 个/任务")
    print(f"  自述为空的任务:         {empty}/{n} = {empty/n:.1%}")
    print("  漏掉的目标 top10:")
    for name, k in Counter(x for r in rows for x in (r["gt"] - r["said"])).most_common(10):
        print(f"    {k:4d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
