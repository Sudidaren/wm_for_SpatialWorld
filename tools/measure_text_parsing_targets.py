#!/usr/bin/env python3
"""How many tasks' targets can be recovered by parsing the instruction text?

The WM's cheat-free replacement for the task's ground-truth target list is one
extra model call per task (the model names the objects itself).  This measures
the cheaper alternative: no model at all, just dictionary matching of object
names in the instruction.

Dictionary: the object types seen while collecting the perception data
(data/frame_index.pkl) -- the scenes' vocabulary, not the evaluation tasks'
targets.  Scoring uses each task's own target_object_types as ground truth.
"""

from __future__ import annotations

import ast
import json
import pickle
import re
import statistics
from collections import Counter
from pathlib import Path

TASK_ROOT = Path("/home/sudidaren/SpatialWorld/data/ai2thor/tasks")
INDEX = Path("/home/sudidaren/lightwm_phases/data/frame_index.pkl")


def load_vocab() -> list[str]:
    with INDEX.open("rb") as fh:
        d = pickle.load(fh)
    types = d.get("object_types") or []
    if isinstance(types, dict):
        types = list(types.keys())
    return sorted({str(t) for t in types if t})


def forms(t: str, relaxed: bool) -> set[str]:
    ws = [w.lower() for w in re.split(r"(?<=[a-z])(?=[A-Z])|[\s_-]+", t) if w]
    out: set[str] = set()
    if ws:
        out.add("".join(ws))
        out.add(" ".join(ws))
        out.add(ws[-1])
        if relaxed:
            out |= set(ws)
    for f in list(out):
        out.add(f[:-1] if f.endswith("s") else f + "s")
    return {f for f in out if f}


def main() -> int:
    vocab = load_vocab()
    print(f"dictionary: {len(vocab)} object types (perception collection scenes)")
    rows = []
    for p in sorted(TASK_ROOT.glob("*/task.json")):
        d = json.loads(p.read_text())
        raw = d.get("target_object_types")
        if isinstance(raw, str):
            try:
                raw = ast.literal_eval(raw)
            except Exception:
                raw = []
        gt = {str(x).lower() for x in (raw or [])}
        if not gt:
            continue
        instr = str(d.get("instruction") or "")
        text = " " + re.sub(r"[^a-zA-Z ]", " ", instr).lower() + " "
        words = set(text.split())
        hit = {}
        for mode in ("strict", "relaxed"):
            found = set()
            for t in vocab:
                fs = forms(t, relaxed=(mode == "relaxed"))
                if any((" " + f + " ") in text if " " in f else f in words for f in fs):
                    found.add(t.lower())
            hit[mode] = gt & found
        rows.append({"tid": p.parent.name, "gt": gt, "instr": instr,
                     "strict": hit["strict"], "relaxed": hit["relaxed"]})

    n = len(rows)
    print(f"tasks: {n}")
    for mode in ("strict", "relaxed"):
        full = sum(1 for r in rows if len(r[mode]) == len(r["gt"]))
        part = sum(1 for r in rows if r[mode])
        rec = statistics.mean(len(r[mode]) / len(r["gt"]) for r in rows)
        print(f"  [{mode:7s}] all targets hit {full}/{n} = {full/n:.1%} | "
              f"at least one {part}/{n} = {part/n:.1%} | target recall {rec:.1%}")

    missed = Counter()
    for r in rows:
        for g in r["gt"] - r["relaxed"]:
            missed[g] += 1
    print("  targets text parsing misses (top 12):")
    for name, k in missed.most_common(12):
        print(f"    {k:4d}  {name}")
    print("  examples:")
    shown = 0
    for r in rows:
        if r["gt"] - r["relaxed"] and shown < 8:
            print(f"    {r['tid']}: GT={sorted(r['gt'])} instr={r['instr'][:70]!r}")
            shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
