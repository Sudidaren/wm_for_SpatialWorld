"""Aggregate Phase D runs into a three-group paired comparison table."""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))


def collect_results(root: str) -> Dict[str, Dict[str, List[bool]]]:
    """root contains run_*/<task>/steps.jsonl per config dir."""
    out: Dict[str, Dict[str, List[bool]]] = {}
    for ep in sorted(glob.glob(os.path.join(root, "*", "episode.json"))):
        d = json.load(open(ep))
        task = d.get("task_id") or os.path.basename(os.path.dirname(ep))
        # find group from path segment like potato_plate_microwave__B_rule_gate
        group = "unknown"
        parts = os.path.relpath(ep, root).split(os.sep)
        for p in parts:
            if "__" in p:
                group = p.split("__")[-1]
                task = p.split("__")[0]
                break
        out.setdefault(task, {}).setdefault(group, []).append(
            bool(d.get("success")))
    return out


def wilcoxon(a: List[bool], b: List[bool]):
    """Paired one-sided test (a > b).  Returns (p, better_mean_diff)."""
    n = min(len(a), len(b))
    if n == 0:
        return 1.0, 0.0
    diffs = [int(a[i]) - int(b[i]) for i in range(n)]
    # exact sign test for small n (pairs)
    pos = sum(1 for x in diffs if x > 0)
    neg = sum(1 for x in diffs if x < 0)
    from math import comb
    p = 0.0
    k = max(pos, neg)
    for j in range(k, pos + neg + 1):
        p += comb(pos + neg, j) * (0.5 ** (pos + neg))
    return min(1.0, p), (pos - neg) / n


def main(root: str):
    res = collect_results(root)
    print(f"{'task':26s} {'A':>6s} {'B':>6s} {'C':>6s}  B-A  C-A")
    for task in sorted(res):
        g = res[task]
        a = g.get("A_baseline", [])
        b = g.get("B_rule_gate", [])
        c = g.get("C_voi_gate", [])
        ma = sum(a) / len(a) if a else float("nan")
        mb = sum(b) / len(b) if b else float("nan")
        mc = sum(c) / len(c) if c else float("nan")
        _, db = wilcoxon(b, a)
        _, dc = wilcoxon(c, a)
        print(f"{task:26s} {ma:6.2f} {mb:6.2f} {mc:6.2f}  "
              f"{db:+5.2f} {dc:+5.2f}")
    print("\nnote: values are success rates per group; deltas are paired means.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phase_d/runs")
