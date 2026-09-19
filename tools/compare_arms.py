#!/usr/bin/env python3
"""Paired comparison of evaluation arms on the tasks they share.

Reading two success rates off two different task subsets is how a comparison
goes wrong: the arms here cover different subsets (a partial closed-model run
covers the first N tasks, an interrupted arm covers whatever finished first).
So every number this prints is computed on the *intersection* of the tasks the
compared arms actually decided, and it reports how much that costs.

    python3 tools/compare_arms.py \
        --arm T2=cardD:/root/autodl-tmp/runs_local/wm_q30_da2_a311/results.csv \
        --arm T1=cardC:/root/autodl-tmp/runs_local/base_q30_a311/results.csv \
        --arm T6=local:/home/sudidaren/spatialworld_eval/runs/gemini31pro_ai2thor_procthor_438_frozen_v1/results.csv

Each spec is NAME=SOURCE:PATH where SOURCE is a card letter or "local".
Prints: per-arm TSR on the shared set, pairwise McNemar (exact binomial on the
discordant pairs) and a task-level bootstrap CI on the difference.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from pathlib import Path

from card_ssh import CARDS, run as card_run  # noqa: PLC2701 - sibling tool


def load(spec: str) -> dict[str, dict]:
    name, _, rest = spec.partition("=")
    source, _, path = rest.partition(":")
    if source == "local":
        text = Path(path).read_text(encoding="utf-8-sig")
    else:
        if source not in CARDS:
            raise SystemExit(f"unknown source {source!r} (use a card letter or 'local')")
        rc, out, err = card_run(source, f"cat {path}", timeout=180)
        if rc != 0 or not out.strip():
            raise SystemExit(f"could not read {path} on card {source}: {err[:200]}")
        text = out
    # the harness writes a UTF-8 BOM; without stripping it the first column is
    # named "\ufeffTask ID" and every row silently lands in the skip branch
    text = text.lstrip("\ufeff")
    rows = {}
    for r in csv.DictReader(io.StringIO(text)):
        tid = (r.get("Task ID") or "").strip()
        if not tid:
            continue
        rows[tid] = r
    return {"name": name, "rows": rows}


def decided(row: dict) -> bool:
    return str(row.get("Completed", "")).lower() in ("true", "false")


def success(row: dict) -> bool:
    return str(row.get("Success", "")).lower() == "true"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for discordant counts b (A only) and c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_ci(pairs: list[tuple[bool, bool]], iters: int = 2000,
                 alpha: float = 0.05) -> tuple[float, float]:
    if not pairs:
        return (0.0, 0.0)
    rng = random.Random(20260919)
    diffs = []
    n = len(pairs)
    for _ in range(iters):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        a = sum(1 for x, _ in s if x) / n
        b = sum(1 for _, y in s if y) / n
        diffs.append(a - b)
    diffs.sort()
    lo = diffs[int(alpha / 2 * iters)]
    hi = diffs[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=SOURCE:PATH (repeatable, source is a card letter or local)")
    ap.add_argument("--scenes", default="ai2thor")
    args = ap.parse_args()

    arms = [load(s) for s in args.arm]
    for a in arms:
        a["rows"] = {k: v for k, v in a["rows"].items()
                     if (v.get("Environment") or "") == args.scenes}

    print(f"scenes={args.scenes}")
    for a in arms:
        rows = a["rows"]
        dec = [r for r in rows.values() if decided(r)]
        ok = sum(1 for r in dec if success(r))
        print(f"  {a['name']:>3}: rows={len(rows):4d} decided={len(dec):4d} "
              f"success={ok:3d} TSR={ok / max(len(dec), 1):.3f}")

    shared = set(arms[0]["rows"])
    for a in arms[1:]:
        shared &= set(a["rows"])
    shared = {t for t in shared if all(decided(a["rows"][t]) for a in arms)}
    shared = sorted(shared)
    print(f"\nshared decided tasks: {len(shared)}")
    if not shared:
        return 1

    print("\n-- on the shared set --")
    for a in arms:
        ok = sum(1 for t in shared if success(a["rows"][t]))
        print(f"  {a['name']:>3}: {ok:3d}/{len(shared)} = {ok / len(shared):.3f}")

    print("\n-- pairwise --")
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            A, B = arms[i], arms[j]
            pairs = [(success(A["rows"][t]), success(B["rows"][t])) for t in shared]
            b = sum(1 for x, y in pairs if x and not y)
            c = sum(1 for x, y in pairs if y and not x)
            diff = sum(1 for x, _ in pairs if x) / len(pairs) - \
                sum(1 for _, y in pairs if y) / len(pairs)
            lo, hi = bootstrap_ci(pairs)
            p = mcnemar_exact(b, c)
            print(f"  {A['name']} vs {B['name']}: diff={diff:+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]  discordant {A['name']}only={b} "
                  f"{B['name']}only={c}  McNemar p={p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
