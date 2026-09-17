#!/usr/bin/env python3
"""Merge a baseline run with its backfill runs and report the per-task result.

Each externally-failed task is re-run (under CloudRendering, which the
originally-hanging rooms need) in a separate run; the merged result takes the
first *decided* outcome for every task id, so a task that the first run never
decided but a backfill did counts as decided.

Usage:
    python3 tools/merge_baselines.py --pull <dir with pulled state.json files>
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

DECIDED = ("success", "failed_model")


def load(path: Path):
    try:
        with path.open() as fh:
            return {t["task_id"]: t for t in json.load(fh).get("tasks", [])}
    except Exception:
        return {}


def merge(paths):
    """First decided outcome wins; failed_external rows are kept only if
    nothing else decided the task."""
    out = {}
    for p in paths:
        if not p.exists():
            continue
        for tid, t in load(p).items():
            cur = out.get(tid)
            if cur is None or (cur.get("status") not in DECIDED
                               and t.get("status") in DECIDED):
                out[tid] = t
    return out


def report(label, tasks, total):
    ai = {k: v for k, v in tasks.items() if True}
    dec = [t for t in ai.values() if t.get("status") in DECIDED]
    suc = [t for t in dec if t.get("success")]
    ext = [t for t in ai.values() if t.get("status") == "failed_external"]
    tsr = len(suc) / len(dec) * 100 if dec else float("nan")
    print(f"{label:<28}{total:>6}{len(dec):>8}{len(suc):>7}{tsr:>8.1f}%{len(ext):>9}{total-len(dec)-len(ext):>7}")
    return len(dec), len(suc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", default=str(Path.home() / "wm_dev/baselines"))
    ap.add_argument("--pull2", default=str(Path.home() / "cloud_monitor/pulled"))
    args = ap.parse_args()
    P = Path(args.pull)
    P2 = Path(args.pull2)

    groups = {
        ("Qwen3-VL-30B-A3B", "AI2-THOR"): (Path.home() / "spatialworld_eval/runs/qwen3vl30b_ai2thor_v1/state.json", [], 311),
        ("Qwen3-VL-30B-A3B", "ProcTHOR"): (Path.home() / "spatialworld_eval/runs/qwen3vl30b_procthor_v1/state.json", [], 127),
        ("Qwen3-VL-8B", "AI2-THOR"): (P / "cqa1__qwen3vl8b_438_v1__state.json",
                                      [P2 / f"weste/qwen3vl8b_ai2thor311_bf{v}/state.json" for v in (5, 6)],
                                      311),
        ("Qwen3-VL-8B", "ProcTHOR"): (P / "cqa1__qwen3vl8b_438_v1__state.json",
                                      [P2 / f"cqa1/qwen3vl8b_procthor127_bf{v}/state.json" for v in (1, 2, 3, 4)],
                                      127),
        ("Kimi-VL-A3B", "AI2-THOR"): (P / "westd__kimivl_a3b_438_v1__state.json",
                                      [P2 / f"westd/kimivl_a3b_ai2thor311_bf{v}/state.json" for v in (1, 2, 3, 4)],
                                      311),
        ("Kimi-VL-A3B", "ProcTHOR"): (P / "westd__kimivl_a3b_438_v1__state.json",
                                      [P2 / f"westd/kimivl_a3b_procthor127_bf{v}/state.json" for v in (1, 2)],
                                      127),
    }
    env_of = {"AI2-THOR": "ai2thor", "ProcTHOR": "procthor"}
    print(f"{'model':<22}{'total':>6}{'decided':>8}{'success':>7}{'TSR':>9}"
          f"{'external':>9}{'missing':>8}")
    print("-" * 80)
    for (model, env), (main_path, backfills, total) in groups.items():
        merged = merge([main_path] + backfills)
        merged = {k: v for k, v in merged.items() if v.get("env") == env_of[env]}
        report(f"{model}", merged, total)
    print("\n(decided reads: total / decided / success / TSR / external / still missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
