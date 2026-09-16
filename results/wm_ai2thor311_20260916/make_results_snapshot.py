#!/usr/bin/env python3
"""Build the published snapshot for the 2026-09-16 AI2-THOR-311 WM run.

Reads the state.json files pulled from the three cloud boxes and writes:
  paired_tasks.csv  -- one row per (model, task) with WM and baseline outcome
  summary.json      -- headline numbers, failure structure, token cost
  (README.md is written by hand)

Usage:  python3 make_results_snapshot.py <out_dir>
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter

PULL = os.path.expanduser("~/wm_dev/pull_latest")
RUNS = os.path.expanduser("~/spatialworld_eval/runs")

MODELS = [
    ("qwen3vl-8b", "Qwen3-VL-8B-Instruct",
     f"{PULL}/wm_ai2thor311_qwen3vl8b_v1.state.json",
     f"{PULL}/qwen3vl8b_438_v1.state.json"),
    ("kimivl-a3b", "Kimi-VL-A3B-Instruct",
     f"{PULL}/wm_ai2thor311_kimivl_a3b_v1.state.json",
     f"{PULL}/kimivl_a3b_438_v1.state.json"),
    ("qwen3vl-30b", "Qwen3-VL-30B-A3B",
     f"{PULL}/wm_ai2thor311_qwen3vl30b_v1.state.json",
     f"{RUNS}/qwen3vl30b_ai2thor_v1/state.json"),
]


def load(path):
    with open(path) as fh:
        return {t["task_id"]: t for t in json.load(fh).get("tasks", [])}


def decided(t):
    return t.get("status") in ("success", "failed_model")


def bucket(reason):
    reason = (reason or "").strip()
    if not reason:
        return "other"
    if "success conditions not met" in reason:
        return "premature_done"
    if "step limit" in reason.lower():
        return "step_limit"
    return "other"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    summary = {"task_set": "ai2thor", "n_tasks_total": 311,
               "metric": "TSR = success / decided; failed_external excluded",
               "pulled_at": None, "models": {}}

    for key, label, wm_path, base_path in MODELS:
        wm, base = load(wm_path), load(base_path)
        base = {k: v for k, v in base.items() if v.get("env") == "ai2thor"}
        ids = [k for k, v in wm.items() if decided(v)]
        w_only = b_only = both = neither = 0
        reasons = Counter()
        toks = []
        for tid in ids:
            w = bool(wm[tid].get("success"))
            b = bool(base.get(tid, {}).get("success"))
            if w and b:
                both += 1
            elif w:
                w_only += 1
            elif b:
                b_only += 1
            else:
                neither += 1
            if not w:
                reasons[bucket(wm[tid].get("fail_reason"))] += 1
            if wm[tid].get("total_tokens"):
                toks.append(wm[tid]["total_tokens"])
            rows.append({
                "model": key, "task_id": tid,
                "scene": wm[tid].get("scene", ""),
                "instruction": wm[tid].get("instruction", ""),
                "wm_success": int(w), "baseline_success": int(b),
                "wm_status": wm[tid].get("status", ""),
                "wm_fail_reason": wm[tid].get("fail_reason") or "",
                "wm_steps": wm[tid].get("actual_steps"),
                "wm_total_tokens": wm[tid].get("total_tokens"),
            })
        n = len(ids)
        ws, bs = w_only + both, b_only + both
        fails = n - ws
        summary["models"][key] = {
            "label": label,
            "decided": n,
            "wm_success": ws,
            "wm_tsr": round(ws / n, 4) if n else None,
            "baseline_success": bs,
            "baseline_tsr": round(bs / n, 4) if n else None,
            "wm_only": w_only,
            "baseline_only": b_only,
            "mcnemar_exact_p": round(mcnemar_exact(b_only, w_only), 4),
            "premature_done": reasons["premature_done"],
            "premature_done_share_of_failures":
                round(reasons["premature_done"] / fails, 4) if fails else None,
            "step_limit": reasons["step_limit"],
            "step_limit_share_of_failures":
                round(reasons["step_limit"] / fails, 4) if fails else None,
            "avg_total_tokens": round(sum(toks) / len(toks)) if toks else None,
        }

    summary["caveats"] = [
        "8B / Kimi baselines come from their 438-task runs, of which only "
        "267/311 AI2-THOR tasks were decided; pairing is restricted to the "
        "WM-decided subset.",
        "The 30B WM arm was still running when this snapshot was taken.",
        "Target hint (WM_TARGET_HINT) and CheckState (WM_STATE_CHECK) were "
        "OFF for every arm in this snapshot.",
    ]

    with open(os.path.join(out_dir, "paired_tasks.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary["models"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
