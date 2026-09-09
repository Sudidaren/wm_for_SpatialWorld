# -*- coding: utf-8 -*-
"""汇总统计：TSR / 分场景 / 分场景+类别 / 步数与 token 摘要。

口径与官方一致：
    - Completed=true/false 的才算已测（TSR 分母）；
    - Completed=null（api/env 故障或未跑）不计入分母；
    - 成功率 = true / (true+false)。
另外给出官方 CSV 之外的效率参考列：
    - SE_success = golden_steps / actual_steps（只统计成功的任务），
      可自行在 eval_config 里关掉或改口径。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def aggregate(rows: list[dict]) -> dict:
    buckets = defaultdict(lambda: {
        "total": 0, "success": 0, "failure": 0, "pending": 0,
        "steps_success": [], "se_success": [],
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "api_calls": 0, "duration": 0.0,
    })
    for r in rows:
        completed = r.get("completed")
        env = r.get("env") or ""
        cat = r.get("category") or ""
        key_all = ("ALL", "ALL")
        for key in ((env, cat), (env, "ALL"), key_all):
            b = buckets[key]
            b["total"] += 1
            status = str(completed).lower() if completed else "null"
            if status == "true":
                b["success"] += 1
                actual = _num(r.get("actual_steps"))
                golden = _num(r.get("golden_steps"))
                if actual:
                    b["steps_success"].append(actual)
                    if golden:
                        b["se_success"].append(golden / actual)
            elif status == "false":
                b["failure"] += 1
            else:
                b["pending"] += 1
            b["prompt_tokens"] += r.get("prompt_tokens") or 0
            b["completion_tokens"] += r.get("completion_tokens") or 0
            b["total_tokens"] += r.get("total_tokens") or 0
            b["api_calls"] += r.get("api_calls") or 0
            b["duration"] += r.get("duration_sec") or 0

    def pack(key):
        b = buckets[key]
        decided = b["success"] + b["failure"]
        mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None
        return {
            "env": key[0],
            "category": key[1],
            "total": b["total"],
            "completed": decided,
            "pending": b["pending"],
            "success": b["success"],
            "failure": b["failure"],
            "tsr": round(b["success"] / decided, 4) if decided else None,
            "avg_steps_success": mean(b["steps_success"]),
            "avg_se_success": mean(b["se_success"]),
            "prompt_tokens": b["prompt_tokens"],
            "completion_tokens": b["completion_tokens"],
            "total_tokens": b["total_tokens"],
            "api_calls": b["api_calls"],
            "duration_sec": round(b["duration"], 1),
        }

    summary = [
        pack(("ALL", "ALL")),
        *[pack((env, "ALL"))
          for env in sorted({r.get("env") for r in rows})],
    ]
    cat_keys = sorted({(r.get("env"), r.get("category"))
                       for r in rows if r.get("category")})
    summary.extend(pack(k) for k in cat_keys)
    summary = [s for s in summary if s is not None]
    return {"groups": summary}


def write_summary(run_dir: Path, rows: list[dict], plan: dict) -> None:
    agg = aggregate(rows)
    out = {
        "plan": plan,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "groups": agg["groups"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = run_dir / "summary_by_env_category.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out["groups"][0].keys()))
        writer.writeheader()
        for g in out["groups"]:
            writer.writerow(g)
    print(f"summary : {run_dir / 'summary.json'}")
    for g in out["groups"]:
        tag = f"[{g['env']} / {g['category']}]" if g["category"] != "ALL" \
            else f"[{g['env'] or 'ALL'}]"
        print(f"  {tag:24s} TSR={g['tsr']} "
              f"({g['success']}/{g['completed']}) "
              f"pending={g['pending']} avg_steps={g['avg_steps_success']}")
