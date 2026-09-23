"""Build the evaluation tables from the runs that actually exist.

Definitions follow ``spatialworld_eval/docs/evaluation_section_draft.md``:

  success/failure/null : official terminal verifier / model failure / API or
                         simulator failure after retries (``failed_external``)
  TSR                  : success / (success + failure)      (null excluded)
  null rate            : null / total
  completion rate      : success / total                    (null counted)
  SE                   : 1 - steps / max_steps
  SE_success           : mean over successful tasks
  SE_all               : mean over success + failure (null excluded)

Run:  python build_tables.py            (writes tables.md next to this file)
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

RUNS = "/home/sudidaren/spatialworld_eval/runs"
HERE = os.path.dirname(os.path.abspath(__file__))

EXTERNAL = {"api_error", "env_error", "external_error", "external"}

BATCHES = [
    ("Qwen3-VL-30B-A3B (BF16, vLLM)", "AI2-THOR",
     "qwen3vl30b_ai2thor_v1", 311),
    ("Qwen3-VL-30B-A3B (BF16, vLLM)", "ProcTHOR",
     "qwen3vl30b_procthor_v1", 127),
    ("Gemini 3.1 Pro (frozen v1)", "AI2-THOR",
     "gemini31pro_ai2thor_procthor_438_frozen_v1", 311),
    ("Gemini 3.1 Pro (frozen v1)", "ProcTHOR",
     "gemini31pro_procthor127_frozen_v1", 127),
]


def _rows(name: str) -> List[Dict]:
    p = os.path.join(RUNS, name, "results.csv")
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _g(r: Dict, k: str) -> str:
    return (r.get(k) or "").strip()


def _int(v, default=None):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def stats(name: str, planned: int) -> Dict:
    rows = _rows(name)
    succ = [r for r in rows if _g(r, "Success").lower() == "true"]
    null = [r for r in rows if _g(r, "Status") == "failed_external"
            or _g(r, "Failure Type") in EXTERNAL]
    decided = [r for r in rows if r not in null]
    fail = [r for r in decided if _g(r, "Success").lower() != "true"]

    def se(rs):
        vals = []
        for r in rs:
            s, m = _int(_g(r, "Actual Steps")), _int(_g(r, "Max Steps"))
            if s is not None and m:
                vals.append(1.0 - s / m)
        return sum(vals) / len(vals) if vals else None

    steps_succ = [_int(_g(r, "Actual Steps")) for r in succ]
    steps_succ = [s for s in steps_succ if s is not None]
    steps_fail = [_int(_g(r, "Actual Steps")) for r in fail]
    steps_fail = [s for s in steps_fail if s is not None]
    return {
        "run": name, "planned": planned, "run_rows": len(rows),
        "success": len(succ), "failure": len(fail), "null": len(null),
        "tsr": (len(succ) / len(decided)) if decided else None,
        "null_rate": (len(null) / len(rows)) if rows else None,
        "completion": (len(succ) / planned) if planned else None,
        "se_success": se(succ), "se_all": se(decided),
        "avg_steps_succ": (sum(steps_succ) / len(steps_succ)) if steps_succ else None,
        "avg_steps_fail": (sum(steps_fail) / len(steps_fail)) if steps_fail else None,
        "complete": len(decided) >= planned and len(null) == 0,
        "decided": len(decided),
    }


def fmt(v: Optional[float], pct: bool = True, nd: int = 1) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.{nd}f}%" if pct else f"{v:.{nd}f}"


def failure_taxonomy(name: str) -> Dict[str, int]:
    from collections import Counter
    rows = _rows(name)
    c: Counter = Counter()
    for r in rows:
        status = _g(r, "Status")
        reason = _g(r, "Failure Reason") or _g(r, "failure_reason")
        ft = _g(r, "Failure Type")
        if status == "failed_external" or ft in EXTERNAL:
            c["API/env null"] += 1
            continue
        if _g(r, "Success").lower() == "true":
            continue
        low = reason.lower()
        if "maximum step" in low:
            c["Max steps"] += 1
        elif "done" in low and "condition" in low:
            c["DONE-but-wrong"] += 1
        elif "consecutive" in low or "action error" in low or "action_error" in ft:
            c["Consecutive action failures"] += 1
        elif "model" in low and "fail" in low:
            c["Model FAIL"] += 1
        else:
            c["Other"] += 1
    return dict(c)


def main() -> int:
    lines: List[str] = ["# Evaluation tables (auto-built from existing runs)", ""]
    lines.append("> `-` = not measured yet. Coverage column shows how much of the")
    lines.append("> planned batch is already decided (nulls excluded).")
    lines.append("")
    lines.append("## Table 5.4.1 — Model x environment")
    lines.append("")
    lines.append("| Method | Env | Success | Failure | Null | Total | "
                 "TSR | Null rate | Completion | SE_success | SE_all | "
                 "Avg steps (succ) | Avg steps (fail) | Coverage |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for method, env, run, planned in BATCHES:
        s = stats(run, planned)
        cov = f"{s['decided']}/{planned}" + ("" if s["complete"] else " (partial)")
        lines.append(
            f"| {method} | {env} | {s['success']} | {s['failure']} | {s['null']} "
            f"| {planned} | {fmt(s['tsr'])} | {fmt(s['null_rate'])} "
            f"| {fmt(s['completion'])} | {fmt(s['se_success'])} "
            f"| {fmt(s['se_all'])} | {fmt(s['avg_steps_succ'], pct=False)} "
            f"| {fmt(s['avg_steps_fail'], pct=False)} | {cov} |")
    lines.append("")
    lines.append("## Table 5.4.3 — Failure taxonomy (decided non-success only)")
    lines.append("")
    lines.append("| Method | Env | DONE-but-wrong | Max steps | "
                 "Consecutive action failures | API/env null | Other |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for method, env, run, _p in BATCHES:
        t = failure_taxonomy(run)
        lines.append(
            f"| {method} | {env} | {t.get('DONE-but-wrong', 0)} "
            f"| {t.get('Max steps', 0)} "
            f"| {t.get('Consecutive action failures', 0)} "
            f"| {t.get('API/env null', 0)} | {t.get('Other', 0)} |")
    lines.append("")
    lines.append("## World-model A/B probe (not part of the frozen batch)")
    lines.append("")
    lines.append("| Task | Arm | Success | Steps | Prompt tokens |")
    lines.append("|---|---|---|---:|---:|")
    ab = os.path.join("/home/sudidaren/ab_probe", "ab_results_main4.json")
    if os.path.isfile(ab):
        import json
        with open(ab, encoding="utf-8") as fh:
            for r in json.load(fh):
                lines.append(f"| {r['task']} | {r['arm']} "
                             f"| {r.get('success')} | {r.get('steps')} "
                             f"| {r.get('prompt_tokens')} |")
    else:
        lines.append("| - | - | - | - | - |")
    lines.append("")
    lines.append("## Still empty (needs data)")
    lines.append("")
    for row in (
        "GPT-5 x {AI2-THOR, ProcTHOR, VirtualHome, Overall}",
        "VirtualHome x {Qwen, Gemini}",
        "GPT-5 + WingmanWM x {AI2-THOR, ProcTHOR, Overall}",
        "DreamerV3 / DIAMOND baselines",
        "Table 5.4.2 category breakdown (TSR)",
        "Table 5.4.4 ablations (LLM only / +perception+memory / +gate)",
        "Table 5.4.5 cost and runtime",
    ):
        lines.append(f"- {row}")
    out = os.path.join(HERE, "tables.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nsaved ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
