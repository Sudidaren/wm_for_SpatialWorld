#!/usr/bin/env python3
r"""Turn a run's results.csv into the cells of the v2 main table.

Reports, per environment: N, TSR, avg steps, premature-DONE share,
step-limit share, refusal share, malformed-output share, avg tokens/task (k),
avg API calls/task, avg seconds/task.

Usage:
  python3 fill_main_table.py --batch "Qwen3-VL-30B-A3B|--|runs/kimivl.../results.csv"

  --batch "Model|WM|path;..."  inspect a single condition; with --latex it
          emits one half-row (6 cells: TSR, Step, Inv, Prem, SLim, Tok) per
          environment.
  --pair  "Model|base.csv|wm.csv;..."  emit a whole table row per environment,
          i.e. the base 6 cells followed by the WM 6 cells, which is the layout
          of main_table_v2.tex.
  --ci additionally prints the 95% Wilson interval (off by default).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load(path: str) -> list[dict]:
    """Rows of one results.csv, de-duplicated by Task ID (last row wins)."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    by_id: "OrderedDict[str, dict]" = OrderedDict()
    for r in rows:
        by_id[r.get("Task ID", "")] = r
    return list(by_id.values())


def decided(r: dict) -> bool:
    return str(r.get("Completed", "")).strip().lower() in ("true", "false")


def fnum(v, default=0.0) -> float:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def stats(rows: list[dict]) -> dict:
    rows = [r for r in rows if decided(r)]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    success = sum(1 for r in rows if str(r.get("Success", "")).strip().lower() == "true")
    steps = [fnum(r.get("Actual Steps")) for r in rows]
    tokens = [fnum(r.get("token_total") or r.get("token total")) for r in rows]
    calls = [fnum(r.get("API Calls")) for r in rows]
    secs = [fnum(r.get("Duration Sec")) for r in rows]

    def share(needle: str) -> float:
        return sum(
            1 for r in rows
            if needle.lower() in str(r.get("failure_reason", "")).lower()
        ) / n

    lo, hi = wilson(success, n)
    return {
        "n": n,
        "success": success,
        "tsr": success / n,
        "ci": (lo, hi),
        "steps": sum(steps) / n,
        "premature_done": share("claimed DONE"),
        "step_limit": share("maximum step limit"),
        "refused": share("cannot be completed or refused"),
        "malformed": share("no result json produced"),
        "tokens_k": sum(tokens) / n / 1000.0,
        "calls": sum(calls) / n,
        "secs": sum(secs) / n,
    }


def fmt(s: dict, ci: bool = False) -> str:
    if s.get("n", 0) == 0:
        return "(no decided rows)"
    lo, hi = s["ci"]
    tsr = f"{s['tsr']*100:.1f}%" + (f" [{lo*100:.1f}, {hi*100:.1f}]" if ci else "")
    return (
        f"TSR {tsr} | "
        f"steps {s['steps']:.1f} | prem.DONE {s['premature_done']*100:.1f}% | "
        f"step-limit {s['step_limit']*100:.1f}% | refused {s['refused']*100:.1f}% | "
        f"malformed {s['malformed']*100:.1f}% | tok/task {s['tokens_k']:.1f}k | "
        f"calls/task {s['calls']:.1f} | {s['secs']:.0f}s"
    )


def metric_cells(s: dict, ci: bool = False) -> list[str]:
    """The six numbers of one condition, in table order."""
    if s.get("n", 0) == 0:
        return ["--"] * 6
    lo, hi = s["ci"]
    tsr = f"{s['tsr']*100:.1f}" + (f"\\,[{lo*100:.1f}, {hi*100:.1f}]" if ci else "")
    return [
        tsr,
        f"{s['steps']:.1f}",
        "--",  # invalid-step share needs the interaction analysis, not results.csv
        f"{s['premature_done']*100:.1f}",
        f"{s['step_limit']*100:.1f}",
        f"{s['tokens_k']:.1f}",
    ]


def latex_half(s: dict, env: str, ci: bool = False) -> str:
    """Half a table row: environment + the six cells of one condition."""
    return f"& {env} & " + " & ".join(metric_cells(s, ci)) + r" \\"


def latex_full_row(base: dict, wm: dict, env: str, ci: bool = False) -> str:
    """A whole table row: base six cells then WM six cells."""
    cells = metric_cells(base, ci) + metric_cells(wm, ci)
    return f"& {env} & " + " & ".join(cells) + r" \\"


def parse_specs(raw: list[str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for chunk in raw:
        for item in chunk.split(";"):
            item = item.strip()
            if not item:
                continue
            model, wm, path = (p.strip() for p in item.split("|", 2))
            out.append((model, wm, path))
    return out


def parse_pairs(raw: list[str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for chunk in raw:
        for item in chunk.split(";"):
            item = item.strip()
            if not item:
                continue
            model, base, wm = (p.strip() for p in item.split("|", 2))
            out.append((model, base, wm))
    return out


def env_stats(path: str) -> dict[str, dict]:
    rows = load(path)
    envs: dict[str, list[dict]] = {}
    for r in rows:
        envs.setdefault(str(r.get("Environment") or "?").strip(), []).append(r)
    out: dict[str, dict] = {}
    for env, erows in envs.items():
        if env.lower().startswith("ai2"):
            label = "AI2-THOR"
        elif env.lower().startswith("proc"):
            label = "ProcTHOR"
        else:
            label = env
        out[label] = stats(erows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="append", default=[],
                    help='"Model|WM|results.csv", ";"-separated, repeatable')
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--ci", action="store_true",
                    help="also emit the 95%% Wilson interval")
    ap.add_argument("--pair", action="append", default=[],
                    help='"Model|base_results.csv|wm_results.csv", repeatable')
    args = ap.parse_args()

    for model, base_path, wm_path in parse_pairs(args.pair):
        base_envs, wm_envs = env_stats(base_path), env_stats(wm_path)
        print(f"== {model}  base={base_path}  wm={wm_path}")
        seen: list[str] = []
        for env in ("AI2-THOR", "ProcTHOR", *sorted(set(base_envs) | set(wm_envs))):
            if env in seen or (env not in base_envs and env not in wm_envs):
                continue
            seen.append(env)
            b = base_envs.get(env) or {"n": 0}
            w = wm_envs.get(env) or {"n": 0}
            print(latex_full_row(b, w, env, args.ci))
            print(f"   base[{env}] {fmt(b)}")
            print(f"   wm  [{env}] {fmt(w)}")

    for model, wm, path in parse_specs(args.batch):
        rows = load(path)
        envs: dict[str, list[dict]] = {}
        for r in rows:
            envs.setdefault(str(r.get("Environment") or "?").strip(), []).append(r)
        print(f"== {model} | WM={wm} | {path}")
        for env, erows in sorted(envs.items()):
            if env.lower().startswith("ai2"):
                label = "AI2-THOR"
            elif env.lower().startswith("proc"):
                label = "ProcTHOR"
            else:
                label = env
            s = stats(erows)
            if args.latex:
                print(latex_half(s, label, args.ci))
            else:
                print(f"  {label:9s} {fmt(s, args.ci)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
