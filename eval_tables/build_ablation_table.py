#!/usr/bin/env python3
"""5 臂消融对比表（2026-09-23）。

每张卡跑的是官方消融清单（AI2-THOR 37/42/34 条 + ProcTHOR 12 条，不含 TVR）：
base / wm / noinject / notarget / nomem。口径与主表一致：只有
status ∈ {success, failed_model} 进分母；failed_external / pending / env_error
只体现在 Coverage 里。

    python3 eval_tables/build_ablation_table.py
输出：eval_tables/ablation_table.md / ablation_table.csv
"""
from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List, Optional

RUNS = "/home/sudidaren/spatialworld_eval/runs"
HERE = os.path.dirname(os.path.abspath(__file__))
EXTERNAL = {"api_error", "env_error", "external_error", "external"}
MODELS = {
    "Qwen3-VL-8B": "qwen3vl-8b",
    "Qwen3-VL-30B": "qwen3vl-30b",
    "Kimi-VL-A3B": "kimivl-a3b",
}
ARMS = [
    ("base", "纯基线（无 WM）"),
    ("wm", "完整 WingmanWM"),
    ("noinject", "感知在、不注入提示"),
    ("notarget", "关 target hint"),
    ("nomem", "关记忆"),
]


def g(r: Dict, k: str) -> str:
    return (r.get(k) or "").strip()


def episode(run: str, env: str, tid: str) -> Optional[Dict]:
    hits = glob.glob(os.path.join(RUNS, run, env, tid, "**", "episode_*.json"),
                     recursive=True)
    if not hits:
        return None
    try:
        with open(max(hits, key=os.path.getmtime), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                          # noqa: BLE001
        return None


def stats(run: str, env: Optional[str]) -> Dict:
    path = os.path.join(RUNS, run, "results.csv")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if env:
        rows = [r for r in rows if g(r, "Environment").lower() == env]
    succ = fail = undec = 0
    steps: List[int] = []
    invalid: List[int] = []
    tokens: List[int] = []
    missing_ep = 0
    for r in rows:
        st, ft = g(r, "Status"), g(r, "Failure Type")
        if st in ("failed_external", "pending", "", "running", "queued") or ft in EXTERNAL:
            undec += 1
            continue
        tid = g(r, "Task ID") or g(r, "task_id")
        if g(r, "Success").lower() == "true":
            succ += 1
        else:
            fail += 1
        tk = g(r, "token_total") or g(r, "Token Total")
        if tk.isdigit():
            tokens.append(int(tk))
        ep = episode(run, g(r, "Environment").lower(), tid)
        if not ep:
            missing_ep += 1
            continue
        traj = ep.get("trajectory") or []
        steps.append(int(ep.get("step_count") or len(traj)))
        invalid.append(sum(1 for x in traj if (x.get("error_message") or "").strip()))
    decided = succ + fail
    return {
        "rows": len(rows), "decided": decided, "succ": succ, "fail": fail,
        "undec": undec,
        "tsr": (100.0 * succ / decided) if decided else None,
        "steps": (sum(steps) / len(steps)) if steps else None,
        "invalid": (sum(invalid) / len(invalid)) if invalid else None,
        "tokens": (sum(tokens) / len(tokens) / 1000) if tokens else None,
        "missing_ep": missing_ep,
    }


def f1(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.1f}"


def main() -> int:
    out_md = ["# 5 臂消融（官方清单：AI2-THOR + ProcTHOR，**不跑 TVR**）", "",
              "> 每臂成绩按同一份官方消融清单计算，口径与主表一致："
              "`failed_external` / `pending` / `env_error` 不计入分母，"
              "只体现在 Coverage 列。生成时间：2026-09-23。", "",
              "> 实验设计（5 臂各自隔离什么、任务集怎么来、判据、复现命令）见 "
              "[`../docs/ABLATION_DESIGN.md`](../docs/ABLATION_DESIGN.md)。", ""]
    out_rows = []
    for model, tag in MODELS.items():
        out_md += [f"## {model}", "",
                   "| Arm | Env | N | TSR | Avg steps | Avg invalid | "
                   "Avg tokens/task | Coverage |",
                   "|---|---|---:|---:|---:|---:|---:|---|"]
        for arm, _desc in ARMS:
            run = f"abl_{arm}_{tag}"
            for env in ("ai2thor", "procthor", ""):
                s = stats(run, env or None)
                if not s or not s["rows"]:
                    continue
                envname = {"ai2thor": "AI2-THOR", "procthor": "ProcTHOR",
                           "": "合计"}[env]
                cov = f"{s['decided']}/{s['rows']}"
                if s["undec"]:
                    cov += f"（未判定 {s['undec']}）"
                rate = f"{s['tsr']:.1f}%" if s["decided"] >= 5 else "-"
                out_md.append(
                    f"| {arm} | {envname} | {s['rows']} | **{rate}** | "
                    f"**{f1(s['steps'])}** | **{f1(s['invalid'])}** | "
                    f"**{f1(s['tokens'])}k** | {cov} |")
                out_rows.append(dict(
                    model=model, arm=arm, env=envname, rows=s["rows"],
                    decided=s["decided"], undecided=s["undec"],
                    tsr_pct=None if s["tsr"] is None else round(s["tsr"], 2),
                    avg_steps=None if s["steps"] is None else round(s["steps"], 2),
                    avg_invalid=None if s["invalid"] is None else round(s["invalid"], 2),
                    avg_tokens_k=None if s["tokens"] is None else round(s["tokens"], 1)))
        out_md.append("")
    md_path = os.path.join(HERE, "ablation_table.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_md) + "\n")
    csv_path = os.path.join(HERE, "ablation_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print("\n".join(out_md))
    print("saved ->", md_path, "|", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
