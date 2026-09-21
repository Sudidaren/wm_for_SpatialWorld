#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 255 条任务的判定从"位姿精确匹配"换成"目标锚点都在画面里"。

为什么换：官方位姿口径是 0.01 m / 1°，而动作粒度是 0.25 m / 45° / 30°，
等于要求落在同一格、同一角度 —— 差一次 LookDown 就是 0 分（实测 GPT-5 在
iTHOR 5 条上最近的一次是 0.35 m / 0° / 30°，全判 0）。改用"目标位姿能看到
哪些可判别锚点、最终画面里是否也都在"这个口径，锚点清单是生成任务时就用
同口径量好的（audit_anchors.py，剔除硬结构面 + 软装饰，掩码 >=0.5%）。

做法（可回滚）：
  * 原来的 `success_conditions` 原样存进 `tvr.pose_success_condition`；
  * `success_conditions` 换成一条 `objects_visible`；
  * 锚点清单来自 `tasks/final_ladder.csv` 的 `anchors` 列（strict 口径）。

用法：
  python3 apply_visibility_condition.py --dry-run     # 先看会改什么
  python3 apply_visibility_condition.py               # 真改（自动 .bak.json）
  python3 apply_visibility_condition.py --revert      # 从 .bak.json 回滚
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks"
LADDER = TASKS / "final_ladder.csv"
MODE = "presence"          # presence: 掩码>0；area1pct: 每个锚点占画面 >=1%


def load_anchors() -> dict:
    out = {}
    with LADDER.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            anchors = [a.strip() for a in (row.get("anchors") or "").split(";") if a.strip()]
            out[row["task_id"]] = anchors
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--mode", default=MODE, choices=["presence", "area1pct"])
    args = ap.parse_args()

    anchors_by_task = load_anchors()
    n_changed = n_skipped = 0
    for d in sorted(TASKS.iterdir()):
        tj = d / "task.json"
        if not tj.is_file():
            continue
        bak = d / "task.json.bak.json"
        if args.revert:
            if bak.is_file():
                shutil.copyfile(bak, tj)
                n_changed += 1
            continue
        task = json.loads(tj.read_text(encoding="utf-8"))
        tid = task["task_id"]
        anchors = anchors_by_task.get(tid)
        if not anchors:
            n_skipped += 1
            print(f"  跳过（没有锚点清单）: {tid}")
            continue
        conds = task.get("success_conditions") or []
        if len(conds) == 1 and conds[0].get("type") == "objects_visible" \
                and conds[0].get("anchors") == anchors:
            continue
        if args.dry_run:
            print(f"  {tid}: {[c.get('type') for c in conds]} -> objects_visible({len(anchors)}) {anchors[:4]}…")
            n_changed += 1
            continue
        if not bak.is_file():
            shutil.copyfile(tj, bak)          # 只备份一次，保留最初的版本
        task.setdefault("tvr", {})
        if conds:
            task["tvr"]["pose_success_condition"] = conds[0]
        task["success_conditions"] = [{
            "type": "objects_visible",
            "anchors": anchors,
            "mode": args.mode,
            "min_area": 0.01,
            "require_stop": False,
        }]
        task["success_logic"] = "AND"
        task["judging"] = {
            "criterion": "objects_visible",
            "anchors_from": "audit_anchors.py (strict: 剔硬结构面+软装饰, 掩码>=0.5%)",
            "pose_criterion_kept_as": "tvr.pose_success_condition（参考指标，不判成败）",
            "changed_at": "2026-09-21",
        }
        tj.write_text(json.dumps(task, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        n_changed += 1

    print(f"{'（dry-run）' if args.dry_run else ''}改动 {n_changed} 条，跳过 {n_skipped} 条，mode={args.mode}")


if __name__ == "__main__":
    main()
