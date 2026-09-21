#!/usr/bin/env python3
"""把闭源主批次（ai2thor 120 + ProCTHOR 20）拼成一张配对表。

数据来源（全部按 run 目录读，不改动任何结果文件）：

  ai2thor 基线         spatialworld_eval/runs/replay_legacy311_v1          (311 条，重放重判)
  ai2thor Gemini+WM    runs/wm_gemini31pro_fix40 + wm_gemini31pro_fix_rest156 (168 条，"最终版")
                       + runs/closed_gemini_wm                               (本次补跑)
  ai2thor GPT-5 基线   三张卡的 runs/main_gpt5_base_s0/1/2                 (本次)
  ai2thor GPT-5+WM    三张卡的 runs/main_gpt5_wm_s0/1/2                   (本次)
  ProCTHOR 基线        runs/gemini31pro_procthor127_frozen_v1              (127 条)
  ProCTHOR Gemini+WM   runs/closed_gemini_wm
  ProCTHOR GPT-5 基线  三张卡的分片
  ProCTHOR GPT-5+WM   三张卡的分片

输出：lightwm_phases/plans/closed_table.csv + 控制台摘要。
用法：python3 build_closed_table.py [--pull]   加 --pull 会先把卡上的结果拉回本地。
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

TOOLS = "/home/sudidaren/lightwm_phases/tools"
SWE = "/home/sudidaren/spatialworld_eval/runs"
PLANS = "/home/sudidaren/lightwm_phases/plans"
PULLED = "/mnt/d/lightwm_out/closed_pull"
SHARDS = {"30b": 0, "8b": 1, "kimi": 2}

sys.path.insert(0, TOOLS)


def load(run: str, name: str | None = None) -> dict[str, dict]:
    """读一个 run 的 results.csv，键是 task_id。后读的覆盖先读的。"""
    path = os.path.join(run, "results.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        tid = r.get("Task ID")
        if tid:
            out[tid] = r
    return out


def merged(*runs: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for run in runs:
        out.update(load(run))
    return out


def ok(r: dict | None) -> bool | None:
    if not r:
        return None
    return str(r.get("Success")).lower() == "true"


def pull() -> None:
    """把三张卡的两条臂的分片结果拉回本地。"""
    os.makedirs(PULLED, exist_ok=True)
    cd = f"cd {TOOLS}"
    for tag, i in SHARDS.items():
        for arm in ("base", "wm"):
            dst = f"{PULLED}/{tag}_main_gpt5_{arm}_s{i}"
            os.makedirs(dst, exist_ok=True)
            for fname in ("results.csv", "state.json", "summary.json", "plan.json"):
                src = f"/home/sudidaren/spatialworld_eval/runs/main_gpt5_{arm}_s{i}/{fname}"
                subprocess.run(f"{cd} && NEWCARD={tag} python3 newcard.py get {src} {dst}/{fname}",
                               shell=True, capture_output=True, timeout=300)
            print(f"  拉回 {tag} {arm} -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    args = ap.parse_args()
    if args.pull:
        print("拉云上结果…")
        pull()

    ai_main = [l.strip() for l in open(f"{PLANS}/ai2thor_main120.txt") if l.strip()]
    pr_main = [l.strip() for l in open(f"{PLANS}/procthor_main20.txt") if l.strip()]

    # 基线
    base = merged(f"{SWE}/replay_legacy311_v1", f"{SWE}/gemini31pro_procthor127_frozen_v1")
    # Gemini+WM：最终版 + 本次补跑
    gem_wm = merged(f"{SWE}/wm_gemini31pro_fix40", f"{SWE}/wm_gemini31pro_fix_rest156",
                    f"{SWE}/closed_gemini_wm")
    # GPT-5 两条臂：三张卡的本地拉回副本
    gpt_base: dict[str, dict] = {}
    gpt_wm: dict[str, dict] = {}
    for tag, i in SHARDS.items():
        gpt_base.update(load(f"{PULLED}/{tag}_main_gpt5_base_s{i}"))
        gpt_wm.update(load(f"{PULLED}/{tag}_main_gpt5_wm_s{i}"))
    if not gpt_base:
        print("!! 还没有拉回 GPT-5 结果，先跑 --pull")

    rows = []
    for env, ids in (("ai2thor", ai_main), ("procthor", pr_main)):
        for tid in ids:
            b = ok(base.get(tid))
            gw = ok(gem_wm.get(tid))
            gb = ok(gpt_base.get(tid))
            gpw = ok(gpt_wm.get(tid))
            rows.append(dict(
                env=env, task_id=tid,
                base=("" if b is None else b),
                gemini_wm=("" if gw is None else gw),
                gpt5_base=("" if gb is None else gb),
                gpt5_wm=("" if gpw is None else gpw),
                gemini_delta=("" if (b is None or gw is None) else (gw and not b)),
                gpt5_delta=("" if (gb is None or gpw is None) else (gpw and not gb)),
            ))

    out = f"{PLANS}/closed_table.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["env", "task_id", "base", "gemini_wm",
                                           "gpt5_base", "gpt5_wm",
                                           "gemini_delta", "gpt5_delta"])
        w.writeheader()
        w.writerows(rows)

    have = lambda k: sum(1 for r in rows if r[k] != "")
    cnt = lambda k: sum(1 for r in rows if r[k] is True)
    print(f"\n写到 {out}")
    print(f"  共 {len(rows)} 条（ai2thor {len(ai_main)} / procthor {len(pr_main)}）")
    print(f"  有 Gemini 基线 {have('base')}，基线成功 {cnt('base')}")
    print(f"  有 Gemini+WM  {have('gemini_wm')}，成功 {cnt('gemini_wm')}")
    print(f"  有 GPT-5 基线 {have('gpt5_base')}，成功 {cnt('gpt5_base')}")
    print(f"  有 GPT-5+WM  {have('gpt5_wm')}，成功 {cnt('gpt5_wm')}")
    paired = [r for r in rows if r["base"] != "" and r["gemini_wm"] != ""]
    if paired:
        b = sum(1 for r in paired if r["base"] is True)
        g = sum(1 for r in paired if r["gemini_wm"] is True)
        print(f"\n  Gemini 配对 {len(paired)} 条：基线 {b} ({100*b/len(paired):.1f}%)"
              f" → WM {g} ({100*g/len(paired):.1f}%)  Δ{g-b:+d}")
    paired5 = [r for r in rows if r["gpt5_base"] != "" and r["gpt5_wm"] != ""]
    if paired5:
        b = sum(1 for r in paired5 if r["gpt5_base"] is True)
        g = sum(1 for r in paired5 if r["gpt5_wm"] is True)
        print(f"  GPT-5  配对 {len(paired5)} 条：基线 {b} ({100*b/len(paired5):.1f}%)"
              f" → WM {g} ({100*g/len(paired5):.1f}%)  Δ{g-b:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
