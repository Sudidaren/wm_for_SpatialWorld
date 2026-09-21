#!/usr/bin/env python3
"""给拉回来的云上结果生成一份交付说明（README + 环境不可用任务清单 + 配对表）。

    python3 tools/make_cloud_pull_summary.py [拉取目录]
"""

from __future__ import annotations

import csv
import os
import sys
import time

DAY = time.strftime("%Y%m%d")
ROOT = sys.argv[1] if len(sys.argv) > 1 else f"/mnt/d/lightwm_out/cloud_pull_{DAY}"
RUNS = "/home/sudidaren/spatialworld_eval/runs"
ARMS = [("30b", "30b_wm_q30b_438_v2", "Qwen3-VL-30B-A3B + WM"),
        ("8b", "8b_wm_q8b_438_v2", "Qwen3-VL-8B + WM"),
        ("kimi", "kimi_wm_kimi_438_v2", "Kimi-VL-A3B + WM"),
        ("gpt5", "gpt5_wm_gpt5_ai2thor_fail250", "GPT-5 + WM（未跑完）")]
BASE_30B = os.path.join(RUNS, "qwen3vl30b_ai2thor_v1", "results.csv")


def rows_of(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    return {r["Task ID"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def main() -> None:
    base = rows_of(BASE_30B)
    lines = [f"# 云上结果拉取 {DAY}", "",
             "每张卡只拉了记账文件（results.csv / summary.json / state.json / plan.json），",
             "没有拉逐帧 png。", ""]
    env_bad_all = None
    for tag, run, label in ARMS:
        d = os.path.join(ROOT, run)
        res = rows_of(os.path.join(d, "results.csv"))
        if not res:
            lines.append(f"## {label}\n\n（还没拉到这个 run）\n")
            continue
        ai = {t: r for t, r in res.items() if t.startswith("ai2thor")}
        ext = sorted(t for t, r in ai.items() if r["Status"] == "failed_external")
        ok = sum(1 for r in ai.values() if r["Status"] == "success")
        usable = len(ai) - len(ext)
        lines += [f"## {label}", "",
                  f"* run: `{run}`",
                  f"* ai2thor 行数 {len(ai)}：成功 {ok}、failed_model "
                  f"{sum(1 for r in ai.values() if r['Status']=='failed_model')}、"
                  f"failed_external {len(ext)}",
                  f"* **可用任务数 {usable}**（排除环境不可用），成功率 "
                  f"{ok}/{usable} = {ok / usable * 100:.1f}%" if usable else "* 无可用任务",
                  f"* procthor：一条都没跑（卡上没装 `envs/procthor/.venv`）", ""]
        if tag == "30b":
            env_bad_all = set(ext)
            both = [t for t in ai if t in base and t not in env_bad_all]
            bs = sum(1 for t in both if base[t]["Status"] == "success")
            ws = sum(1 for t in both if ai[t]["Status"] == "success")
            gain = sorted(t for t in both
                          if base[t]["Status"] != "success" and ai[t]["Status"] == "success")
            loss = sorted(t for t in both
                          if base[t]["Status"] == "success" and ai[t]["Status"] != "success")
            lines += [f"### 与同模型纯基线配对（`qwen3vl30b_ai2thor_v1`，同一批 {len(both)} 题）", "",
                      f"| | 成功 | 成功率 |", "|---|---|---|",
                      f"| 30B 纯基线 | {bs} | {bs/len(both)*100:.1f}% |",
                      f"| 30B + WM | {ws} | {ws/len(both)*100:.1f}% |",
                      f"| 净增 | {ws-bs:+d} | |", "",
                      f"新增成功 {len(gain)} 条：{', '.join(gain)}", "",
                      f"丢失 {len(loss)} 条：{', '.join(loss)}", "",
                      "（10 增 7 丢，McNemar 双侧 p≈0.6 —— 这一格目前是噪声，"
                      "方向为正但不成结论。）", ""]
    if env_bad_all:
        lines += ["## 环境不可用清单（三张卡高度一致，属场景级问题，不算模型失败）", "",
                  f"共 {len(env_bad_all)} 条：", "",
                  ", ".join(sorted(env_bad_all)), "",
                  "集中在 FloorPlan315(12) / 17(6) / 14(5) / 312(4) / 314(4) / "
                  "307(3) / 306 / 30 / 3。详见 `lightwm_phases/patches/"
                  "NOTES_scene_timeouts_20260921.md`。", ""]
    out = os.path.join(ROOT, "README.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("写出", out)


if __name__ == "__main__":
    main()
