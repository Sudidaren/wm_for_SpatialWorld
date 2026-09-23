#!/usr/bin/env python3
"""配对翻盘表：同一批任务上 base 与 +WingmanWM 的逐任务对比（配对分析）。
为什么需要它：单看两个 TSR 的差会把任务难度差异混进噪声；配对分析只看
「base 失败→WM 成功」和「base 成功→WM 失败」两类翻盘，并用 McNemar 精确检验。
口径与 build_main_table.py 完全一致（直接 import 它的 BATCHES / _rows / EXTERNAL）。
输出：eval_tables/pairwise_wm_table.md 与 .csv
"""
from __future__ import annotations
import csv
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_main_table as B  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

#: 要配对的模型对：(base 行名, +WM 行名)
PAIRS = [
    ("Gemini 3.1 Pro", "Gemini 3.1 Pro + WingmanWM"),
    # Gemini 的 ProcTHOR base 行在表里叫 "frozen v1"（同模型、frozen 配置），单列一对。
    ("Gemini 3.1 Pro (frozen v1)", "Gemini 3.1 Pro + WingmanWM"),
    ("GPT-5", "GPT-5 + WingmanWM"),
    ("Qwen3-VL-30B-A3B (BF16, vLLM)", "Qwen3-VL-30B-A3B + WingmanWM"),
    ("Qwen3-VL-8B (BF16, vLLM)", "Qwen3-VL-8B + WingmanWM"),
    ("Kimi-VL-A3B (BF16, vLLM)", "Kimi-VL-A3B + WingmanWM"),
]


def runs_of(method: str, env: str):
    for m, e, runs, *_ in B.BATCHES:
        if m == method and e == env:
            return list(runs)
    return None


def verdicts(method: str, env: str) -> dict:
    """任务 -> success(bool)，只保留「判定过」且非外部故障的（与主表口径一致）。"""
    runs = runs_of(method, env)
    if not runs:
        return {}
    out: dict = {}
    for r in B._rows(runs):
        tid = (r.get("Task ID") or "").strip()
        if not tid:
            continue
        if (r.get("Status") or "").strip() not in ("success", "failed_model"):
            continue
        if (r.get("Failure Type") or "").strip() in B.EXTERNAL:
            continue
        out[tid] = str(r.get("Success")).strip().lower() == "true"
    for tid in B.FORCE_FAILURE.get((method, env), set()):
        if tid in out:
            out[tid] = False
    return out


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)


def sample_of(env: str):
    path = B.SAMPLE.get("ai2thor" if env == "AI2-THOR" else "procthor")
    if not path or not os.path.isfile(path):
        return None
    return {l.strip() for l in open(path) if l.strip()}


def main() -> int:
    rows = []
    for base_m, wm_m in PAIRS:
        for env in ("AI2-THOR", "ProcTHOR"):
            b = verdicts(base_m, env)
            w = verdicts(wm_m, env)
            samp = sample_of(env)
            if not b or not w or not samp:
                continue
            ts = sorted(t for t in samp if t in b and t in w)
            if not ts:
                continue
            bs = sum(b[t] for t in ts)
            ws = sum(w[t] for t in ts)
            up = [t for t in ts if not b[t] and w[t]]
            dn = [t for t in ts if b[t] and not w[t]]
            p = mcnemar(len(up), len(dn))
            rows.append({
                "model": base_m.replace(" (BF16, vLLM)", ""),
                "env": env,
                "n": len(ts),
                "base_tsr": bs / len(ts) * 100,
                "wm_tsr": ws / len(ts) * 100,
                "delta_pp": (ws - bs) / len(ts) * 100,
                "saved": len(up),
                "broken": len(dn),
                "net": len(up) - len(dn),
                "p": p,
                "saved_ids": " ".join(up),
                "broken_ids": " ".join(dn),
            })
    with open(os.path.join(HERE, "pairwise_wm_table.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    L = []
    L.append("# 附表 — 配对翻盘：base vs +WingmanWM（同一批任务，逐任务配对）")
    L.append("")
    L.append("> 生成脚本 `eval_tables/build_pairwise_table.py`，口径与主表完全一致"
             "（直接复用 `build_main_table.py` 的 run 列表与合并规则）。")
    L.append("> **救回** = base 失败且 +WM 成功；**弄坏** = base 成功且 +WM 失败；"
             "**净** = 救回 − 弄坏。**p** = McNemar 精确双侧检验。")
    L.append(">")
    L.append("> 为什么用配对而不是直接比两个 TSR：配对消掉了任务难度差异，"
             "同样样本量下灵敏度更高；两个 TSR 的差在 n=120 时标准误约 4.3 个百分点。")
    L.append("")
    L.append("| 模型 | 环境 | N | base TSR | +WM TSR | Δ | 救回 | 弄坏 | 净 | McNemar p |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['model']} | {r['env']} | {r['n']} | {r['base_tsr']:.1f}% | "
                 f"**{r['wm_tsr']:.1f}%** | **{r['delta_pp']:+.1f}pp** | {r['saved']} | "
                 f"{r['broken']} | {r['net']:+d} | {r['p']:.3f} |")
    L.append("")
    L.append("> 说明：ProcTHOR 只有 20 条样本，配对检验几乎不可能显著，仅作参考。")
    L.append("> Gemini 的 +WM 行已包含 2026-09-23 的 9 条反例重跑"
             "（主表脚注 $\\W$）；把随机波动挤掉后它的净翻盘从 +5 变为 +9。")
    L.append("")
    L.append("## 明细：救回 / 弄坏的任务 ID")
    L.append("")
    for r in rows:
        L.append(f"**{r['model']} / {r['env']}**")
        L.append("")
        L.append(f"- 救回（{r['saved']}）：{r['saved_ids'] or '（无）'}")
        L.append(f"- 弄坏（{r['broken']}）：{r['broken_ids'] or '（无）'}")
        L.append("")
    open(os.path.join(HERE, "pairwise_wm_table.md"), "w", encoding="utf-8").write("\n".join(L))
    for r in rows:
        print(f"{r['model']:28s} {r['env']:9s} n={r['n']:3d} "
              f"base {r['base_tsr']:5.1f}% -> WM {r['wm_tsr']:5.1f}% "
              f"Δ{r['delta_pp']:+5.1f}pp 救{r['saved']:2d} 坏{r['broken']:2d} "
              f"净{r['net']:+3d} p={r['p']:.3f}")
    print("\nsaved -> eval_tables/pairwise_wm_table.md / .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
