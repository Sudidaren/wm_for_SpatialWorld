"""Main evaluation table: success rate, average steps, average invalid actions.

Definitions
-----------
invalid action
    a step whose environment call came back with a non-empty ``error_message``
    (e.g. "X is not in view", "Y is blocking Agent 0", "does not exist in
    scene").  These are actions that cost a step and changed nothing.

    Taken from ``episode_*.json -> trajectory[i].error_message`` (the per-step
    record the runner already writes), NOT from the "Action failed" lines in
    ``run_stream.log`` -- those include retried attempts and would double count.

success rate
    ``success / decided`` where ``decided = success + failure`` and API /
    simulator failures (``failed_external``) are excluded; the Null count is
    reported next to it so a partial batch cannot be mistaken for a final one.

    python build_main_table.py
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

#: 2026-09-23 用户指示：`ai2thor03075` 不按冻结口径排除，而是**按失败计入分母**。
#: 理由：这条 episode 确实以失败告终（官方 wrapper 把模型合理的 `ThrowObject(...)`
#: 拼成非法参数 → controller 抛 ValueError → 官方 runner 终止 episode）。
#: 只对下面这些 (方法, 环境) 行、这些任务生效。
FORCE_FAILURE = {
    ("Qwen3-VL-30B-A3B + WingmanWM", "AI2-THOR"): {"ai2thor03075"},
}
#: 旧的三张卡下线了，卡上那两条 GPT-5 臂的 episode 没法再取；这些 run 名
#: 对应的是从卡上拉回来的 results.csv（没有 episode 目录），无效动作列会缺。
STALE_CARD_RUNS = {"main_gpt5_base_s0", "main_gpt5_base_s1", "main_gpt5_base_s2",
                   "main_gpt5_wm_s0", "main_gpt5_wm_s1", "main_gpt5_wm_s2"}

#: 我们 2026-09-21 夜切的那批闭源主样本（与 311/127 同分布的分层抽样）
SAMPLE = {
    "ai2thor": "/home/sudidaren/lightwm_phases/plans/ai2thor_main120.txt",
    "procthor": "/home/sudidaren/lightwm_phases/plans/procthor_main20.txt",
}

# 每行 = (方法, 环境, [run...], 环境目录名, planned, 是否完整, 脚注, 任务过滤)
BATCHES = [
    ("Qwen3-VL-30B-A3B (BF16, vLLM)", "AI2-THOR",
     ["qwen3vl30b_ai2thor_v1"], "ai2thor", 311, True, "", None),
    ("Qwen3-VL-30B-A3B (BF16, vLLM)", "ProcTHOR",
     ["qwen3vl30b_procthor_v1"], "procthor", 127, True, "", None),
    # ---- 自托管三臂 + WingmanWM（2026-09-21 云上批次）----
    # 注意：这三批的 ProcTHOR 是**在 WM 接入 procthor 之前**跑的，等于纯基线，
    # 用 $\aleph$ 标出来，不许当 WM 结果引用。
    ("Qwen3-VL-30B-A3B + WingmanWM", "AI2-THOR",
     ["wm_q30b_438_v2", "wmv_q30b_438_v2", "main30b_wm_ai2thor120",
      "main30b_wm_ai2thor120_fill8"], "ai2thor", 311, True, r"$\L$", None),
    ("Qwen3-VL-30B-A3B + WingmanWM", "ProcTHOR",
     ["main30b_wm_procthor20"], "procthor", 127, True, "", None),
    ("Qwen3-VL-8B + WingmanWM", "AI2-THOR",
     ["wm_q8b_438_v2", "wmv_q8b_438_v2", "main8b_wm_missing8",
      "main8b_wm_missing8_local"], "ai2thor", 311, True, r"$\L$", None),
    ("Qwen3-VL-8B + WingmanWM", "ProcTHOR",
     ["main8b_wm_procthor20"], "procthor", 127, True,
     r"$\P$", None),
    ("Kimi-VL-A3B + WingmanWM", "AI2-THOR",
     ["wm_kimi_438_v2", "wmv_kimi_438_v2", "mainkimi_wm_ai2thor120",
      "mainkimi_wm_ai2thor120_fill8"], "ai2thor", 311, True, r"$\L$", None),
    ("Kimi-VL-A3B + WingmanWM", "ProcTHOR",
     ["mainkimi_wm_procthor20"], "procthor", 127, True, "", None),
    # 8B / Kimi 的纯基线：当时只落了 state.json 没落 results.csv，2026-09-22
    # 从 wm_dev/pull_latest/*.state.json 转出来的（438 条整批）。
    ("Qwen3-VL-8B (BF16, vLLM)", "AI2-THOR",
     ["q8b_base_438_v1", "main8b_base_ai2thor120",
      "main8b_base_ai2thor120_fill7"], "ai2thor", 311, True, r"$\L$", None),
    ("Qwen3-VL-8B (BF16, vLLM)", "ProcTHOR",
     ["q8b_base_438_v1", "main8b_base_procthor20",
      "main8b_base_procthor20_fill2"], "procthor", 127, True, "", None),
    ("Kimi-VL-A3B (BF16, vLLM)", "AI2-THOR",
     ["kimi_base_438_v1", "mainkimi_base_ai2thor120_local",
      "mainkimi_base_ai2thor120_fill8"], "ai2thor", 311, True, r"$\L$", None),
    ("Kimi-VL-A3B (BF16, vLLM)", "ProcTHOR",
     ["kimi_base_438_v1", "mainkimi_base_procthor20"], "procthor", 127, True, "", None),
    # ---- WM v1（2026-09-16/17 那批：target_hint 与 state_check 全关，
    #      测的是"只给记忆、不给提示"的底数；与 v2 是不同方法变体）----
    ("Qwen3-VL-30B-A3B + WingmanWM v1", "AI2-THOR",
     ["wmv1_q30b_a311"], "ai2thor", 311, True, r"$\S$", None),
    ("Qwen3-VL-8B + WingmanWM v1", "AI2-THOR",
     ["wmv1_q8b_a311"], "ai2thor", 311, True, r"$\S$", None),
    ("Kimi-VL-A3B + WingmanWM v1", "AI2-THOR",
     ["wmv1_kimi_a311"], "ai2thor", 311, True, r"$\S$", None),
    # Complete 311-task AI2-THOR numbers for Gemini: the legacy trajectories
    # replayed through the fixed verifier (the original verdicts were broken).
    ("Gemini 3.1 Pro", "AI2-THOR",
     ["replay_legacy311_v1"], "ai2thor", 311, True, "", None),
    ("Gemini 3.1 Pro (frozen v1)", "ProcTHOR",
     ["gemini31pro_procthor127_frozen_v1"], "procthor", 127, False, r"$\dagger$", None),
    # ---- 2026-09-21 夜：闭源主批次（120 + 20 分层样本）----
    ("Gemini 3.1 Pro + WingmanWM", "AI2-THOR",
     ["wm_gemini31pro_fix40", "wm_gemini31pro_fix_rest156", "closed_gemini_wm"],
     "ai2thor", 120, True, "", "ai2thor"),
    ("Gemini 3.1 Pro + WingmanWM", "ProcTHOR",
     ["closed_gemini_wm"], "procthor", 20, True, "", "procthor"),
    ("GPT-5", "AI2-THOR",
     ["main_gpt5_base_s0", "main_gpt5_base_s1", "main_gpt5_base_s2",
      "closed_gpt5_base_fill7"],
     "ai2thor", 120, True, r"$\star$", "ai2thor"),
    ("GPT-5", "ProcTHOR",
     ["main_gpt5_base_s0", "main_gpt5_base_s1", "main_gpt5_base_s2",
      "closed_gpt5_base_procthor"],
     "procthor", 20, True, r"$\star$", "procthor"),
    ("GPT-5 + WingmanWM", "AI2-THOR",
     ["main_gpt5_wm_s0", "main_gpt5_wm_s1", "main_gpt5_wm_s2", "closed_gpt5_wm",
      "closed_gpt5_wm_fill2"],
     "ai2thor", 120, True, r"$\ddagger$", "ai2thor"),
    ("GPT-5 + WingmanWM", "ProcTHOR",
     ["main_gpt5_wm_procthor", "closed_gpt5_wm_procthor"],
     "procthor", 20, True, r"$\ddagger$", "procthor"),
]


def _allowed(kind: Optional[str]) -> Optional[set]:
    if not kind:
        return None
    path = SAMPLE[kind]
    if not os.path.isfile(path):
        return None
    return {l.strip() for l in open(path) if l.strip()}


def _rows(runs) -> List[Dict]:
    """读一个或多个 run 的 results.csv；同名任务后者覆盖前者（用于拼接分片）。

    2026-09-23 补充：**已判定的记录优先于未判定的记录**。补跑批次被中途打断时，
    会把"还没跑完"的任务写成 failed_external/pending；如果按纯"后者覆盖"合并，
    这些未判定行会盖掉主批次里已经判定好的结果（实测踩到：Kimi+WM 的 05515）。
    """
    if isinstance(runs, str):
        runs = [runs]
    merged: Dict[str, Dict] = {}
    decided: Dict[str, bool] = {}
    for run in runs:
        p = os.path.join(RUNS, run, "results.csv")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                tid = _g(r, "Task ID") or _g(r, "task_id")
                if tid:
                    is_decided = (_g(r, "Status") in ("success", "failed_model")
                                  and _g(r, "Failure Type") not in EXTERNAL)
                    if tid in merged and decided.get(tid) and not is_decided:
                        continue
                    merged[tid] = r
                    decided[tid] = is_decided
    return list(merged.values())


def _g(r: Dict, k: str) -> str:
    return (r.get(k) or "").strip()


def _episode(runs, env: str, task_id: str) -> Optional[Dict]:
    # Layout differs per backend: procthor is
    #   <run>/procthor/<task>/worker_N/episode_*.json
    # while ai2thor nests one more level:
    #   <run>/ai2thor/<task>/worker_N/<task>/episode_*.json
    #: 2026-09-23 修正：原来按 mtime 取"最新"，但 `cp` 会把 mtime 重置成拷贝时刻，
    #: 导致合并多个 run 后随机选中残档（实测：GPT-5 ProcTHOR 选中了 1 步的 stub，
    #: steps 显示 1.0）。改成**按 run 列表顺序取**，与 results.csv 的"后者优先"一致。
    if isinstance(runs, str):
        runs = [runs]
    for run in reversed(runs):
        hits = glob.glob(os.path.join(RUNS, run, env, task_id, "**",
                                      "episode_*.json"), recursive=True)
        if not hits:
            continue
        try:
            with open(max(hits, key=os.path.getmtime), encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            continue
    return None


def stats(runs, env: str, planned: int, allowed: Optional[set] = None,
          force_fail: Optional[set] = None) -> Dict:
    rows = _rows(runs)
    #: 有些 run 的 results.csv 同时含 ai2thor 与 procthor（例如云上那三批
    #: `wmv_*_438_v2`），必须按 Environment 列过滤，否则两个环境会互相污染。
    rows = [r for r in rows if _g(r, "Environment").lower() == env]
    if allowed is not None:
        rows = [r for r in rows
                if (_g(r, "Task ID") or _g(r, "task_id")) in allowed]
    succ, null, fail = [], [], []
    for r in rows:
        st, ft = _g(r, "Status"), _g(r, "Failure Type")
        tid = _g(r, "Task ID") or _g(r, "task_id")
        #: 用户指定：这些任务即使被官方记成 env_error，也按"失败"计入分母。
        if force_fail and tid in force_fail and st in ("failed_model", "failed_external"):
            fail.append(r)
            continue
        #: 只有 success / failed_model 才是"判定过"。`pending`（卡中途下线，
        #: 任务没跑完）和空 Status 都算没判定 —— 2026-09-22 核查时发现旧逻辑
        #: 把 7 条 pending 当成了失败，把 GPT-5 基线从 19.5% 压到 18.3%。
        if st in ("failed_external", "pending", "", "running", "queued") or ft in EXTERNAL:
            null.append(r)
            continue
        if _g(r, "Success").lower() == "true":
            succ.append(r)
        elif _g(r, "Success").lower() == "false" or st == "failed_model":
            fail.append(r)
        else:
            null.append(r)
    decided = succ + fail
    steps: List[int] = []
    invalid: List[int] = []
    tokens: List[int] = []
    missing = 0
    for r in decided:
        tid = _g(r, "Task ID") or _g(r, "task_id")
        #: 平均任务花费 token（口径见 2026-09-16 那份快照）：优先整数字段
        #: token_total，没有就 prompt+completion；只统计进分母（decided）的任务。
        tk = _g(r, "token_total") or _g(r, "Token Total") or _g(r, "Total Tokens")
        if tk.isdigit():
            tokens.append(int(tk))
        else:
            p = _g(r, "Prompt Tokens") or _g(r, "prompt_tokens")
            c = _g(r, "Completion Tokens") or _g(r, "completion_tokens")
            if p.isdigit() or c.isdigit():
                tokens.append(int(p or 0) + int(c or 0))
        ep = _episode(runs, env, tid)
        if not ep:
            missing += 1
            s = _g(r, "Actual Steps")
            if s.isdigit():
                steps.append(int(s))
            continue
        traj = ep.get("trajectory") or []
        steps.append(int(ep.get("step_count") or len(traj)))
        invalid.append(sum(1 for x in traj
                           if (x.get("error_message") or "").strip()))
    return {
        "run": runs if isinstance(runs, str) else "+".join(runs),
        "planned": planned, "seen": len(rows),
        "success": len(succ), "failure": len(fail), "null": len(null),
        "decided": len(decided),
        "success_rate": (len(succ) / len(decided)) if decided else None,
        "avg_steps": (sum(steps) / len(steps)) if steps else None,
        "avg_tokens": (sum(tokens) / len(tokens)) if tokens else None,
        "n_tokens": len(tokens),
        "avg_invalid": (sum(invalid) / len(invalid)) if invalid else None,
        "n_invalid_measured": len(invalid), "missing_episode": missing,
        "invalid_rate": ((sum(invalid) / len(invalid)) /
                         (sum(steps) / len(steps)))
        if invalid and steps and sum(steps) else None,
        "median_steps": (sorted(steps)[len(steps) // 2] if steps else None),
    }


def f1(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.1f}"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("sample", "full"), default="sample")
    args = ap.parse_args()
    mode = args.mode
    L: List[str] = [
        ("# 主表 — 统一样本（AI2-THOR 120 + ProcTHOR 20）" if mode == "sample"
         else "# 附表 — 各模型实际跑过的完整批次"), ""]
    L.append("> **无效动作** = 该步环境返回非空 `error_message`（如 `X is not in view`、"
             "`Y is blocking Agent 0`、`does not exist in scene`）——花了步数但什么也没改变。")
    L.append("> 取自 `episode_*.json → trajectory[i].error_message`，"
             "不是 `run_stream.log` 里的 `Action failed`（那里含重试，会重复计数）。")
    L.append("> **成功率（TSR）** = success / (success+failure)。不进分母的有三类："
             "`Status=failed_external`（API/模拟器故障）、`Status=pending`（批次没跑完）、"
             "以及 `Failure Type ∈ {api_error, env_error, external_error, external}` —— "
             "最后一类是历史口径（少数 `failed_model` 的失败原因其实是环境异常，"
             "例如模型传了非法参数导致模拟器报错），一律按未判定处理。"
             "三者都只体现在 Coverage 列里。")
    L.append("> **Avg tokens/task** = 该行进入分母的那些任务的平均总 token"
             "（`token_total`）——即跑一条任务平均要花多少 token 的直接口径。")
    L.append("")
    L.append("| Method | Env | N | **TSR** | **Avg steps** | "
             "**Avg invalid actions** | **Avg tokens/task** | Coverage |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|")
    data = []
    for method, env, runs, envdir, planned, complete, marker, filt in BATCHES:
        if mode == "sample":
            #: 统一样本模式：无视各行自己的过滤设置，一律取 120/20 那套共同样本，
            #: N 也随之变成样本规模，这样跨模型可以直接比。
            filt = "ai2thor" if envdir == "ai2thor" else "procthor"
            planned = 120 if envdir == "ai2thor" else 20
        else:
            filt = None
            #: 全量表：N 用该环境的全集规模，Coverage 才表示"实际跑到多少"。
            planned = 311 if envdir == "ai2thor" else 127
        s = stats(runs, envdir, planned, _allowed(filt),
                  FORCE_FAILURE.get((method, env)))
        data.append((method, env, s))
        cov = f"{s['decided']}/{planned}"
        if not complete:
            cov += " ⚠️partial"
        if s["missing_episode"]:
            cov += f" · ep{s['n_invalid_measured']}"
        rate = "-" if s["success_rate"] is None else f"{s['success_rate'] * 100:.1f}%"
        ishare = ("-" if s["invalid_rate"] is None
                  else f"{s['invalid_rate'] * 100:.0f}%")
        tok = "-" if s["avg_tokens"] is None else f"{s['avg_tokens'] / 1000:.0f}k"
        #: 覆盖不到一半的行，指标全是噪声（例：GPT-5+WM 的 procthor 当时只判了 2/20），
        #: 一律打 — ，等跑完再填。
        #: 判定数不到 10 条的行，指标全是噪声（例：GPT-5+WM 的 procthor 只判了 2 条），
        #: 一律打 — ，等跑完再填；Coverage 列仍然照实显示进度。
        if s["decided"] < 10:
            rate = tok = "-"
            steps_s = inv_s = "-"
        else:
            steps_s, inv_s = f1(s["avg_steps"]), f1(s["avg_invalid"])
        L.append(f"| {method} {marker} | {env} | {planned} | "
                 f"**{rate}** | **{steps_s}** | **{inv_s}** | "
                 f"**{tok}** | {cov} |")
    L.append("")
    L.append("> 脚注：`⚠️partial` = 批次没跑完或该环境还没跑；`· epN` = 该行只有 N 条"
             "能拿到 `episode_*.json`（旧卡下线，卡上两条 GPT-5 臂的 episode 取不回来），"
             "**无效动作**列只在有 episode 的样本上算。")
    L.append("")
    if mode == "sample":
        L.append("> `Gemini 3.1 Pro + WingmanWM` / `GPT-5` / `GPT-5 + WingmanWM` 这三组的"
                 "**原批次**跑的是 120 + 20 的分层抽样（与 311/127 同分布）；本表里所有"
                 "模型都被限制到这同一套样本上，N 就是样本规模，Coverage 是实际判定数。")
    L.append("")
    L.append("> $\\aleph$ = **该批 ProcTHOR 没有接入 WM**（2026-09-21 之前 procthor 的 agent "
             "循环里根本没有 MemoryProbe，当天修的正是这个）——这几行的数值等同纯基线，"
             "**不得当作 WM 结果引用**，重跑未做。")
    L.append(">")
    L.append("> $\\S$ = **WingmanWM v1**（2026-09-16/17 那批，`WM_TARGET_HINT` 与 "
             "`WM_STATE_CHECK` 全关）——测的是「只给记忆、不给提示」的底数，"
             "与上面不带标记的 v2（两道通道全开）是**不同方法变体**，不能混着比。")
    L.append(">")
    L.append("> $\\P$ = **2026-09-22 重跑**（run `main8b_wm_procthor20`，卡 "
             "`connect.bjb1.seetacloud.com:25766`）——原 $\\aleph$ 那格在 09-21 修好 "
             "ProcTHOR 的 WM 接入之前跑，等于纯基线；本次 WM 感知栈（RF-DETR + DA2 深度）"
             "改在 GPU 上跑，同一套 20 条样本、同一 BF16 权重、同一注入参数"
             "（`WM_TARGET_HINT=1`、`WM_STATE_CHECK=1`）。结果 20/20 判定、"
             "`episode_*.json` 20/20，所以 **Avg invalid actions 首次可填（2.40）**；"
             "步数 22.2→38.1、token 277k→592k 即 WM 真正在注入提示的证据。来源："
             "`spatialworld_eval/runs/main8b_wm_procthor20`（2026-09-22 20:40 完成，rc=0）。")
    L.append(">")
    L.append("> $\\L$ = **2026-09-23 本地渲染 + 云端 vLLM 补跑**（§8.4/§9.4）。"
             "有 7 条 AI2-THOR 任务（`ai2thor05022/05024/05028/05029/05515/05519/05521`）"
             "在云端每条臂上都会卡死在第一个 `step`（`pending`、attempts=3），"
             "另有 `ai2thor03075` 记为 env_error；这几条改在**本机渲染**"
             "（AI2-THOR Linux64 + `DISPLAY=:0`）、**模型仍走云端 vLLM**"
             "（`BASE_URL=http://127.0.0.1:1800x/v1`，隧道直连对应卡）跑，"
             "任务集、BF16 权重、注入参数（`WM_TARGET_HINT=1`、`WM_STATE_CHECK=1`、"
             "`LIGHTWM_DEPTH_SOURCE=da2`）与主表一致。run："
             "`main8b_wm_missing8_local`、`main8b_base_ai2thor120_fill7`、"
             "`mainkimi_base_ai2thor120_fill8`、`main30b_wm_ai2thor120_fill8`、"
             "`mainkimi_wm_ai2thor120_fill8`（2026-09-23）。")
    if mode == "sample":
        L.append(">")
        L.append("> $\\G$ = **已知缺口（2026-09-23 收尾）**："
                 "`Qwen3-VL-30B-A3B + WingmanWM` AI2-THOR = 119/120："
                 "`ai2thor03075`（指令为 *throw the apple into the trash can*，"
                 "gold 路径是 `PutObject(GarbageCan)`）在该臂上让模型选择了动作 "
                 "`ThrowObject(Apple)`；**官方 wrapper** 会把它拼成 "
                 "`{action: \"ThrowObject\", objectId: <id>, moveMagnitude: 150}`，"
                 "而 AI2-THOR 的 `ThrowObject` 只接受 `moveMagnitude`/`forceAction`"
                 "（不接受 `objectId`），于是 `controller.step` 抛 `ValueError`，"
                 "**官方 runner**（上游 init 提交）的 `except Exception` 再把它写成 "
                 "`Environment exception: Action \"ThrowObject\" called with invalid "
                 "argument: 'objectId'` 并 `should_continue=False` **终止该 episode**，"
                 "该 episode 确实以失败告终，因此**按用户指示按「失败」计入分母**"
                 "（不按冻结口径排除；本表只此一处这样处理）。"
                 "另：`WingmanWM v1` 三行、Gemini 两行的缺口见 `· ep0` 与 "
                 "`⚠️partial`：属 2026-09-16/17 与 09-21 的历史批次遗留，"
                 "要补必须按各自配置重跑。")
    if mode == "sample":
        L.append(">")
        L.append("> **本表所有模型都只取同一套共同样本**（AI2-THOR 120 / ProcTHOR 20，"
                 "分层抽样、与 311/127 同分布），这样跨模型可以直接比。"
                 "各模型自己跑过的完整批次见 `main_table_full.md`。")
    else:
        L.append(">")
        L.append("> 这一张是**各模型真实跑过的全量**（311 / 127 / 438），"
                 "不同模型的 N 和覆盖率都不一样——跨模型比之前先看 Coverage 列。"
                 "统一样本的主表见 `main_table.md`。")
    if mode == "full":
        L.append(">")
        L.append("> Gemini + WM 那一行的 Coverage 只算「**最终版配置**」"
                 "（`fix40` ∪ `fix_rest156`）。把 9/19 起所有 WM 版本并起来是 "
                 "**266/311**，但混了 `nohint` 等不同配置，**没有合并进这一行**；"
                 "procthor 侧只有 3 条（`probe6`），等于没测。")
    out = os.path.join(HERE, "main_table.md" if mode == "sample" else "main_table_full.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    for method, env, s in data:
        print(f"# {method} {env}: episodes with trajectory = "
              f"{s['n_invalid_measured']}/{s['decided']} "
              f"(missing {s['missing_episode']})")
    print("\nsaved ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
