# Evaluation Section — Draft (tables with `-` placeholders)

> 已确认的评测决策（2026-09-12）：
> 1. 评测集 = **家庭任务，仅单 agent** = AI2-THOR 311 + ProcTHOR 127 +
>    VirtualHome 38 = **476**；
> 2. 步数效率 SE 报两列：`SE_success` 与 `SE_all`，定义
>    `SE = 1 − steps / max_steps`；
> 3. 失败/无结果规则：`null`（API/env 故障）不计入 TSR 分母、可重试；
>    论文同时报告 `null rate` 与 `completion rate`；
> 4. 每个模型单次运行（single run，temperature = 1.0）。
>
> 所有性能数字在跑之前一律为 `-`。

---

## 5.1 Evaluation Set

We evaluate on the household subset of SpatialWorld under the **single-agent
protocol**, i.e. 476 tasks from three backends: AI2-THOR (311), ProcTHOR (127),
and VirtualHome (38). Multi-agent tasks (46), CARLA (80), EmbodiedCity (53),
and Game (105) are out of scope. The final task list is frozen as
`eval_set_v1.json` (SHA256 `54626ae4…7227`), with one entry per task
(task id, backend, category, task type, scene, golden action count, target
objects, `task.json` hash).

| Backend | Protocol | Tasks | Included | Excluded (reason) |
|---|---|---:|---|---|
| AI2-THOR | single-agent | 311 | yes | – |
| ProcTHOR | single-agent | 127 | yes | – |
| VirtualHome | single-agent | 38 | yes | – |
| AI2-THOR / ProcTHOR | multi-agent | 46 | no | different (dual) protocol |
| CARLA | single-agent | 80 | no | out of household scope |
| EmbodiedCity | single-agent | 53 | no | out of household scope |
| Game | single-agent | 105 | no | out of household scope |
| **Total** | – | **760 records / 724 unique ids** | **476** | – |

Task composition of the frozen set (static counts from the official
classification file):

| Backend | Daily | Work | Entertain | Travel | Social | Navigation | Interaction | Hybrid | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AI2-THOR | 219 | 41 | 40 | 11 | 0 | 0 | 31 | 280 | 311 |
| ProcTHOR | 92 | 10 | 23 | 2 | 0 | 6 | 0 | 121 | 127 |
| VirtualHome | 27 | 8 | 3 | 0 | 0 | 0 | 4 | 34 | 38 |
| **Total** | 338 | 59 | 66 | 13 | 0 | 6 | 35 | 435 | 476 |

Split integrity: the evaluation rooms/houses do not appear in the training
pool; the room-level holdout lists are given in Appendix A (`-` until the
training split is frozen).

## 5.2 Environments and Protocol

| Backend | Simulator / runtime | Assets or dataset revision | Observation | Action interface | Step budget | Determinism | Smoke check |
|---|---|---|---|---|---|---|---|
| AI2-THOR | - | - | 800×600 RGB, FOV 60° | unified text actions | 10 + 2·g | - | - |
| ProcTHOR | - | - | 800×600 RGB, FOV 60° | unified text actions | 10 + 2·g | - | - |
| VirtualHome | - | - | 640×480 RGB, FOV 60° | unified text actions | 10 + 2·g | - | - |

Hardware / software used for the run:

| Item | Value |
|---|---|
| GPU / driver | - |
| OS / display mode | - |
| Per-env Python venv | - |
| SpatialWorld commit | - |
| LightWM repo commit | - |
| Model gateway / model snapshot | - |
| Temperature / context window / token cap | - |
| API timeout / retries | 120 s / 2 |

## 5.3 Metrics

- **Success / Failure / Null.** A task is `success` if the official
  terminal-state verifier returns 1.0 after the agent issues `EndTask(DONE)`;
  `failure` if the model fails (wrong DONE, max steps, consecutive action
  failures, parse/action errors) or explicitly ends with `FAIL`; `null` if the
  episode could not be evaluated (API or simulator failure) after 2 retries.
- **TSR (official):** `success / (success + failure)` — `null` excluded from
  the denominator.
- **Null rate:** `null / total`, reported separately.
- **Completion rate:** `success / total` (conservative, null counted).
- **Steps:** official `step_count` (excluding initialization actions);
  reported separately for successful and failed episodes.
- **SE_success:** `mean over successful tasks of (1 − steps / max_steps)`.
- **SE_all:** `mean over success + failure of (1 − steps / max_steps)`
  (failed episodes that hit the budget contribute 0).

## 5.4 Main Results

### Table 5.4.1 Model × environment

| Method | Env | Success | Failure | Null | Total | TSR | Null rate | Completion rate | SE_success | SE_all | Avg steps (succ) | Avg steps (fail) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5 | AI2-THOR | - | - | - | 311 | - | - | - | - | - | - | - |
| GPT-5 | ProcTHOR | - | - | - | 127 | - | - | - | - | - | - | - |
| GPT-5 | VirtualHome | - | - | - | 38 | - | - | - | - | - | - | - |
| GPT-5 | Overall | - | - | - | 476 | - | - | - | - | - | - | - |
| Gemini | Overall | - | - | - | 476 | - | - | - | - | - | - | - |
| Qwen | Overall | - | - | - | 476 | - | - | - | - | - | - | - |
| GPT-5 + WingmanWM | AI2-THOR | - | - | - | 311 | - | - | - | - | - | - | - |
| GPT-5 + WingmanWM | ProcTHOR | - | - | - | 127 | - | - | - | - | - | - | - |
| GPT-5 + WingmanWM | Overall (438) | - | - | - | 438 | - | - | - | - | - | - | - |
| DreamerV3 | Overall | - | - | - | 476 | - | - | - | - | - | - | - |
| DIAMOND | Overall | - | - | - | 476 | - | - | - | - | - | - | - |

> Note: WingmanWM requires metric depth and is therefore evaluated on
> AI2-THOR + ProcTHOR (438) only; VirtualHome is LLM-only.

### Table 5.4.2 Category breakdown (TSR)

| Method | Daily | Work | Entertain | Travel | Social | Overall |
|---|---:|---:|---:|---:|---:|---:|
| - | - | - | - | - | - | - |

### Table 5.4.3 Failure taxonomy (counts)

| Method | Env | DONE-but-wrong | Max steps | Consecutive action failures | Parse/action error | Model FAIL | API/env null | Other |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| - | - | - | - | - | - | - | - | - |

### Table 5.4.4 Agent variants / ablation

| Variant | Memory | Gate | AI2-THOR TSR / SE | ProcTHOR TSR / SE | VirtualHome TSR / SE |
|---|---|---|---|---|---|
| LLM only | - | - | - | - | - |
| + WingmanWM (perception + memory) | - | - | - | - | - |
| + WingmanWM (with gate) | - | TBD | - | - | - |

### Table 5.4.5 Cost and runtime

| Method | Tokens / task | API calls / task | Wall-clock / task | Simulator hours | GPU hours |
|---|---:|---:|---:|---:|---:|
| - | - | - | - | - | - |

## 5.5 Reproducibility

| Item | Value |
|---|---|
| Evaluation set manifest | `eval_set_v1.json` |
| Evaluation set SHA256 | `54626ae4ae859c7178a6e1481da201a8f1a2573c2963e05b94433dc4f3037227` |
| SpatialWorld commit | - |
| LightWM / pipeline commit | - |
| Run date / instance | - |
| Retries / timeout | 2 / 120 s |
| Runs per model | 1 |

## Appendix A — Per-task results

| task_id | env | category | task_type | scene | success | steps | golden steps | SE | failure reason | tokens | log path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| - | - | - | - | - | - | - | - | - | - | - | - |

