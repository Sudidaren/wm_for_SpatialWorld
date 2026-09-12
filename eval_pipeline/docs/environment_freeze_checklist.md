# Environment Freeze Checklist (values `-` until confirmed)

> 目的：在批量评测前，把"评测集 + 环境"冻结成两份可核对的文档。
> 本文件是环境部分；评测集部分是 `eval_sets/eval_set_v1.json`
> （SHA256 `54626ae4…7227`，476 个家庭单 agent 任务）。

## A. Simulator / assets

| Item | AI2-THOR | ProcTHOR | VirtualHome |
|---|---|---|---|
| Version / commit | - | - | - |
| Executable path | - | - | - |
| Dataset / asset revision | - | - | - |
| Scene / house count used | - | - | - |
| Known limitations | no depth issue (n/a) | - | no metric depth |

Known facts (to be re-verified before the run):

- AI2-THOR: python package 5.0.0, Unity build `thor-Linux64-f0825767…`
  (`~/.ai2thor/releases`, 2.1 GB), verified by a camera probe on 2026-09-09.
- ProcTHOR: `prior` cache at `~/.prior/datasets/allenai/procthor-10k`,
  revision `439193522244720b86d8c81cde2e51e3a4d150cf`.
- VirtualHome: `envs/virtualhome/simulation/unity_simulator/linux_exec.v2.2.4.x86_64`.

## B. Per-task runtime configuration

| Item | AI2-THOR | ProcTHOR | VirtualHome |
|---|---|---|---|
| Observation | 800×600 RGB, FOV 60 | 800×600 RGB, FOV 60 | 640×480 RGB, FOV 60 |
| Depth available | yes | yes | no |
| Action interface | unified | unified | unified |
| Step budget | 10 + 2·g | 10 + 2·g | 10 + 2·g |
| Init state | task `init.json` | house default | task `init.json` |
| Headless / display | - | - | - |
| Parallel workers | - | - | - |
| Timeout / retries | 120 s / 2 | 120 s / 2 | 120 s / 2 |

## C. Model / API

| Item | Value |
|---|---|
| Provider / gateway | - |
| Model snapshots (GPT / Gemini / Qwen / GPT-6) | - |
| Temperature / top_p | 1.0 / - |
| Context window (turns) | - |
| Max completion tokens | - |
| API timeout / retry policy | 120 s / - |

## D. Hardware / software

| Item | Value |
|---|---|
| Machine / GPU / driver | - |
| OS / WSL / display mode | - |
| Per-env Python venv | - |
| Pipeline commit | - |
| SpatialWorld upstream commit | - |

## E. Pre-run verification (all must be checked)

| Check | Status | Evidence |
|---|---|---|
| 476-task manifest frozen, SHA256 recorded | - | - |
| Every task `task.json` loads; success conditions parse | - | - |
| Per-env smoke run (golden actions) succeeds | - | - |
| Model API reachable; one sample call returns | - | - |
| Eval rooms/houses disjoint from training rooms | - | - |
| Disk space / GPU availability | - | - |
| null/retry policy implemented (TSR excludes null; retry 2) | - | - |
| Results tables/aggregation script ready | - | - |

Sign-off (fill before the batch run):

| Role | Name | Date |
|---|---|---|
| Evaluator | - | - |
| Project owner | - | - |

