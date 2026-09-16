# AI2-THOR 311：WM 臂 vs 基线（配对快照，2026-09-16 23:19）

## 这一批是什么

三个开源 VLM 各自接上 **WingmanWM**（只吃 RGB + 动作日志的空间世界模型），在
**AI2-THOR 311 条**任务上跑，与**同一个模型、同一批任务**的纯基线做逐题配对。

> ⚠️ **重要口径说明：本批次的 `WM_TARGET_HINT`（目标物位置提示）与
> `WM_STATE_CHECK`（CheckState）全部为关闭状态，`WM_PLAN=off`。**
> 也就是说，这一批测的是**"只给记忆、不给任何提示"的纯 WM 底数**，
> 不是 "WM + 目标物提示" 的完整方法。判据来自云端每个任务生成的 YAML
> （`memory_probe.target_hint.enabled: false`），不是读代码推测的。

## 结果

| 模型 | WM 判定 | WM 成功 | WM TSR | 基线成功 | 基线 TSR | WM 独有 | 基线独有 | McNemar p |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | 177 | 9 | **5.1%** | 3 | 1.7% | 8 | 2 | 0.109 |
| Kimi-VL-A3B-Instruct | 191 | 1 | **0.5%** | 2 | 1.0% | 0 | 1 | 1.000 |
| Qwen3-VL-30B-A3B | 86 | 3 | **3.5%** | 4 | 4.7% | 1 | 2 | 1.000 |

30B 臂在快照时**仍在跑**（86/311），只能算趋势；8B / Kimi 也尚未跑满。

## 失败结构（占失败数比例）

| 模型 + WM | 成功 | 提前 DONE | 步数耗尽 | 其他 | Avg tokens/task |
|---|---|---|---|---|---|
| 8B + WM | 9 | 13 (8%) | **148 (88%)** | 7 | 342,907 |
| Kimi + WM | 1 | **112 (59%)** | 42 (22%) | 36 | 175,059 |
| 30B + WM | 3 | 22 (27%) | **60 (72%)** | 1 | 294,694 |

**结论**：8B 的病是"找不到东西 → 步数耗尽"；Kimi 的病是"真以为做完了 → 提前 DONE"。
两种病对应两种不同的干预手段（位置提示 vs 完成前自查），不能用一个方案糊过去。

## 基线来源

| 模型 | 基线批次 | 说明 |
|---|---|---|
| Qwen3-VL-30B-A3B | `qwen3vl30b_ai2thor_v1`（311 条，20 成功 = 6.4%） | 与官方主表 AI2-THOR 6.4% 一致 |
| Qwen3-VL-8B | `qwen3vl8b_438_v1` 取 AI2-THOR 子集 | 438 批里只有 267/311 有判定（其余 failed_external） |
| Kimi-VL-A3B | `kimivl_a3b_438_v1` 取 AI2-THOR 子集 | 同上 267/311 |

**两个必须写进论文的限制**：
1. 配对是在"WM 已判定集"上算的，不是全 311（8B/Kimi 基线有外部失败任务）。
2. 30B 臂样本还不完整。

## 文件

| 文件 | 内容 |
|---|---|
| `paired_tasks.csv` | 逐 (模型, 任务) 的 WM / 基线成败、失败原因、步数、token |
| `summary.json` | 上表的机器可读版 + 口径说明 + caveats |
| `make_results_snapshot.py` | 快照生成脚本（从 `state.json` 复算全部数字） |

## 口径

- **TSR = success / decided**；`failed_external` 记为 Null，**不进分母**。
- `decided` = `status ∈ {success, failed_model}`。
- "提前 DONE" = `fail_reason` 含 `Model claimed DONE but success conditions not met`。
- "步数耗尽" = `fail_reason` 含 `step limit`。
- 任务集：AI2-THOR 311 条（同一次冻结切分）。

## 复算

```bash
# 1) 从云端回拉 state.json 到 ~/wm_dev/pull_latest/
python3 eval_pipeline/cloud/monitor_cloud_eval.py --write
# 2) 生成同样的表
python3 results/wm_ai2thor311_20260916/make_results_snapshot.py <out_dir>
# 3) 三卡配对对比 + 失败结构（含旧词表版对照）
python3 eval_pipeline/cloud/compare_latest.py
```
