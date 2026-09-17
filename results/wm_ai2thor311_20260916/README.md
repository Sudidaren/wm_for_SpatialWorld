# AI2-THOR 311：WingmanWM 配对评测快照

## 这批是什么

三个开源 VLM 各自接入 **WingmanWM**（只使用 RGB + 动作日志的空间世界模型），在
**AI2-THOR 311 条**任务上与同模型、同任务集的纯基线逐题配对。

> **配置**：目标物位置提示（`WM_TARGET_HINT`）与 `CheckState`（`WM_STATE_CHECK`）
> **均为关闭**。即这一批测的是"世界模型只提供记忆读数（手持 / 移动被挡 / 上一个
> 动作成败）"这一配置，不是"世界模型 + 目标物位置提示"。

## 结果（全部 311 条判定，已把外部失败补测补齐）

原先有一批任务在软件渲染（Linux64 + Xvfb）下环境初始化超时、拿不到判定；它们在
CloudRendering 下重跑后全部有了结果，按 `task_id` 合并（先判定的为准）。

| 模型 | 判定 | 成功 | TSR | 基线成功 | 基线 TSR | WM 独有 | 基线独有 | McNemar p |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | 311 | 16 | **5.1%** | 6 | 1.9% | 13 | 3 | **0.021** |
| Kimi-VL-A3B-Instruct | 311 | 7 | **2.3%** | 6 | 1.9% | 4 | 3 | 1.000 |
| Qwen3-VL-30B-A3B | 311 | 17 | **5.5%** | 20 | 6.4% | 4 | 7 | 0.549 |

基线（同一批补测后）：8B 6/311、Kimi 6/311、30B 20/311。

## 失败结构（占失败数比例）

| 模型 + WM | 成功 | 提前 DONE | 步数耗尽 | tokens/task |
|---|---|---|---|---|
| 8B + WM | 16 | 25 (8%) | 259 (88%) | 284,609 |
| Kimi + WM | 7 | 174 (57%) | 73 (24%) | 162,518 |
| 30B + WM | 17 | 65 (22%) | 206 (70%) | 251,153 |

**读法**：8B 的失败集中在步数耗尽（搜索效率），Kimi 的失败集中在提前 DONE
（把自己以为做完当成做完）。两种失败模式对应不同的干预方向。

## 基线来源与口径

| 模型 | 基线批次 | 说明 |
|---|---|---|
| Qwen3-VL-30B-A3B | 纯模型，AI2-THOR 311 条 | 20 成功 = 6.4%，与官方表格一致 |
| Qwen3-VL-8B | 438 条批次的 AI2-THOR 子集 + 补测 | 补齐到 311/311 |
| Kimi-VL-A3B | 438 条批次的 AI2-THOR 子集 + 补测 | 补齐到 311/311 |

配对覆盖全部 311 条任务，双方都有判定。

## 文件

| 文件 | 内容 |
|---|---|
| `paired_tasks.csv` | 逐 (模型, 任务) 的双方成败、失败原因、步数、token |
| `summary.json` | 上表的机器可读版 + 口径与限制说明 |
| `make_results_snapshot.py` | 由 `state.json` 复算全表 |

## 指标定义

- **TSR = success / decided**；`decided` = `status ∈ {success, failed_model}`。
- `failed_external`（环境/外部故障）记为 Null，**不进 TSR 分母**。
- 终止 = decided + failed_external。
- **提前 DONE** = `fail_reason` 含 `Model claimed DONE but success conditions not met`。
- **步数耗尽** = `fail_reason` 含 `step limit`。
- 任务集：AI2-THOR 311 条（固定切分）。

## 复算

```bash
# 1) 从云端回拉 state.json / results.csv
python3 eval_pipeline/cloud/monitor_cloud_eval.py --write
# 2) 生成同样的表
python3 results/wm_ai2thor311_20260916/make_results_snapshot.py <out_dir>
# 3) 三卡配对对比 + 失败结构
python3 eval_pipeline/cloud/compare_latest.py
```
