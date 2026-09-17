# AI2-THOR 311：WingmanWM 配对评测快照

## 这批是什么

三个开源 VLM 各自接入 **WingmanWM**（只使用 RGB + 动作日志的空间世界模型），在
**AI2-THOR 311 条**任务上与同模型、同任务集的纯基线逐题配对。

> **配置**：目标物位置提示（`WM_TARGET_HINT`）与 `CheckState`（`WM_STATE_CHECK`）
> **均为关闭**。即这一批测的是"世界模型只提供记忆读数（手持 / 移动被挡 / 上一个
> 动作成败）"这一配置，不是"世界模型 + 目标物位置提示"。

## 结果

| 模型 | 终止 | 判定 | 成功 | TSR | 外部失败 | 基线成功 | 基线 TSR | WM 独有 | 基线独有 | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | 311 | 272 | 14 | **5.1%** | 39 | 6 | 2.2% | 11 | 3 | 0.057 |
| Kimi-VL-A3B-Instruct | 311 | 262 | 6 | **2.3%** | 49 | 5 | 1.9% | 4 | 3 | 1.000 |
| Qwen3-VL-30B-A3B | 311 | 311 | 17 | **5.5%** | 0 | 20 | 6.4% | 4 | 7 | 0.549 |

## 失败结构（占失败数比例）

| 模型 + WM | 成功 | 提前 DONE | 步数耗尽 | 其他 | tokens/task |
|---|---|---|---|---|---|
| 8B + WM | 14 | 23 (9%) | 224 (87%) | 11 | 291,726 |
| Kimi + WM | 6 | 147 (57%) | 62 (24%) | 47 | 161,888 |
| 30B + WM | 17 | 65 (22%) | 206 (70%) | 23 | 251,154 |

**读法**：8B 的失败集中在步数耗尽（搜索效率），Kimi 的失败集中在提前 DONE
（把自己以为做完当成做完）。两种失败模式对应不同的干预方向。

## 基线来源与口径

| 模型 | 基线批次 | 说明 |
|---|---|---|
| Qwen3-VL-30B-A3B | 纯模型，AI2-THOR 311 条 | 20 成功 = 6.4%，与官方表格一致 |
| Qwen3-VL-8B | 438 条批次的 AI2-THOR 子集 | 其中 267/311 有判定 |
| Kimi-VL-A3B | 438 条批次的 AI2-THOR 子集 | 其中 267/311 有判定 |

配对是在"世界模型已判定"的任务集上做的。基线在这些任务里有少数没有判定
（8B 5 条、Kimi 7 条，多为环境类失败），按"基线未成功"计入，
因此这两行的基线 TSR 略偏保守。

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
