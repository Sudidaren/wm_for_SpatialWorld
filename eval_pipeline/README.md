# SpatialWorld 家庭任务统一评测管线

覆盖 **AI2-THOR / ProcTHOR / VirtualHome** 三类场景的官方家庭任务，测试
LLM 系与 RL 系模型。核心设计：**官方评测逻辑一行不改**，本目录只做四件事：

1. 按配置选择场景/任务；
2. 按配置生成模型配置（纯 LLM 或 WingmanWM+LLM）；
3. 逐任务调用官方 `run_task`（同样的 10+2g 步数预算、同样的终态校验、
   同样的 `log.json` / `episode_*.json` / 逐帧图片 / 轨迹记录）；
4. 容错重试、断点续跑、CSV + JSON 统计。

## 目录

```
eval_config.py      ← 实验配置（场景/模型族/LLM/任务/路径），改这里即可
env_spec.py         任务清单与分类（官方 data/ + classification CSV）
config_builder.py   每个任务生成一份官方格式 YAML（含 per-task WM targets）
supervisor.py       调度器：重试/超时/断点/并发/统计
summarize.py        TSR 与分场景×类别汇总
rl_agents.py        DreamerV3 / DIAMOND 接入接口（接口已定，推理体待接）
```

## 1. 选择评测范围（改代码即可，全部在 eval_config.py）

```python
# 默认：AI2-THOR + ProcTHOR（wingman_llm 可用范围）
SCENES = ["ai2thor", "procthor"]

# 三类都测（VirtualHome 无深度，配合 PROFILE="llm" 使用）：
SCENES = ["ai2thor", "procthor", "virtualhome"]

# 只测一类：
SCENES = ["virtualhome"]
```

官方单 agent 家庭任务范围：AI2-THOR 282（311 中 29 个为 multi-agent 社交
任务，走官方 dual 协议，不在这条评测线）+ ProcTHOR 127 + VirtualHome 38，
合计 447。管线会按 `task_classification_detail.csv` 自动排除 multi-agent 任务。

```python
# 任务：[] = 全测；填 id = 只测这几个；TASK_ID_FILTER 可加正则
TASK_IDS = []                              # 全部
TASK_IDS = ["ai2thor03073", "procthor100"] # 指定
```

## 2. 选择模型

```python
# 模型族
PROFILE = "wingman_llm"   # llm | wingman_llm | rl

# 底层 LLM（GPT/Gemini/Qwen 已内置预设；模型名/网关/api key 用
# LLM_OVERRIDES 覆盖，不写死在预设里）
LLM_PRESET = "gpt-5"
LLM_OVERRIDES = {"base_url": "https://your-gateway/v1",
                 "api_key": "sk-..."}
```

说明：
- `llm`：官方 MLLM agent loop（官方 prompt / 29 轮上下文 / τ=1.0）。
- `wingman_llm`：同一官方 loop + LightWM 世界模型运行时。开启后 agent 端
  关闭模拟器语义/位姿作弊；感知只用 RGB+depth，`memory_probe.targets`
  从每个任务自己的 `task.json` 自动读取（所以任意任务集都能跑，不限于
  之前手写的几个 potato/keys 配置）。感知权重见 `PERCEPTION_CKPT`。
- `rl`：DreamerV3 / DIAMOND 接入缝见 `rl_agents.py`。因为官方仓库目前
  没有这两个模型的推理代码/权重，接口定好但推理体留空——选择 rl 会明确
  报错而不是假装能跑。

## 3. 运行

```bash
cd /home/sudidaren/spatialworld_eval

# 预览将执行的任务（不启动模拟器）
python supervisor.py --dry-run

# 正式跑（按 eval_config.py）
python supervisor.py

# WingmanWM + GPT 全量（AI2-THOR + ProcTHOR；VirtualHome 另行 llm 批）
python supervisor.py --profile wingman_llm --scenes ai2thor,procthor

# 常用临时覆盖
python supervisor.py --scenes ai2thor,procthor \
                     --profile wingman_llm \
                     --task ai2thor03073
python supervisor.py --profile llm --workers 2 --no-verify
python supervisor.py --force          # 忽略已完成，全部重跑
```

## 4. 产物结构

`runs/<时间戳>_<profile>_<model>/`

```
plan.json             本次计划（场景/模型/任务清单）
task_configs/         每个任务一份官方格式 YAML（可复现）
ai2thor/<task_id>/attempt_0/...   官方 run_task 输出：
                                  log.json / episode_*.json /
                                  逐帧图片 / steps.jsonl / world_model/
results.csv           官方口径列（Task ID, Completed, ...）+ 明细列
state.json            每任务进度（断点续跑依据）
summary.json          分组统计
summary_by_env_category.csv
```

`Completed` 与官方一致：`true` / `false` / `null`。api/env 故障与崩溃一律
写 `null` 并自动重试（`MAX_EXTERNAL_RETRIES`），TSR 分母只统计 true+false。

## 5. 统计口径

- TSR = 成功 / (成功+失败)，pending(null) 不计入分母（与官方聚合脚本一致）。
- 步数记录来自官方 `run_task`（不含 init 动作的 `step_count`）。
- 效率参考列 `avg_se_success` = golden/actual 只在成功任务上平均；如需官方
  论文口径的 SE 定义可到 `summarize.py` 一处替换。
- 可选 golden 复核：默认开启（`VERIFY_GOLDEN_REPLAY`），用官方
  `evaluate_actions_<env>` 在干净环境重放 golden，行为与官方 run_benchmark
  的复核一致；测试预算紧张时可关。

## 6. 注意事项

- 每个场景使用各自官方 uv venv（`envs/<scene>/.venv`）执行，保证依赖与官方
  一致；首次跑 AI2-THOR 会下载 Unity 运行时到本机缓存。
- VirtualHome 无 metric depth，wingman_llm 的深度头无法在 VH 上运行；VH
  评测建议用 `llm` profile，或等感知头 VH 支持后再开。
- 服务器上跑请保持 `HEADLESS=True`；AutoDL 等无显示环境不要关。
- 评测任务与训练房间的隔离（防泄漏）由数据/训练侧保证，本管线不改任务。

## 7. 与官方仓库对照后的已知差异（审计记录）

本管线以"官方现有可执行代码"为对齐基准，发现官方仓库自身存在几处不一致，
我们做了明确处理，避免被误认为本管线的偏差：

1. `scripts/*/work/run_task.py` 只有在 `--output-dir` 的 basename 以
   `worker_` 开头（ai2thor / virtualhome；procthor 单任务例外）时才使用
   该目录，否则自建 `outputs/run_<ts>/`。官方 `run_benchmark` 直接传
   `<benchmark>/<task_id>`，实际上不会被 run_task 采用。本管线统一传
   `worker_<n>` 前缀目录，保证每任务产物落在受控 attempt 目录内。
2. 官方 `run_benchmark`（procthor / virtualhome 分支）的 golden 复核引用
   `scripts/evaluate_action_sequence.py`，该文件当前仓库不存在；ai2thor
   分支引用 `scripts/evaluate_actions_ai2thor.py`（存在）。本管线按仓库中
   实际存在的 `evaluate_actions_{env}.py` 实现复核，结果解析命名随各脚本：
   ai2thor/procthor → `action_sequence_result_*.json`，
   virtualhome → `vh_action_result_*.json`。
3. 官方 CSV 的扩展列名与判定规则（`decide_csv_status_from_result`、
   `determine_failure_reason`、HTTP 错误码 → null 等）逐函数移植到
   `official_compat.py`，并用 13 组样例与官方函数逐例对比一致。
4. golden 动作计数以 `task.json` 的 `golden_actions.steps` 字段为准
   （与官方 `load_task_action_count` 一致；`steps` 含结尾 DONE/FAIL 时
   照实记录，不自行扣减）。

未决事项：golden 复核会再启动一次模拟器（每个成功任务约一倍的模拟开销），
批量跑时可 `--no-verify`；如需完全关闭请把 `VERIFY_GOLDEN_REPLAY=False`。

## 默认感知更新

`wingman_llm` 现在默认使用 RF-DETR Small 检测器和已有单目深度模块，阈值 0.40。
在各任务 worker 的 Python 环境安装 `../requirements-rfdetr.txt`，并重新应用本仓库的 runtime overlay。
权重保留在 NFS，通过 `LIGHTWM_STORAGE_ROOT` 或各权重环境变量定位。
完整说明及结果见 [RF-DETR 感知模块](../docs/perception_rfdetr.md)。此变更不启动或恢复评测任务。
