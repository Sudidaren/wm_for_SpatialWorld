# LightWM × SpatialWorld 统一评测管线（eval_pipeline）

对 **AI2-THOR / ProcTHOR / VirtualHome** 三类官方家庭任务做统一的端到端
评测，可接入：

- 纯 MLLM：GPT / Gemini / Qwen（官方 agent loop）
- WinmanWM(LightWM 世界模型) + 上述 MLLM
- 任意 RL 策略（DreamerV3 / DIAMOND …，只需实现一个 `act()` 接口）

设计原则：**不改官方评测逻辑**。本目录只负责“选任务、选模型、调度、容错、
统计”；每个任务仍然调用官方 `run_task`（10+2g 步数预算、官方终态 verifier、
`log.json` / `episode_*.json` / 逐帧截图），判定口径与官方 `run_benchmark`
逐函数对齐（见 `official_compat.py`）。

---

## 1. 目录说明

| 文件 | 作用 |
|---|---|
| `eval_config.py` | **实验配置入口**：选场景、模型族、LLM、任务、超时/重试 |
| `env_spec.py` | 扫描官方任务清单与分类（`data/<env>/tasks` + classification CSV） |
| `config_builder.py` | 每个任务生成一份官方格式 YAML（model.vlm / WM 开关） |
| `supervisor.py` | 调度器：流式执行、看门狗、自动重试、断点续跑、结果 CSV/state |
| `official_compat.py` | 官方 run_benchmark 判定逻辑的移植（可审计、可单测） |
| `summarize.py` | TSR / 分环境×类别 / 步数 / token 汇总 |
| `rl_agents.py` | RL 策略接入层 + 内置 golden 自检策略 |

---

## 2. 环境选择（三类 / 两类 / 一类）

改 `eval_config.py` 的 `SCENES`：

```python
SCENES = ["ai2thor", "procthor", "virtualhome"]   # 三类全测
SCENES = ["ai2thor", "procthor"]                  # 只测两类（默认）
SCENES = ["virtualhome"]                          # 只测一类
```

或命令行：

```bash
python supervisor.py --scenes ai2thor,procthor
```

官方单 agent 家庭任务范围：AI2-THOR 282（311 中 29 个 multi-agent 社交任务
自动排除）+ ProcTHOR 127 + VirtualHome 38 = **447 个**。

注意：VirtualHome 没有米制深度，`winman_llm` 只支持 AI2-THOR/ProcTHOR；
VH 请用 `llm`（代码有硬守卫，选错会明确报错）。

---

## 3. 任务选择

```python
TASK_IDS = []                  # 全部任务
TASK_IDS = ["ai2thor03073"]    # 只跑指定任务
TASK_ID_FILTER = "ai2thor0(3|4)"   # 可选正则过滤
```

命令行：

```bash
python supervisor.py --task ai2thor03073 --task procthor100
python supervisor.py --force          # 忽略已完成，全部重跑
```

---

## 4. 模型选择

### 4.1 模型族（PROFILE）

| 值 | 含义 |
|---|---|
| `llm` | 纯 MLLM（GPT/Gemini/Qwen），官方 agent loop |
| `winman_llm` | WinmanWM(LightWM) + 上述 LLM（AI2-THOR/ProcTHOR） |
| `rl` | RL 策略（DreamerV3/DIAMOND…），见第 7 节 |

### 4.2 底层 LLM

```python
LLM_PRESET = "gpt-5"    # gpt-5 / gpt-5.4 / gemini-3.1-pro / qwen3.5

# 每台机器的网关/密钥只放这里：
LLM_OVERRIDES = {
    "base_url": "https://your-gateway/v1",
    "api_key": "sk-...",
    "model_name": "your-model-name",   # 可选
}
```

也可以在环境变量里提供：

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://your-gateway/v1
```

API 凭据解析顺序：`LLM_OVERRIDES` 中的值 → 环境变量
`OPENAI_API_KEY` / `OPENAI_BASE_URL`。缺 key 时启动前会明确报错，
不会空跑。

### 4.3 切换示例

| 想要的效果 | PROFILE | LLM_PRESET |
|---|---|---|
| 纯 GPT | `llm` | `gpt-5` |
| 纯 Qwen | `llm` | `qwen3.5` |
| 纯 Gemini | `llm` | `gemini-3.1-pro` |
| GPT + WingmanWM | `winman_llm` | `gpt-5` |
| Qwen + WingmanWM | `winman_llm` | `qwen3.5` |
| DreamerV3 / DIAMOND | `rl` | 见第 7 节 |

命令行快捷切换：`--profile llm|winman_llm|rl`、`--model <预设名>`。

---

## 5. API 调用与超时机制

评测代码本身不直接调 API：它生成每个任务的 YAML 配置，再调用官方
`scripts.<env>.work.run_task`；官方 runner 每步把截图+文本发给
`model.vlm` 指定的 OpenAI 兼容端点。

防卡死机制（已修复）：

1. **请求级超时**：子进程注入 `VLM_API_TIMEOUT=120`，每次 LLM 请求最多
   等 120 秒（官方 provider 支持该环境变量）。TCP 连接卡死不会再无限等待。
2. **兜底看门狗**：`STALL_TIMEOUT_SEC=600`，整进程超过 600 秒没有任何输出
   才判定僵死，杀掉进程组并按外部故障整任务重试。
3. **全程日志**：每个 attempt 的 stdout/stderr 实时写入
   `<task>/worker_N/run_stream.log`，卡住时可精确定位环节。

以上常量都在 `eval_config.py` 可调。

---

## 6. 运行与产物

```bash
cd eval_pipeline
python supervisor.py --dry-run          # 只打印计划，不启动模拟器
python supervisor.py --profile llm --task ai2thor03073 --no-verify
python supervisor.py                    # 按 eval_config.py 全量
```

每次运行在 `runs/<时间戳>_<profile>_<model>/` 下生成：

```
plan.json              本次计划（场景/模型/任务清单）
task_configs/          每个任务的官方格式 YAML（可复现）
<env>/<task_id>/worker_N/<task_id>/  官方 run_task 原始产物：
                                  log.json / episode_*.json /
                                  逐帧截图 / run_stream.log
results.csv           官方口径（Task ID, Completed）+ 明细列
state.json            断点续跑依据
summary.json          分组统计（TSR/步数/token）
summary_by_env_category.csv
```

`Completed` 与官方一致：`true` / `false` / `null`（null = 外部故障未完成，
不计入 TSR 分母）。中断后重跑同一命令会自动跳过已出决定性结果的任务。

---

## 7. RL 策略接入（DreamerV3 / DIAMOND / 任意策略）

不需要了解 RL 内部实现，只要实现一个方法：

```python
class Policy:
    def act(self, observation, step: int, task: dict) -> dict:
        # 返回 unified action dict，例如：
        # {"action_type": "navigation", "action_name": "MoveAhead",
        #  "granularity": "Medium"}
        # {"action_type": "interaction", "action_name": "PickupObject",
        #  "object_type": "Egg"}
        # {"action_type": "task_completion", "action_name": "DONE"}
        ...
```

配置：

```python
PROFILE = "rl"
RL_POLICY_IMPORT = "my_pkg.policy:Policy"   # 模块:类名
RL_CHECKPOINT = "/path/to/checkpoint.pt"    # 会传给 __init__(checkpoint=...)
```

`RL_POLICY_IMPORT` 留空 = 内置 **GoldenActionsPolicy**：按任务 golden
动作重放一遍，用于不依赖任何 RL 代码的自检。

RL 与 LLM 路线共用：官方 env wrapper、官方终态 verifier
（`perform_final_evaluation`）、同格式 `log.json`/`episode_*.json`、
同一套 CSV/统计/断点/重试。

---

## 8. 与官方仓库的关系与审计记录

以官方仓库 `Hongcheng-Gao/SpatialWorld`（upstream main）为基准。审计发现
官方仓库自身存在几处不一致，本管线做了明确处理：

1. `run_task`（ai2thor/virtualhome）只有 `--output-dir` 以 `worker_` 开头
   时才使用指定目录，否则自建 `outputs/run_<ts>/`；本管线统一传
   `worker_N` 前缀目录，产物受控。
2. 官方 run_benchmark（procthor/virtualhome）golden 复核引用不存在的
   `scripts/evaluate_action_sequence.py`；本管线调用仓库里实际存在的
   `evaluate_actions_{env}.py`，按各脚本真实结果文件名解析
   （ai2thor/procthor → `action_sequence_result_*.json`，
   virtualhome → `vh_action_result_*.json`）。
3. 三个环境各自的 run_benchmark 判定/结果命名不同；`official_compat.py`
   逐函数移植并用 13 组样例与官方 `decide_csv_status_from_result` 比对一致。
4. golden 动作计数按 `task.json` 的 `golden_actions.steps`（与官方
   `load_task_action_count` 一致），不自行扣减。

---

## 9. 已知边界

- `winman_llm` 需要本地 LightWM 运行时（`world_model.py` / `memory_probe.py`，
  不在官方仓库），且只在 AI2-THOR/ProcTHOR 上可用。
- `rl` 真实策略需要你提供 `act()` 封装；内置 golden 策略可先自检链路。
- golden 复核会再启动一次模拟器（成功任务约双倍模拟开销），批量可
  `--no-verify` 或 `VERIFY_GOLDEN_REPLAY=False`。
- `eval_config.py` 里的 `SPATIALWORLD_ROOT` / venv 路径是绝对路径，换机器
  需按第 2 节改路径。

---

## 10. 快速自检

```bash
python -m py_compile *.py
python supervisor.py --dry-run --scenes ai2thor,procthor --profile winman_llm
python supervisor.py --dry-run --scenes ai2thor,procthor,virtualhome --profile llm
```
