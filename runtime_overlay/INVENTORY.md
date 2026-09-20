# runtime_overlay 文件清单（53 个）

每一个文件都说明了**为什么必须随包发布**。没有无法解释的残留文件：
`verify_delivery.sh` 的第 2 步会检查"overlay 里有没有未列进 MANIFEST 的文件"，
`tools/audit_repo.py` 会检查"仓库里有没有匹配不到任何规则的追踪文件"。

图例：**新增** = 官方 `f47b1e0` 没有这个文件；**修补** = 官方有，我们改了行为。

## A. WM 运行时本体（5 个，新增）——本方法的核心

| 文件 | 作用 | 少了他会怎样 |
|---|---|---|
| `mllm_base_agent/agent/world_model.py` | 世界模型：RGB+单目深度建锚点、航位推算位姿、回环纠偏、记忆账本、`_to_metadata()` 输出契约 | 没有 WM |
| `mllm_base_agent/agent/memory_probe.py` | 每步提示块：手持 / 移动被挡 / 上一动作成败；挂载目标物提示 | 模型看不到记忆 |
| `mllm_base_agent/agent/self_observation.py` | 帧差判动作成败（MSE>1），替代模拟器 error_message | 只能退回读模拟器错误串（作弊） |
| `mllm_base_agent/agent/target_priority.py` | 目标物位置提示（`WM_TARGET_HINT`，默认关）；相关集**只取指令里点名的物体**，动作目标通道 `LIGHTWM_TARGET_FROM_ACTION` 默认关（消融用） | 消融开关失效 |
| `mllm_base_agent/agent/object_query.py` | `CheckState()` 完成前查状态（`WM_STATE_CHECK`，默认关） | 消融开关失效 |
| `mllm_base_agent/agent/state_variants.py` | 加工态改名规则（`SliceObject(X)→XSliced` 等）**从模型本来就收到的系统提示词里解析**，用于把动作里的 `LettuceSliced` 归并回 WM 自己的 `Lettuce` 槽位 | 切过的物体判不出手持/成败、相关集挂不上 |

## B. 官方文件的行为修补（6 个，修补）——每一处都有非改不可的理由

| 文件 | 改了什么 | 为什么必须 |
|---|---|---|
| `mllm_base_agent/agent/runner.py` | WM 接线（建 WM、注入提示、解析 `CheckState`）、tokens/api_calls 计量、去掉 `allow_sim_seg` 后门 | 不接就没有 WM 臂 |
| `mllm_base_agent/agent/state.py` | agent 状态定义：轨迹、会话历史、短时上下文、token 计量字段 | 运行时状态类型 |
| `mllm_base_agent/environments/ai2thor/wrapper.py` | `AI2THOR_SERVER_TIMEOUT` / `AI2THOR_START_TIMEOUT` 可配置 | 软件渲染下"慢"会被误判成环境故障 |
| `mllm_base_agent/llm/provider.py` | 推理模型（gpt-6 / o 系列）改用 `max_completion_tokens` | 否则 GPT-6 臂直接报错 |
| `scripts/ai2thor/work/run_task.py` | 归一化 `success_conditions`（复数键 + `success_logic`） | 否则官方 evaluator 走 legacy 分支抛异常，所有任务恒判失败 |
| `scripts/procthor/work/run_task.py` | 同上（ProcTHOR 侧） | 同上 |

## C. 测试（7 个，新增）

| 文件 | 覆盖 |
|---|---|
| `tests/test_wm_delivery.py` | 信息隔离审计（13 项）：无模拟器通道、无任务真值、无对象词表 + 功能冒烟（记忆读数 / 目标提示 / CheckState / 锚点与位姿） |
| `tests/test_object_query.py` | `CheckState()` 解析与汇总（7 项） |
| `tests/test_object_query_loop.py` | 不拦 DONE、只多一次调用、协议只注入一次（9 项） |
| `tests/test_target_priority.py` | 目标提示分层/配额/降权 + "源码里不得出现对象词表"（10 项） |
| `tests/test_state_variants.py` | 改名规则来自提示词、变体归并到基类槽位、动作目标并入相关集、`PutObject` 目的地不并入（15 项） |
| `tests/test_held_state.py` | 手持状态：看不见但够得着（帧差确认）要认、帧差未知/超出伸手范围不认、`Put` 清空、手上东西不再进"视野内"行（8 项） |
| `tests/test_blocked_hint.py` | 被挡住时压掉打架的移动建议（只压一步）、距离情报保留、事实说硬、走法全列且不下断言、交互选项必须带条件（9 项） |
| `tests/test_param_provenance.py` | **相机/深度参数红线**：每个旋钮的默认值与声明逐字对齐、来源必须在 {definition, non_eval_rooms, geometry} 内、FOV 约定必须由模拟器定义导出（5 项） |

## D. phase C 隐物体信念（不随包发布）

`hidden_location_advisor.py` 与它的单测属 phase C
（`phase_c/hidden_world_belief/`）的独立产出，运行时不引用，2026-09-19
起不再进 overlay：`verify_delivery.sh` 第一步要求 overlay 与工作区逐字节
一致，而工作区里没有这两个文件（它们只存在于 phase_c 目录）。

## E. 实验配置（21 个，`experiments/configs/`）

GPT-5 / GPT-6 Astra 官方基线、memory_probe 提示臂、worldmodel 臂、若干噪声与消融
变体（eggpot / keys / potatoplate / nav），以及 Qwen3-VL-Plus MaaS 的 A/B 探针
（`config_qwen_maas_base.yaml` 与 `config_qwen_maas_hint.yaml`，两者除提示外逐字节相同），
另有 CARLA / ProcTHOR / VirtualHome 各一份 `config_close_gpt-5.yaml`。
它们是历史实验的可复现入口；当前主评测由 `eval_pipeline` 运行时生成配置，不依赖这些文件。

## F. 其它环境与诊断脚本（13 个）——本方法不依赖，但确认归属前不删

| 文件 | 说明 |
|---|---|
| `evaluation/carla/base.py`、`mllm_base_agent/environments/carla/wrapper.py`、`scripts/carla/work/run_task.py` | CARLA 侧改动（我们不跑 CARLA） |
| `scripts/embodiedcity/work/utils/api_utils.py` | EmbodiedCity 侧改动 |
| `mllm_base_agent/environments/game/core/__init__.py` | game 环境占位（另一条线） |
| `scripts/start_local_planner.sh`、`scripts/test_windows_envs.ps1`、`scripts/ai2thor/safe_run.sh` | 早期环境工具 |
| `envs/ai2thor/probe_expA.py`、`probe_move.py`、`run_task_loop_only.py`、`test_connection.py`、`validate_occupancy.py` | 早期探查与验证脚本 |

它们与其它环境（CARLA / EmbodiedCity / game）以及早期探查脚本有关，
不属于世界模型路径，保留原样以免影响其它环境的运行。

## G. 元文件（1 个）

`.gitignore` —— 官方仓库的忽略规则（overlay 会覆盖同名文件）。

## 不随包发布的东西

| 内容 | 原因 |
|---|---|
| `wm_modules/`（hint_gate / occupancy_completion / frontier_topology / memory_eviction / exploration_scoring / feasibility_features） | 伙伴的独立模块，从未推送到本仓库；运行时没有任何 import。位置预测模块已决定暂不做 |
| `envs/ai2thor/step_*.png` 调试截图 | 调试产物 |
| `port_8080.txt` / `port_8789.txt` | 本地端口记录 |
| 模型权重 | 体积过大，见 GitHub Release 与 `REPRODUCE_DETECTOR.md` |
