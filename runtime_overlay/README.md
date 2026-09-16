# LightWM runtime overlay（相对官方 SpatialWorld 的全部代码改动）

> **2026-09-16 更新**：overlay 已重刷为"无作弊 WM v2"运行时（**52 个文件**），
> 与 `SpatialWorld` 本地工作区逐字节一致，`MANIFEST.sha256` 已重建。
> 逐文件说明见 [`INVENTORY.md`](INVENTORY.md)，完整交接见
> [`docs/HANDOVER_2026-09-16.md`](../docs/HANDOVER_2026-09-16.md)。

## 这是什么

`runtime_overlay/` 里的文件 = 相对官方
`github.com/Hongcheng-Gao/SpatialWorld`（commit `f47b1e0`，main）的全部 LightWM
代码改动（52 个文件）。克隆官方仓库后把这些文件覆盖过去，再按下面脚本
校验，即可得到与本地一致的运行时代码。

## 怎么用（三条命令）

```bash
# 0) 前提：机器上已按官方 README 装好 SpatialWorld 环境（venv/模拟器/API key）
git clone https://github.com/Hongcheng-Gao/SpatialWorld.git
cd SpatialWorld && git checkout f47b1e0

# 1) 克隆本仓库（含代码与 overlay）
git clone git@github.com:Sudidaren/wm_for_SpatialWorld.git lightwm

# 2) 应用 overlay 并校验
bash lightwm/runtime_overlay/setup_lightwm_runtime.sh \
    --repo $(pwd) --overlay lightwm/runtime_overlay

# 3) 一条命令验证"拉下来即可用、且没有任何作弊通道"
bash lightwm/runtime_overlay/verify_delivery.sh \
    --repo $(pwd) --overlay lightwm/runtime_overlay
```

`verify_delivery.sh` 会依次检查：① 52 个文件 sha256 一致；② overlay 里没有未登记的
文件（无冗余）；③ 运行时可编译；④ **信息隔离审计 + 功能冒烟**（`tests/test_wm_delivery.py`：
无模拟器通道 / 无任务真值 / 无对象词表，且记忆读数、目标提示、CheckState、锚点与位姿
都能真的跑出来）；⑤ 其余三组单测。全程不需要 GPU、模拟器和模型权重。

校验通过 = 52 个文件与本地逐字节一致（sha256）。

> **2026-09-16 实测**：在一个干净的 `f47b1e0` worktree 上执行上面三条命令，
> 52/52 `sha256 OK`、`verify_delivery.sh` **7 个步骤全过**（含 14 项信息隔离
> 审计与功能冒烟、7+9+10 项单测）。
> overlay 只做"新增 + 覆盖"，官方 `mllm_base_agent/agent/` 在 f47b1e0 只有
> 4 个文件（`__init__.py` / `graph.py` / `runner.py` / `state.py`），
> 其余运行时文件全部由本 overlay 提供，**不需要手工删除任何官方文件**。

## 环境变量（运行时必须）

```bash
export LIGHTWM_PHASES=/path/to/lightwm            # phase 代码仓库根
export PERCEPTION_CKPT=/path/to/dense_depth_best.pt
export LIGHTWM_DATA_ROOT=/data/lightwm_data       # 6 个数据池根（见 lightwm REPRODUCE_DETECTOR.md）
```

`PERCEPTION_CKPT` 会覆盖配置里的绝对路径；权重与数据下载方式见
`REPRODUCE_DETECTOR.md` 与 HF 说明。

## 本 overlay 包含的关键改动（2026-09-16 版）

**WM 运行时（`mllm_base_agent/agent/`）**

- `world_model.py`：dense+depth 感知（RF-DETR Small 检测 + DINOv2 单目深度，
  不再用模拟器分割/真值深度）、物体中心 3D 锚点、位姿纯航位推算（冻结 sim 位姿）、
  地标回环纠偏、锚点不确定度；`_to_metadata()` 是给提示层唯一的出口
- `self_observation.py`（新）：帧差判动作成败（MSE>1），替代模拟器 error_message
- `memory_probe.py`：每步提示块（手持 / 移动被挡 / 上一动作成败）；
  已删除 oracle 走格回退、FD、噪声观测等旧通道
- `target_priority.py`（新）：目标物位置提示（**默认关**，`WM_TARGET_HINT=1` 开）；
  物品名由模型开局自由文本自述，**无词表、无别名表**
- `object_query.py`（新）：`CheckState()` 完成前查状态（**默认关**，`WM_STATE_CHECK=1` 开）；
  不拦 DONE；通用 `Query(<物体>)` 已砍
- `runner.py`：接线与计量（tokens / api_calls / 步数），无 `perception_ckpt` 直接报错

**相对上一版 overlay 被移除的模块（不必手工删，官方仓库里本来就没有）**

- `mllm_base_agent/agent/failure_detection.py`（读模拟器真值判成败 → 改为帧差）
- `mllm_base_agent/agent/noisy_observer.py`（旧信息隔离层）
- `mllm_base_agent/agent/subgoals.py`（旧子目标分解）
- `mllm_base_agent/agent/plan.py`（任务级子目标分解，2026-09-17 删除：无收益，
  且会在上下文里多塞一份机器生成的计划。`runner.py` 里的接线、`wm_config_patch`
  的 `WM_PLAN` 开关一并移除；`run_ablation.sh` 的 `plan` 位置参数保留但已不再被读取）

它们只存在于**旧版** overlay 里，是早期实验的产物；现在这套运行时不再需要，
官方 `f47b1e0` 也从未包含它们，所以照上文两条命令操作即可，无需删除动作。

**环境与模型适配**

- `environments/ai2thor/wrapper.py`：`AI2THOR_SERVER_TIMEOUT` / `AI2THOR_START_TIMEOUT`
  可调（软件渲染下仅因慢而超时，不该算环境故障）
- `llm/provider.py`：推理模型（gpt-6 / o1 / o3 / o4）自动改用 `max_completion_tokens`
- `scripts/ai2thor/work/run_task.py`、`scripts/procthor/work/run_task.py`：
  归一化 `success_conditions`（复数键 + `success_logic`），否则官方 evaluator
  走 legacy 分支对 list 调 `.get()` 抛异常、所有任务恒判失败

**测试**

- `tests/test_object_query.py`（7/7）、`tests/test_object_query_loop.py`（9/9）、
  `tests/test_target_priority.py`（10/10，含"源码里不得出现对象词表"的断言）

**未接入 overlay 的独立产出**

- `mllm_base_agent/agent/hidden_location_advisor.py` 与
  `test_hidden_location_advisor.py` 属 phase C（`phase_c/hidden_world_belief/`）
  的独立模块，运行时没有任何代码 import 它们，保留只为该目录文档里的测试路径可用。

## 历史改动（2026-09-14 及以前）

- `mllm_base_agent/agent/world_model.py`：dense+depth 感知（不再用模拟器分割）、
  地标位姿修正、锚点不确定度（distance_err）、sim 位姿运行时封锁
- `mllm_base_agent/agent/memory_probe.py`：± 区间提示（**已于 09-16 删除**）
- `experiments/configs/ai2thor/`：12 个实验配置（oracle 对照显式打标）
- 其余为早期 LightWM 改动（FD、noisy observer、worldmodel 配置等）

> 调试截图 `envs/ai2thor/step_*.png` 为误留产物，未包含在 overlay 中，
> 不影响任何功能。
