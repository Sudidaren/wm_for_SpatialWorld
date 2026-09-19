# LightWM runtime overlay（相对官方 SpatialWorld 的全部代码改动）

> 本 overlay 含 **53 个文件**，与 SpatialWorld 工作区逐字节一致（`MANIFEST.sha256`）。
> 逐文件说明见 [`INVENTORY.md`](INVENTORY.md)，完整交接见
> [`docs/HANDOVER_2026-09-16.md`](../docs/HANDOVER_2026-09-16.md)。

## 这是什么

`runtime_overlay/` 里的文件 = 相对官方
`github.com/Hongcheng-Gao/SpatialWorld`（commit `f47b1e0`，main）的全部 LightWM
运行时代码（53 个文件）。克隆官方仓库后把这些文件覆盖过去，再按下面脚本校验，
即可得到与本地一致的运行时代码。

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

`verify_delivery.sh` 会依次检查：① 53 个文件 sha256 一致；② overlay 里没有未登记的
文件（无冗余）；③ 运行时可编译；④ **信息隔离审计 + 功能冒烟**（`tests/test_wm_delivery.py`：
无模拟器通道 / 无任务真值 / 无对象词表，且记忆读数、目标提示、CheckState、锚点与位姿
都能真的跑出来）；⑤ 其余三组单测。全程不需要 GPU、模拟器和模型权重。

校验通过 = 53 个文件与本地逐字节一致（sha256）。

> **2026-09-16 实测**：在一个干净的 `f47b1e0` worktree 上执行上面三条命令，
> 52/52 `sha256 OK`（当时版本）、`verify_delivery.sh` **7 个步骤全过**（含 14 项信息隔离
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

## 本 overlay 的内容

**世界模型运行时（`mllm_base_agent/agent/`）**

- `world_model.py`：RGB + 单目深度感知（RF-DETR Small 检测头 + DINOv2 深度头，
  不使用模拟器分割或真值深度）、物体 3D 锚点与多视角三角化、纯动作日志航位推算、
  相机模型（含 `cameraHorizon` 与 `CAMERA_Y = 0.675`）、地标位姿校正、回环纠偏、
  锚点不确定度；`_to_metadata()` 是给提示层的唯一出口
- `self_observation.py`：由相邻帧差异判断动作是否改变画面
- `memory_probe.py`：每步提示块（手持 / 移动被挡 / 上一动作成败）
- `target_priority.py`：目标物位置提示（`WM_TARGET_HINT=1` 开启，默认关）
- `object_query.py`：`CheckState()` 完整前状态汇总（`WM_STATE_CHECK=1` 开启，默认关；不拦截 DONE）
- `runner.py`：接线与计量（tokens / api_calls / 步数）；没有 `perception_ckpt` 时直接报错

**环境与模型适配**

- `environments/ai2thor/wrapper.py`：`AI2THOR_SERVER_TIMEOUT` / `AI2THOR_START_TIMEOUT` 可调
- `llm/provider.py`：推理模型（gpt-6 / o1 / o3 / o4）自动使用 `max_completion_tokens`
- `scripts/ai2thor/work/run_task.py`、`scripts/procthor/work/run_task.py`：
  归一化 `success_conditions`（复数键 + `success_logic`），供官方 evaluator 使用

**测试**

- `tests/test_wm_delivery.py`（信息隔离审计 + 功能冒烟，21 项）
- `tests/test_object_query.py`（7）、`tests/test_object_query_loop.py`（9）、
  `tests/test_target_priority.py`（15）、`tests/test_state_variants.py`（15）

**独立产出（运行时不引用、不随 overlay 发布）**

- `hidden_location_advisor.py` 与它的单测属 phase C
  （`phase_c/hidden_world_belief/`）的独立模块，2026-09-19 起移出 overlay：
  工作区里没有这两个文件，而第一步校验要求逐字节一致。

**实验配置**

- `experiments/configs/` 下为各模型与各环境的对照配置。

> 调试截图 `envs/ai2thor/step_*.png` 不包含在 overlay 中。
