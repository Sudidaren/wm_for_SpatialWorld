# LightWM runtime overlay（相对官方 SpatialWorld 的全部代码改动）

## 这是什么

`runtime_overlay/` 里的文件 = 相对官方
`github.com/Hongcheng-Gao/SpatialWorld`（commit `f47b1e0`，main）的全部 LightWM
代码改动（41 个文件，640KB）。克隆官方仓库后把这些文件覆盖过去，再按下面脚本
校验，即可得到与本地一致的运行时代码。

## 怎么用（博士侧，两条命令）

```bash
# 0) 前提：机器上已按官方 README 装好 SpatialWorld 环境（venv/模拟器/API key）
git clone https://github.com/Hongcheng-Gao/SpatialWorld.git
cd SpatialWorld && git checkout f47b1e0

# 1) 克隆本仓库（含代码与 overlay）
git clone git@github.com:Sudidaren/wm_for_SpatialWorld.git lightwm

# 2) 应用 overlay 并校验
bash lightwm/runtime_overlay/setup_lightwm_runtime.sh \
    --repo $(pwd) --overlay lightwm/runtime_overlay
```

校验通过 = 41 个文件与本地逐字节一致（sha256）。

## 环境变量（运行时必须）

```bash
export LIGHTWM_PHASES=/path/to/lightwm            # phase 代码仓库根
export PERCEPTION_CKPT=/path/to/dense_depth_best.pt
export LIGHTWM_DATA_ROOT=/data/lightwm_data       # 6 个数据池根（见 lightwm REPRODUCE_DETECTOR.md）
```

`PERCEPTION_CKPT` 会覆盖配置里的绝对路径；权重与数据下载方式见
`REPRODUCE_DETECTOR.md` 与 HF 说明。

## 本 overlay 包含的关键改动

- `mllm_base_agent/agent/world_model.py`：dense+depth 感知（不再用模拟器分割）、
  地标位姿修正、锚点不确定度（distance_err）、sim 位姿运行时封锁
- `mllm_base_agent/agent/memory_probe.py`：± 区间提示；oracle 可走格回退默认关
- `mllm_base_agent/agent/runner.py`：success_conditions 默认不注入；
  无 perception_ckpt 拒绝运行
- `experiments/configs/ai2thor/`：12 个实验配置（oracle 对照显式打标）
- 其余为早期 LightWM 改动（FD、noisy observer、worldmodel 配置等）

> 调试截图 `envs/ai2thor/step_*.png` 为误留产物，未包含在 overlay 中，
> 不影响任何功能。
