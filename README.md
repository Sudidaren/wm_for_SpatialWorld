# LightWM：世界模型（阶段 B/C/D）

LightWM 是一个"人类式情境顾问"世界模型：**仅凭 RGB 图像与动作日志**记忆、理解并
预测空间关系，并在正确时机递上最小、能阻止错误、带置信度的提示（VoI 门控）。
本仓库包含阶段 B（感知/空间记忆）、阶段 C（VoI 门控）、阶段 D（评测框架）的完整代码。

## 总体架构

```
RGB + 动作日志
    │
    ▼
感知层：DINOv2-S（冻结）→ 适配器 → 稠密检测头（物体/类别）
                        ├─ 深度头（单目深度，训练用 GT 深度监督）
                        └─ 可行性头（动作成败预测）
    │  检测框 + 深度 + 位姿
    ▼
记忆层：物体中心 3D 锚点（跨帧融合）→ 场景图 → 占用补全
    │  候选信息
    ▼
VoI 门控：打分（P(阻止错误)×价值 − 成本），每步放行 0~1 条提示
    │
    ▼
提示渲染 ──► GPT-5
```

设计要点：
- 空间记忆以**物体中心 3D 锚点**为核心（非 2D 格子），保留高度/身份/拓扑
- 运行时不作弊：不读模拟器位姿、语义元数据、深度真值（深度仅训练监督）
- 坐标约定（`shared/geometry.py`，有单测）：相机 = agent + (0, 0.675, 0)；
  yaw 前向 = (sin, cos)；俯仰负=抬头；深度 PNG = 米×1000

## 目录结构

```
shared/   坐标几何、3D 锚点空间记忆（含不确定性/外观库）、闭环、场景图、
          记忆管理器（置信度/新鲜度/重要性/淘汰）、运行时门控桥接
phase_b/  感知模型（DINOv2+适配器+稠密检测头+深度头+可行性头）、数据集、
          训练/评测、数据采集（collect_coverage.py）
phase_c/  门控候选提取、打分头训练、离线拦截率
phase_d/  9 任务定义、三组评测配置生成、配对统计
```

## 数据

三部分数据，用环境变量指向实际路径：

| 环境变量 | 默认路径 | 内容 |
|---|---|---|
| `LIGHTWM_DATA_ROOT` | `/mnt/d/lightwm_data` | 原始 34,326 帧（117 场景） |
| `LIGHTWM_COV_ROOT` | `/mnt/d/lightwm_data_cov` | 穷举覆盖 5,316 帧（12 场景：全位置×4朝向×3俯仰） |
| `LIGHTWM_OBJVIEW_ROOT` | `/mnt/d/lightwm_data_objviews` | 多角度物体视图 25,719 帧（56 类小物品，89 场景） |
| `LIGHTWM_FD_ROOT` | `/mnt/d/fd_benchmark_full_20260811_224644` | FD 轨迹 3,618 帧（可行性头） |

目录结构：
```
<root>/
  episodes/<scene>__<episode>/episode.json   # 每帧：RGB/depth/seg 路径、动作、
  frames/step_00000_rgb.png                  #  成败、agent 位姿、visible_objects bbox
  frames/step_00000_depth.png
  frames/step_00000_seg.png
  scene_gt/<scene>.json                      # 场景物体世界坐标真值
```

最小可运行子集：任意 2-3 个 episode 目录 + 对应 `scene_gt/*.json` 即可跑通
索引与空间记忆评测。

## 环境搭建

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers numpy pillow tqdm pyyaml scipy
export HF_ENDPOINT=https://hf-mirror.com   # 国内：DINOv2 权重走镜像
```

## 快速上手

```bash
# 1) 建帧索引
export LIGHTWM_DATA_ROOT=<你的路径>/lightwm_data
export LIGHTWM_COV_ROOT=<你的路径>/lightwm_data_cov
python shared/data_index.py

# 2) 空间记忆评测（GT 输入，验证记忆/方向/高度/距离）
python phase_b/eval_spatial.py --max-episodes 80

# 3) 训练感知头
python -u phase_b/train.py --task dense --epochs 8 --batch 24 --out checkpoints
python -u phase_b/train.py --task depth --epochs 4 --batch 24 --out checkpoints
python -u phase_b/train.py --task feasibility --epochs 6 --batch 64 --out checkpoints

# 4) 评估
python phase_b/eval_perception.py --ckpt checkpoints/dense_best.pt --head dense \
  --depth-ckpt checkpoints/depth_best.pt
python phase_b/eval_feasibility.py

# 5) VoI 门控（离线）
python phase_c/extract_candidates.py
python phase_c/train_gate.py

# 6) 阶段 D 配置（9 任务 × 3 组）
python phase_d/gen_configs.py
```

## 在线评测（需 GPT-5 API key）

```bash
python scripts/ai2thor/work/run_task.py \
  --config phase_d/configs/<task>__<group>.yaml   # group: A_baseline/B_rule_gate/C_voi_gate
python phase_d/aggregate.py <runs目录>            # 配对比较 + 显著性
```

## 云训练（检测头升级）

`phase_b/cloud_train.sh`：DINOv2-Base 冻结 + 336px + 宽头 + AMP，同一套代码。

云上数据准备：
```bash
# 1) 把数据放到云机（三部分，见"数据"一节），然后设置路径：
export LIGHTWM_DATA_ROOT=/data/lightwm_data
export LIGHTWM_COV_ROOT=/data/lightwm_data_cov
export LIGHTWM_FD_ROOT=/data/fd_benchmark_full_20260811_224644
# 2) 跑（建议 screen/tmux 挂后台，8-10 小时）：
bash phase_b/cloud_train.sh
```

参数对照：本地小档 `--variant small --resolution 224 --width 256`；
云上大档 `--variant base --resolution 336 --width 384 --amp`。

## 当前状态（截至 2026-08-25）

| 能力 | 状态 |
|---|---|
| 空间记忆（3D 锚点） | 方向 99%、转 180° 保持 100%、锚点误差 0.33m |
| 深度头（单目） | MAE 0.39m |
| 可行性头（动作成败） | 92.1% |
| VoI 门控（离线） | 精度 100%、零误报 |
| 稠密检测头 | 优先训练项（云训/更多 epoch） |
