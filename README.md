# LightWM 阶段 B/C/D 实现（2026-08-25 起）

目标：世界模型必须能**记忆 / 理解 / 预测空间关系**，然后在正确时机给正确信息
（VoI 门控）。本目录是阶段 B/C/D 的代码与实验结果。

## 环境
- venv：`.venv/`（torch 2.11.0+cu128，GPU：RTX 4050 6GB）
- 数据：`/mnt/d/lightwm_data`（1067 集 / 34326 帧，GT bbox/depth/动作标签）
- FD 轨迹（门控训练）：`/mnt/d/fd_benchmark_full_20260811_224644`
- DINOv2 权重经 HF 镜像：`HF_ENDPOINT=https://hf-mirror.com`

## 数据集（2026-08-25 扩充后）
- 原始集：34,326 帧 / 1067 集 / 117 场景（GT bbox+深度+动作）
- 穷举覆盖采集（`phase_b/collect_coverage.py`）：**5,316 帧 / 12 场景**
  每场景所有可达位置（GetReachablePositions）× 4 朝向 × 3 俯仰
  （含 FloorPlan1/2/3/4/5/6/7/10/12/17/20/25；FloorPlan8 无导航网格跳过）
- 合并索引：**38,641 帧 / 117 类物体**（`data/frame_index.pkl`）
- FD 轨迹：3,618 帧（RGB+动作+错误标签，可行性头用，错误分布更全）

## 坐标约定（已用数据验证，见 `shared/test_geometry.py`）
- 相机 = agent.position + (0, 0.675, 0)（AI2-THOR cameraY 默认值）
- yaw：fwd = (sin(yaw), cos(yaw))，right = (cos(yaw), -sin(yaw))
- horizon：负=抬头、正=低头（ManipulaTHOR 约定），已加入投影/反投影
- depth PNG = 米 × 1000（uint16），为前向深度

## 坐标约定（已用数据验证，见 `shared/test_geometry.py`）
- 相机 = agent.position + (0, 0.675, 0)（AI2-THOR cameraY 默认值）
- yaw：fwd = (sin(yaw), cos(yaw))，right = (cos(yaw), -sin(yaw))
- depth PNG = 米 × 1000（uint16），为前向深度
- 已验证：地面/台面反投影高度自洽；旋转后方向三角化一致

## 架构（四层）
1. **感知**（`phase_b/model.py`，可训练 ~2.8-3.1M 参数）
   DINOv2-S 冻结 → 2 层 Conv 适配器 → Slot Attention(16槽) →
   box/objectness/class 头；**CenterNet 式稠密检测头**（高斯目标+NMS，
   当前主用）+ 深度头（训练时用 GT depth，运行时只用 RGB）
2. **持久物体记忆**（`shared/spatial_memory.py`）
   3D 锚点跨帧融合（多视角关联 + EMA 位置更新 + 置信度），
   查询输出 方向/距离档/高低档/置信度
3. **场景图**（`shared/spatial_memory.py::scene_graph`）
   物体↔物体关系边（on/in/above/left/right + 距离）
4. **占用补全**（`phase_b/occupancy.py`，U-Net <0.5M 参数）
   部分可走地图 → 预测遮挡后的可走区域（"桌子后面有空地"）

## 命令
```bash
# 数据索引（已生成 data/frame_index.pkl）
.venv/bin/python shared/data_index.py

# 阶段 B 训练（GPU 需在沙箱外运行）
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -u phase_b/train.py \
  --task perception --epochs 4 --batch 24 --out checkpoints
# 同上 --task depth / --task feasibility

# 占用补全训练
.venv/bin/python -u phase_b/train_occupancy.py --epochs 10

# 评测
.venv/bin/python phase_b/eval_spatial.py --max-episodes 60
.venv/bin/python phase_b/eval_perception.py --ckpt checkpoints/perception_best.pt

# 阶段 C 门控
.venv/bin/python phase_c/extract_candidates.py   # 生成 data/gate_fd.npz
.venv/bin/python phase_c/train_gate.py           # 训练 + 离线拦截率

# 阶段 D 评测配置生成
.venv/bin/python phase_d/gen_configs.py          # 9 任务 × 3 组 = 27 配置
```

## 当前结果（2026-08-25 晚，全量训练后）
- 空间记忆（GT 输入、80 集 2100 锚点）：锚点中位误差 0.33 m；
  方向正误 99%（中位偏角 1.0°）；高度档 85%；距离档 90%；
  **转过 >150° 后记忆保持率 100%**
- 门控（FD 离线，验证集 98 候选）：阈值 0.3 → 精度 100%、召回 0.79、零误报
  （规则基线 20.8% 总拦截 / 手占 100% 但含 23 个误报）
- 全量训练（`phase_b/train_all.sh`，38.6k 帧 + FD 合并）：
  - 稠密检测 8 epoch：loss 2.33→1.58；thr=0.5 时 P=0.079 / R=0.080
    （检测头是弱项，云上 base 档 + 更多 epoch 是下一步）
  - **深度头：MAE 0.39 m**（56×56，判断距离达标）
  - **可行性头：动作成败预测准确率 92.1%**（失败原因分类 21%，弱）
  - 占用补全：未见过可走格预测弱（正类仅 0.03%，标注为难点）

## 运行时集成（休眠，等离线验证通过再开）
- `shared/runtime_bridge.py`：SpatialMemoryAdapter（锚点提示）+
  GateAdapter（打分放行 0-1 条）+ HintRenderer（**绕行提示**：
  "绕行时直线距离会先增后减，请继续绕行而不是折返"）
- 接入点：`mllm_base_agent/agent/memory_probe.py` 的探索建议位置；
  配置项 `memory_probe.gate: none|rule|learned`

## 已知问题 / 下一步
- 感知头用 GT depth 训练深度头；运行时换单目深度（当前 depth MAE 待训后复测）
- 门控数据仅 488 条（FD 规则可覆盖部分）；阶段 C 需要加入 LightWM 运行日志
  （steps.jsonl 含 mem_hint + 推理 + 动作）扩充正负样本
- 阶段 D 的 GPT-5 在线评测需要 API key；配置已生成，跑法见
  `scripts/ai2thor/work/run_task.py --config phase_d/configs/<task>__<group>.yaml`
- 云/大 GPU 训练：`phase_b/cloud_train.sh`（base 档 336px + AMP + 宽头）
