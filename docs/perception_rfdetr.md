# 默认感知：RF-DETR Small + 单目深度

新的 `wingman_llm` 配置默认使用 `rfdetr_small_depth`，替换原 DINO 稠密检测头。
RF-DETR 接收 512×512 RGB，输出 117 类检测框；保留已训练的 DINOv2 单目深度模块。
两部分共 **54,962,173 参数**，其中检测器 32,208,316。推理不读取模拟器深度或语义分割。

## 安装与权重

Python 3.12 的已验证推理环境：

```bash
python -m pip install -r requirements-rfdetr.txt
export LIGHTWM_STORAGE_ROOT="${LIGHTWM_STORAGE_ROOT:-$PWD}"
python scripts/download_rfdetr.py
source scripts/selected_perception_env.sh
python scripts/check_rfdetr.py
python scripts/check_rfdetr.py --image /path/to/frame.png --device cuda
```

安装到**实际执行任务的 Python 环境**中；不要同时安装旧的 `requirements.txt`。
`eval_pipeline/eval_config.py` 默认仍使用各模拟器自己的环境，以保留模拟器依赖。
如使用共享环境，设置 `LIGHTWM_EVAL_PYTHON=/path/to/python`，该环境也必须安装所选模拟器及 GPT 客户端依赖。
本服务器的 RF-DETR 推理环境为
`/nfs-stor/junchi.yao/2027ICLR/wm_for_spatialworld/envs/rfdetr_1_10_1_probe/bin/python`。

**训练权重已公开发布到 [GitHub Release：perception-rfdetr-small-v1](https://github.com/Sudidaren/wm_for_SpatialWorld/releases/tag/perception-rfdetr-small-v1)**，无需登录或访问我们的 NFS。
Release 包含检测权重、深度权重、深度配置与类别表，以及 `SHA256SUMS` 和模型清单。
下载脚本会逐文件检查大小和 SHA256，跳过已校验的文件，并拒绝覆盖内容不匹配的已有权重。
仅需 Python 标准库即可下载，不需要先安装 PyTorch。

上面的命令将权重存到当前仓库的 `checkpoints/`；也可以先把 `LIGHTWM_STORAGE_ROOT` 设为自己的 NFS 路径。
本服务器继续使用 `/nfs-stor/junchi.yao/2027ICLR/wm_for_spatialworld`。以下文件与我们实际评测使用的文件逐字节一致：

| 文件 | 相对存储根目录的路径 |
|---|---|
| 检测权重 | `checkpoints/rfdetr_small_228094/checkpoint_best_total.pth` |
| 深度权重 | `checkpoints/small_objects_20260910/dense_depth_best.pt` |
| 深度配置／类别表 | `checkpoints/small_objects_20260910/dense_depth_best.pt.json` |

完整 SHA256、公开下载地址、文件大小、参数量和选择信息见 [权重清单](../configs/rfdetr_small.json)。
也可通过 `LIGHTWM_DETECTOR_PATH`、`PERCEPTION_CKPT` 分别指定本地路径。
深度骨干直接从深度检查点加载，不再额外下载 DINOv2 权重。缺少权重时明确报错，不回退到旧检测器。

检测 `.pth` 可以作为 RF-DETR Small 的 `pretrain_weights` 继续微调；它是评测选中的模型权重，
不是包含优化器状态的训练恢复检查点。深度 `.pt` 包含已有 DINOv2 编码器和已训练的感知头，
其 `.json` 必须一同保留，以确保分辨率和 117 类名称顺序一致。新数据继续微调时需单独准备对应数据与训练配置。

## 运行接入

```python
from phase_b.runtime_factory import build_runtime

perception = build_runtime({'device': 'cuda'})
result = perception(rgb_uint8)  # HWC RGB
# result: detections / depth（原图大小）/ rgb_shape
```

`eval_pipeline` 将后端、两个权重路径及阈值传给 runtime overlay 的 runner。
更新 SpatialWorld 时重新应用 `runtime_overlay/setup_lightwm_runtime.sh`；已有旧 overlay
不会因为拉取本仓库自动更新。设置 `SPATIALWORLD_ROOT` 指向你的 SpatialWorld checkout。
默认使用完整图像、阈值 **0.40**；`LIGHTWM_ZOOM=0`。`source` 配置和检查脚本不会发起 GPT 或任务评测。

旧 DINO 实验可显式设置 `LIGHTWM_DETECTOR=dino`，并设置对应的 `PERCEPTION_CKPT`、
`LIGHTWM_OBJ_THR`；旧模型代码保留用于复现。新检测器的类别表和参数量会在加载时检查。
上游 RF-DETR 导入会修改矩阵乘法精度；适配器保存并恢复调用者的设置。

## 已完成的评测

两个模型使用相同的 600 张独立测试图，7711 个目标，其中小物体 5791 个。
匹配要求类别正确且 IoU≥0.5；小物体定义为归一化框面积 <(32/224)²。
各模型阈值只在验证集选择。RF-DETR 的 600 张验证图 F1 为 80.06%，精确率 84.62%，小物体召回率 69.86%。

| 模型 | 测试 F1 | 精确率 | 小物体召回率 | 含深度参数量 | 感知平均耗时 |
|---|---:|---:|---:|---:|---:|
| DINO，全图 | 35.85% | 40.59% | 26.66% | 2489 万 | 19.8 ms |
| DINO，九个裁剪加全图 | 39.14% | 41.80% | 39.82% | 2489 万 | 145.0 ms |
| RF-DETR，全图 | **72.27%** | **80.15%** | **60.71%** | **5496 万** | **33.6 ms** |

耗时在同一 A100 上用 24 张相同图片各重复三次，包含检测、深度、缩放、传输和 NMS，
排除读文件及初始化。DINO 和 RF-DETR 使用不同依赖环境，但矩阵乘法 TF32 均关闭、cuDNN TF32 均开启。
RF-DETR 峰值张量显存约 265 MiB。共享节点上的顺序测速不代表 GPT 任务总耗时。
训练使用 43,004 张图，20 轮，两张 A100，零排队，耗时 2 小时 51 分 59 秒；选择第 15 轮权重。
此次同时更换检测器、微调骨干、提高分辨率，不能将增益归因于单一因素。叉子与勺子测试召回仍仅 25% 和 18.75%。

## GPT 与 WM+GPT 的已封存小样本

| 任务 | GPT-only | WM+GPT |
|---|---:|---:|
| AI2-THOR，10 个 | 2/10 | 7/10 |
| ProcTHOR，7 个 | 0/7 | 0/7 |
| 合计，17 个 | 11.8% | 41.2% |

这批使用旧 OWLv2＋深度模块（1.777 亿参数），**不是本次新模型的任务成绩**。
另有三个任务因 API 额度中断，未计为失败。样本小，精确 McNemar p=0.0625；
AI2-THOR 子集没有感知测试场景，ProcTHOR 还受历史交互反馈缺陷影响。
WM 平均每任务增加约 4.13 万 token、64.8 秒。按用户要求，任务评测保持停止。

原始预测、协议和统计存于 NFS 的 `runs/rfdetr_final_comparison_228094`，
任务快照为 `runs/paired_pilot20_20260911/completed_17`。
