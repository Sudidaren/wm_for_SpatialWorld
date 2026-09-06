# 复现 dense 检测头（与本地完全一致）

目标：同伙 clone 本仓库后，跑出与 2026-09-05 本地 dense 检测头（
`checkpoints_local/dense_best.pt`，R@IoU0.5 ≈ 0.32–0.40）一致的训练与评测。

## 0. 环境

- 系统：Ubuntu（WSL 也可），Python 3.12.3，NVIDIA GPU（6GB 起步，建议 ≥12GB）
- 依赖见 `requirements.txt`（torch 2.11.0+cu128 / torchvision 0.26.0+cu128）
- 建 venv：
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  ```

## 1. 数据（与本地一致）

训练帧来自 6 个数据池（`frame_index.pkl` 汇总），数据包在 HuggingFace：
`Sudidaren/lightwm-data`（若为私有，需要给同伙开 read 权限）。

```bash
export LIGHTWM_DATA_ROOT=/data/lightwm/lightwm_data
export LIGHTWM_COV_ROOT=/data/lightwm/lightwm_data_cov
export LIGHTWM_OBJVIEW_ROOT=/data/lightwm/lightwm_data_objviews
export LIGHTWM_FD_ROOT=/data/lightwm/fd_benchmark_full_20260811_224644
export LIGHTWM_PROCTHOR_ROOT=/data/lightwm/lightwm_data_procthor
export LIGHTWM_VIRTUALHOME_ROOT=/data/lightwm/lightwm_data_virtualhome
```

用 `cloud_setup.sh`（会 clone 本仓库、装环境、从 HF 下载并解压 6 个 tar），
或手动执行：

```bash
python shared/data_index.py          # 重建 frame_index.pkl（含所有池）
```

> 数据切分（训练/验证/考试房间）用 `data/splits_v2.json`（scene 级，
> seed 42，test 房零训练帧，类别×尺寸覆盖检查通过）。生成器：
> `shared/make_splits_v2.py`。

## 2. 训练 dense 头（本地 2026-09-05 同款配置）

```bash
python phase_b/train.py --task dense \
  --variant small --resolution 224 --width 256 --amp \
  --epochs 12 --batch 32 --lr 3e-4 --workers 12 \
  --class-balanced --copy-paste --eval-every 1 \
  --out checkpoints_local
```

云端放大版（base@336/448，见 `phase_b/cloud_train.sh`）：

```bash
RES=336 EPOCHS=12 BATCH=16 bash phase_b/cloud_train.sh
```

训练内验证：`obj_thr=0.25` 同时报 IoU>0.5 / IoU>0.3 的 P/R。
本地 12ep 参考曲线（R@IoU0.5）：0.213→0.319。

## 3. 评测

```bash
python phase_b/eval_perception.py --ckpt checkpoints_local/dense_best.pt \
  --variant small --resolution 224
```

参考结果（独立 60 帧，obj_thr=0.25）：P=0.203 / R=0.396；
按 GT 尺寸分桶：≤16px R≈0.03、17–48px R≈0.16、49–144px R≈0.46、>144px R≈0.52。

## 4. 权重

- 本地最优：`checkpoints_local/dense_best.pt`（100MB，small@224 修复版，12ep）
- 云端 base@336 三头：`checkpoints_cloud/{dense,depth,feasibility}_best.pt`（各 372MB）
- 权重体积超出 GitHub 限制，托管在 HuggingFace（路径与 SHA256 待定/见 README 更新）。

## 5. 已知差异点（保证"一致"要一起看）

- 检测头关键修复（相对老 commit）：DINOv2 输入 ImageNet 归一化；dense 目标改
  高斯晕 obj>0.3 全片监督；cls 权重 1.5；decode grid 自动取特征图尺寸；
  adapter 尚未加入 dense 优化器（下次训练要加）。
- 随机性：固定 seed 可复现数据顺序；GPU 浮点仍有极小差异，允许
  ±0.005 的 P/R 抖动。
- `data/frame_index.pkl` 是本地索引缓存，不入库；首次必须重建或下载 HF 数据包。
