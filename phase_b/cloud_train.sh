#!/usr/bin/env bash
# Cloud / big-GPU training commands for Phase B (scale-up config).
# Requires: python venv with torch (cu12x), transformers, ~16GB+ GPU.
# Data: /mnt/d/lightwm_data mounted or copied to $DATA_DIR.
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
VENV="${VENV:-$REPO_ROOT/.venv/bin/python}"
OUT="${OUT:-$REPO_ROOT/checkpoints}"
RES="${RES:-336}"          # 336 (4090) or 448 (A100, best for small objects)
EPOCHS="${EPOCHS:-12}"
BATCH="${BATCH:-16}"       # 448px: use 8 on 24GB

# ---- Dense detector, DINOv2-BASE frozen, 336px, wide head, AMP ----
"$VENV" -u phase_b/train.py --task dense \
  --variant base --resolution "$RES" --width 384 --amp \
  --epochs "$EPOCHS" --batch "$BATCH" --lr 5e-4 --eval-every 1 \
  --class-balanced --copy-paste --out "$OUT"

# ---- Depth head (same backbone) ----
"$VENV" -u phase_b/train.py --task depth \
  --variant base --resolution 336 --width 384 --amp \
  --epochs 5 --batch 16 --lr 3e-4 --out "$OUT"

# ---- Feasibility head (with FD merge) ----
"$VENV" -u phase_b/train.py --task feasibility \
  --variant base --resolution 336 --width 384 --amp \
  --epochs 8 --batch 64 --lr 3e-4 --out "$OUT"

# ---- Occupancy completion (small U-Net, cheap) ----
"$VENV" -u phase_b/train_occupancy.py --epochs 20 --batch 256

echo "Phase B cloud training done."
