#!/usr/bin/env bash
# Cloud / big-GPU training commands for Phase B (scale-up config).
# Requires: python venv with torch (cu12x), transformers, ~16GB+ GPU.
# Data: /mnt/d/lightwm_data mounted or copied to $DATA_DIR.
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
VENV="${VENV:-/home/sudidaren/lightwm_phases/.venv/bin/python}"
OUT="${OUT:-/home/sudidaren/lightwm_phases/checkpoints}"

# ---- Dense detector, DINOv2-BASE frozen, 336px, wide head, AMP ----
"$VENV" -u phase_b/train.py --task dense \
  --variant base --resolution 336 --width 384 --amp \
  --epochs 10 --batch 16 --lr 5e-4 --out "$OUT"

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
