#!/usr/bin/env bash
# Train everything together on the FULL dataset (original + coverage + FD).
# Run after all coverage collection finishes:
#   1) rebuild frame index (picks up coverage episodes)
#   2) dense detector  (Gaussian targets + cosine LR, 8 epochs)
#   3) depth head      (4 epochs, GT depth supervision)
#   4) feasibility     (6 epochs, LightWM + FD merge)
#   5) occupancy       (GPU, 20 epochs, U-Net ch=64)
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PY="${PY:-/home/sudidaren/lightwm_phases/.venv/bin/python}"
OUT="${OUT:-/home/sudidaren/lightwm_phases/checkpoints}"

echo "== 1/5 rebuild index =="
"$PY" shared/data_index.py

echo "== 2/5 dense detector (8 epochs) =="
"$PY" -u phase_b/train.py --task dense --epochs 8 --batch 24 \
  --lr 5e-4 --out "$OUT"

echo "== 3/5 depth head (4 epochs) =="
"$PY" -u phase_b/train.py --task depth --epochs 4 --batch 24 \
  --lr 3e-4 --out "$OUT"

echo "== 4/5 feasibility (6 epochs, +FD) =="
"$PY" -u phase_b/train.py --task feasibility --epochs 6 --batch 64 \
  --lr 3e-4 --out "$OUT"

echo "== 5/5 occupancy completion (20 epochs, GPU) =="
"$PY" -u phase_b/train_occupancy.py --epochs 20 --batch 128

echo "ALL DONE"
