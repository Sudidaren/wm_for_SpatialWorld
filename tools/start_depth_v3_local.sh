#!/bin/bash
# Start the v3 depth-head retrain on the local GPU (handover route B).
#
# Why this script exists rather than the bare command in the handover doc:
#
# 1. TMPDIR must NOT live on /mnt/d.  The D: drive is a Windows (drvfs) mount
#    and cannot host the unix sockets `multiprocessing` creates, so
#    `--workers > 0` dies with `OSError: [Errno 95] Operation not supported`.
#    Only the socket dir is moved to ext4; every large artefact still goes to D:.
# 2. HF_HOME must point at the D: cache, because the DINOv2 weights are only
#    cached there (there is no ~/.cache/huggingface/hub on this box).
#
# Log: /mnt/d/lightwm_out/train_v3.log     Weights: /mnt/d/lightwm_out/depth_v3
#
# Usage:  bash tools/start_depth_v3_local.sh          (background, ~2.5 h)
#         tail -f /mnt/d/lightwm_out/train_v3.log     (watch)
#
# Stop it by PID, never with a pattern: `pkill -f train_depth_v2` matches the
# shell that runs it too (see the handover doc's list of past mistakes).

set -u
cd "$(dirname "$0")/.."

export HF_HOME=/mnt/d/lightwm_cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/tmp/lightwm_tmp
mkdir -p "$TMPDIR" /mnt/d/lightwm_out

PY=/mnt/d/lightwm_venvs/depth/bin/python
OUT=/mnt/d/lightwm_out/depth_v3
LOG=/mnt/d/lightwm_out/train_v3.log

setsid nohup "$PY" phase_b/train_depth_v2.py \
    --epochs 12 --resolution 448 --depth-size 448 \
    --batch 16 --workers 6 --device cuda --amp \
    --jitter 0.3 --scene-balanced --epoch-samples 45000 \
    --splits data/splits_noneval.json \
    --ckpt checkpoints/small_objects_20260910/dense_depth_best.pt \
    --out "$OUT" > "$LOG" 2>&1 < /dev/null &

sleep 5
echo "started pid $! -> $LOG"
echo "the first lines to check are:"
echo "  leak check: 0/179733 training frames from the 81 evaluation rooms"
echo "  scene-balanced sampling: ... frames/epoch"
