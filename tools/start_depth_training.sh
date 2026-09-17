#!/bin/bash
# Start depth-head training on a machine that already has the pools + index.
#   bash tools/start_depth_training.sh [--epochs N] [--resolution N]
set -u
WM=/home/sudidaren/lightwm_phases
PY=/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python
DATA=/root/autodl-tmp
EPOCHS=${EPOCHS:-8}; RES=${RES:-448}; BATCH=${BATCH:-16}; WORKERS=${WORKERS:-8}
while [ $# -gt 0 ]; do
  case "$1" in
    --epochs) EPOCHS=$2; shift 2 ;;
    --resolution) RES=$2; shift 2 ;;
    --batch) BATCH=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

export LIGHTWM_DATA_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_COV_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_COV2_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_PROCTHOR_ROOT=$DATA/lightwm_data_procthor2
export LIGHTWM_PROCTHOR2_ROOT=$DATA/lightwm_data_procthor2
export LIGHTWM_VALHOUSE_ROOT=$DATA/lightwm_data_valhouses
export LIGHTWM_OBJVIEW_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_VIRTUALHOME_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_SPLITS=$WM/data/splits_noneval.json
# The box has no route to huggingface.co; the DINOv2 backbone must come from
# the local HF cache (shipped alongside the repo).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$WM" || exit 1
pkill -f 'train_depth_v[2].py' 2>/dev/null
sleep 2
setsid nohup "$PY" "$WM/phase_b/train_depth_v2.py" \
  --epochs "$EPOCHS" --resolution "$RES" --depth-size "$RES" --batch "$BATCH" \
  --workers "$WORKERS" --device cuda --amp \
  --out "$DATA/depth_v2" > /root/train_depth_v2.log 2>&1 < /dev/null &
sleep 12
echo -n "训练进程: "; ps -eo args | grep -c 'train_depth_v[2].py'
echo "日志: /root/train_depth_v2.log"
