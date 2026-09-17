#!/bin/bash
# One-shot: rebuild the frame index over the non-evaluation pools, write the
# non-eval split, then start depth-head training (scale-aware loss, 448).
#
# Run this ON the training box after the data has been synced.
#   bash tools/train_depth_on_cloud.sh [--epochs 8] [--resolution 448]
set -u
WM=/home/sudidaren/lightwm_phases
PY=/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python
DATA=/root/autodl-tmp
EPOCHS=8; RES=448; BATCH=16; WORKERS=8
while [ $# -gt 0 ]; do
  case "$1" in
    --epochs) EPOCHS=$2; shift 2 ;;
    --resolution) RES=$2; shift 2 ;;
    --batch) BATCH=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

# The other pools live locally on the workstation, not on this box, so point the
# index at what is actually here.
export LIGHTWM_DATA_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_COV_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_COV2_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_PROCTHOR_ROOT=$DATA/lightwm_data_procthor2
export LIGHTWM_PROCTHOR2_ROOT=$DATA/lightwm_data_procthor2
export LIGHTWM_VALHOUSE_ROOT=$DATA/lightwm_data_valhouses
export LIGHTWM_OBJVIEW_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_VIRTUALHOME_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_SPLITS=$WM/data/splits_noneval.json

cd "$WM" || exit 1
echo "== 1/3 rebuild frame index =="
$PY - <<'PY'
import os, sys
sys.path.insert(0, "/home/sudidaren/lightwm_phases")
from shared.data_index import build_index, DEFAULT_INDEX
idx = build_index()
print("index:", DEFAULT_INDEX)
print("frames:", len(idx["frames"]))
scenes = {}
for fr in idx["frames"]:
    s = str(fr.get("scene") or "")
    scenes[s] = scenes.get(s, 0) + 1
print("scenes:", len(scenes))
for s in sorted(scenes)[:8]:
    print(f"   {s}: {scenes[s]}")
PY

echo "== 2/3 (re)generate the non-eval split =="
$PY "$WM/tools/make_noneval_split.py" 2>/dev/null || echo "(split file shipped in the repo, reusing it)"

echo "== 3/3 train =="
setsid nohup $PY "$WM/phase_b/train_depth_v2.py" \
  --epochs "$EPOCHS" --resolution "$RES" --depth-size "$RES" --batch "$BATCH" \
  --workers "$WORKERS" --device cuda --amp \
  --out "$DATA/depth_v2" > /root/train_depth_v2.log 2>&1 < /dev/null &
sleep 10
echo -n "训练进程: "; ps -eo args | grep -c '[t]rain_depth_v2'
echo "日志: /root/train_depth_v2.log"
