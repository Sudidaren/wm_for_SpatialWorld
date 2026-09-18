#!/bin/bash
# Run the v3 depth-head retrain ON THE CLOUD BOX (weste).
# Driven from the workstation by tools/run_cloud_depth_v3.py; can also be run
# by hand after `scp`-ing it over.
#
# What v3 fixes versus v2: v2 trained on 27 rooms of which ZERO were classic
# homes (FloorPlan1-30), while 57.7% of the objects visible in the evaluation
# rooms belong to that family -- so the head's metric scale collapsed there
# (predictions at 0.59-0.62x the truth).  v3 adds the classic pool, scene-
# balances the sampling, and photometrically jitters so the scale cannot be
# bound to the look of a particular house.
#
# Prerequisites on the box:
#   /root/autodl-tmp/lightwm_data_cov      <- uploaded by the driver (2.5 GB)
#   /root/autodl-tmp/lightwm_data_cov2
#   /root/autodl-tmp/lightwm_data_procthor2
#   /root/autodl-tmp/lightwm_data_valhouses
#   /root/.cache/huggingface/hub           <- DINOv2 (no route to huggingface.co)
set -u
WM=${LIGHTWM_ROOT:-/root/lightwm_phases}
DATA=/root/autodl-tmp
# The box's interpreter layout has changed between sessions, so probe for one
# that actually has torch with CUDA instead of hard-coding a path.
pick_py() {
  for cand in "${LIGHTWM_PY:-}" \
              /root/miniconda3/envs/ai2thor/bin/python \
              /root/miniconda3/bin/python \
              /opt/conda/envs/ai2thor/bin/python \
              /opt/conda/bin/python \
              /usr/bin/python3 "$(command -v python3 2>/dev/null)"; do
    [ -n "$cand" ] && [ -x "$cand" ] || continue
    if "$cand" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
         >/dev/null 2>&1; then echo "$cand"; return 0; fi
  done
  return 1
}
PY=$(pick_py) || {
  echo "no python with CUDA torch found; set LIGHTWM_PY=<interpreter>"; exit 1; }
echo "python: $PY ($("$PY" -c 'import torch;print(torch.__version__)'))"

export LIGHTWM_DATA_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_COV_ROOT=$DATA/lightwm_data_cov          # <- the classic homes
export LIGHTWM_COV2_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_PROCTHOR_ROOT=$DATA/lightwm_data_procthor2
export LIGHTWM_PROCTHOR2_ROOT=$DATA/lightwm_data_procthor2
export LIGHTWM_VALHOUSE_ROOT=$DATA/lightwm_data_valhouses
export LIGHTWM_OBJVIEW_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_VIRTUALHOME_ROOT=$DATA/lightwm_data_cov2
export LIGHTWM_SPLITS=$WM/data/splits_noneval.json
# The DINOv2 backbone comes from the local HF cache when it is there (the
# boxes have no route to huggingface.co); otherwise fall back to the mirror,
# which *is* reachable from AutoDL.  Deciding this here saves shipping a
# 745 MB cache to a box that can just download the 90 MB backbone.
if ls /root/.cache/huggingface/hub 2>/dev/null | grep -q dinov2; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  echo "backbone: local HF cache"
else
  export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
  echo "backbone: downloading from $HF_ENDPOINT"
fi
# keep the socket tmpdir off any fuse/network mount so DataLoader workers start
export TMPDIR=/root/tmp
mkdir -p "$TMPDIR" "$DATA/depth_v3"

cd "$WM" || { echo "no repo at $WM"; exit 1; }
echo "== pool sizes =="
for p in lightwm_data_cov lightwm_data_cov2 lightwm_data_procthor2 lightwm_data_valhouses; do
  printf '  %-24s episodes=%s\n' "$p" "$(ls "$DATA/$p/episodes" 2>/dev/null | wc -l)"
done

echo "== rebuilding the frame index (pools changed, the pickle is a cache) =="
"$PY" shared/data_index.py data/frame_index.pkl || exit 1
echo "== regenerating the split (train/val never touch the 81 evaluation rooms) =="
"$PY" tools/make_noneval_split.py || exit 1

echo "== v3 training =="
setsid nohup "$PY" phase_b/train_depth_v2.py \
    --epochs 12 --resolution 448 --depth-size 448 \
    --batch 16 --workers 8 --device cuda --amp \
    --jitter 0.3 --scene-balanced --epoch-samples 45000 \
    --splits data/splits_noneval.json \
    --ckpt checkpoints/small_objects_20260910/dense_depth_best.pt \
    --out "$DATA/depth_v3" > /root/train_depth_v3.log 2>&1 < /dev/null &

sleep 20
echo "== first lines of the log (the leak check must say 0) =="
head -8 /root/train_depth_v3.log
echo
echo "log : /root/train_depth_v3.log     weights: $DATA/depth_v3"
