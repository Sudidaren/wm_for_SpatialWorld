#!/usr/bin/env bash
# Source before launching new evaluations; does not start any task.
export LIGHTWM_STORAGE_ROOT="${LIGHTWM_STORAGE_ROOT:-/nfs-stor/junchi.yao/2027ICLR/wm_for_spatialworld}"
export LIGHTWM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LIGHTWM_DETECTOR=rfdetr_small_depth
export LIGHTWM_DETECTOR_PATH="$LIGHTWM_STORAGE_ROOT/checkpoints/rfdetr_small_228094/checkpoint_best_total.pth"
export PERCEPTION_CKPT="$LIGHTWM_STORAGE_ROOT/checkpoints/small_objects_20260910/dense_depth_best.pt"
export LIGHTWM_OBJ_THR=0.40
export LIGHTWM_ZOOM=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Keep simulator environments intact. Install requirements-rfdetr.txt in the
# worker environments, or explicitly provide a fully equipped worker Python.
