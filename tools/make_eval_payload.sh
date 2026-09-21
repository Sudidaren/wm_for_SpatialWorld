#!/bin/bash
# 为一个评测卡打包：代码 + 任务数据 + 感知权重
set -euo pipefail
OUT=/mnt/d/eval_card_payload
mkdir -p "$OUT"
cd /home/sudidaren
echo "[1/3] lightwm_phases 代码+权重"
tar cf "$OUT/lightwm_phases.tar" \
  --exclude='__pycache__' --exclude='*.pyc' \
  lightwm_phases/phase_b lightwm_phases/shared lightwm_phases/runtime_overlay \
  lightwm_phases/tools lightwm_phases/configs lightwm_phases/scripts \
  lightwm_phases/data/eval_rooms.json lightwm_phases/data/splits_noneval.json \
  lightwm_phases/data/inventory.json lightwm_phases/requirements-rfdetr.txt \
  lightwm_phases/checkpoints/rfdetr_small_228094 \
  lightwm_phases/checkpoints/small_objects_20260910 \
  lightwm_phases/checkpoints/da2_metric_indoor_small
echo "[2/3] spatialworld_eval 评测 harness"
tar cf "$OUT/spatialworld_eval.tar" --exclude='__pycache__' --exclude='runs' \
  --exclude='*.pyc' --exclude='__pycache__' \
  spatialworld_eval
echo "[3/3] SpatialWorld 任务数据 + 代码"
# 2026-09-20 教训：只打 data/mllm_base_agent/evaluation/experiments/tests 会漏掉
# scripts/ —— 评测入口就是 `python -m scripts.ai2thor.work.run_task`，卡上会报
# ModuleNotFoundError: No module named 'scripts'。actions/assets/configs/core
# 同理（动作解析、资源、配置、核心模块）。这几个加起来才 3.4MB，别省。
tar cf "$OUT/spatialworld.tar" --exclude='__pycache__' --exclude='*.pyc' \
  SpatialWorld/data SpatialWorld/mllm_base_agent SpatialWorld/evaluation \
  SpatialWorld/experiments SpatialWorld/tests \
  SpatialWorld/scripts SpatialWorld/actions SpatialWorld/configs \
  SpatialWorld/core SpatialWorld/assets
ls -la "$OUT" | awk '{print $5/1e6" MB", $NF}'
