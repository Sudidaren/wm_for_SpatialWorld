#!/bin/bash
# Collect AI2-THOR coverage data for the non-evaluation rooms.
#
# The 31 scenes below are the classic rooms that the SpatialWorld evaluation
# never uses (81 of the 120 rooms are evaluation rooms; 8 of the remaining 39
# are already covered by the main/coverage/objectview pools).
#
# Policy (same as the existing coverage pool, so the two are comparable):
#   every reachable grid cell x 4 yaws x 3 horizons (0, -30, +30)
#   RGB + metric depth + instance segmentation + pose + visible-object metadata
#
# Writes to /mnt/d/lightwm_data_cov2 and logs to /mnt/d/collect_cov2_shard*.log.
set -u
cd "$(dirname "$0")/.."

PY=~/SpatialWorld/envs/ai2thor/.venv/bin/python
OUT=${OUT:-/mnt/d/lightwm_data_cov2}
LOG=${LOG:-/mnt/d}

SHARD1="FloorPlan204 FloorPlan207 FloorPlan209 FloorPlan221 FloorPlan222 FloorPlan223 FloorPlan224 FloorPlan225 FloorPlan226 FloorPlan227 FloorPlan228"
SHARD2="FloorPlan229 FloorPlan230 FloorPlan309 FloorPlan325 FloorPlan327 FloorPlan328 FloorPlan403 FloorPlan404 FloorPlan405 FloorPlan406"
SHARD3="FloorPlan407 FloorPlan408 FloorPlan410 FloorPlan411 FloorPlan413 FloorPlan414 FloorPlan416 FloorPlan419 FloorPlan420 FloorPlan421"

for i in 1 2 3; do
  eval "SCENES=\$SHARD$i"
  DISPLAY=${DISPLAY:-:0} setsid nohup "$PY" phase_b/collect_coverage.py \
      --scenes $SCENES --step 0.5 --yaws 4 --horizons 3 --out "$OUT" \
      > "$LOG/collect_cov2_shard$i.log" 2>&1 < /dev/null &
  echo "shard$i started: $(echo $SCENES | wc -w) scenes"
done
sleep 5
pgrep -cf collect_coverage.py
