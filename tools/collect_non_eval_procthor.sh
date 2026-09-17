#!/bin/bash
# Collect ProcTHOR-10k data for non-evaluation houses, then wait for the
# AI2-THOR coverage collection to finish if it is still running.
#
# The SpatialWorld ProcTHOR tasks run on the procTHOR-10k *train* partition
# (10,000 houses).  Houses from the **val** partition (1,000) can therefore
# never appear in the evaluation; we start at val index 105 because 0-104 are
# already in /mnt/d/lightwm_data_procthor.
#
# Policy matches the AI2-THOR coverage pool: positions (every 4th reachable
# cell) x 4 yaws x 3 horizons, RGB + metric depth + instance segmentation +
# pose + visible-object metadata.
#
# Usage:  bash tools/collect_non_eval_procthor.sh [--wait-ai2thor]
set -u
cd "$(dirname "$0")/.."

PY=~/SpatialWorld/envs/ai2thor/.venv/bin/python
OUT=${OUT:-/mnt/d/lightwm_data_procthor2}
LOG=${LOG:-/mnt/d}

if [ "${1:-}" = "--wait-ai2thor" ]; then
  while pgrep -f collect_coverage.py > /dev/null; do sleep 60; done
fi

for spec in "105 11 1" "116 10 2" "126 10 3"; do
  set -- $spec
  start=$1; houses=$2; shard=$3
  DISPLAY=${DISPLAY:-:0} setsid nohup "$PY" phase_b/collect_procthor.py \
      --split val --start "$start" --houses "$houses" \
      --yaws 4 --horizons 3 --position-step 4 --out "$OUT" \
      > "$LOG/collect_procthor2_shard$shard.log" 2>&1 < /dev/null &
  echo "procthor shard$shard: val $start..$((start+houses-1)) ($houses houses)"
  sleep 20
done
sleep 5
pgrep -cf collect_procthor.py
