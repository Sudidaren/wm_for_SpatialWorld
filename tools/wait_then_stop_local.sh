#!/usr/bin/env bash
# 等当前在飞的这几条 ai2thor 落地，然后立刻停掉本机批次。
# 用户 2026-09-22 00:0x 要求：本机不再跑 ai2thor。
set -uo pipefail
RES=/home/sudidaren/spatialworld_eval/runs/closed_gpt5_wm/results.csv
LOG=/mnt/d/lightwm_out/wait_then_stop.log
WANT="ai2thor05528 ai2thor05534 ai2thor05536 ai2thor05537 ai2thor05541"

echo "[wait $(date '+%F %T')] 等落地: $WANT" >> "$LOG"
for i in $(seq 1 40); do
    got=0
    for t in $WANT; do
        if grep -q "^$t," "$RES" 2>/dev/null; then got=$((got + 1)); fi
    done
    echo "[wait $(date '+%F %T')] 已落地 $got/5" >> "$LOG"
    [ "$got" -ge 5 ] && break
    sleep 30
done
echo "[wait $(date '+%F %T')] 停本机批次" >> "$LOG"
bash /home/sudidaren/lightwm_phases/tools/stop_gpt5_wm_local.sh >> "$LOG" 2>&1
echo "[wait $(date '+%F %T')] 完成" >> "$LOG"
