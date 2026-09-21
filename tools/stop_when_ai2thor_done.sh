#!/usr/bin/env bash
# 用户要求：本机只跑 ai2thor，procthor 搬到卡上。
# supervisor 已经在内存里拿着完整清单（ai2thor 21 条在前，procthor 20 条在后），
# 改文件对它无效 —— 所以盯着 results.csv，ai2thor 那 21 条一跑完就立刻停。
set -uo pipefail
RES=/home/sudidaren/spatialworld_eval/runs/closed_gpt5_wm/results.csv
LST=/home/sudidaren/lightwm_phases/plans/gpt5_wm_ai2thor_left.txt
LOG=/mnt/d/lightwm_out/stop_when_ai2thor_done.log
TOTAL=$(grep -c . "$LST")

echo "[guard $(date '+%F %T')] 要等 $TOTAL 条 ai2thor 落地" >> "$LOG"
for i in $(seq 1 240); do
    got=0
    while read -r t; do
        [ -n "$t" ] || continue
        if grep -q "^$t," "$RES" 2>/dev/null; then got=$((got + 1)); fi
    done < "$LST"
    if [ $((i % 5)) -eq 1 ]; then
        echo "[guard $(date '+%F %T')] $got/$TOTAL" >> "$LOG"
    fi
    if [ "$got" -ge "$TOTAL" ]; then
        echo "[guard $(date '+%F %T')] ai2thor 全部落地，停本机批次（procthor 交给卡）" >> "$LOG"
        bash /home/sudidaren/lightwm_phases/tools/stop_gpt5_wm_local.sh >> "$LOG" 2>&1
        echo "[guard $(date '+%F %T')] 完成" >> "$LOG"
        exit 0
    fi
    sleep 30
done
echo "[guard $(date '+%F %T')] 超时退出（未全部落地）" >> "$LOG"
