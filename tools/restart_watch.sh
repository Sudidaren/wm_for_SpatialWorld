#!/usr/bin/env bash
# 重启闭源批次看护（改过目标数/新增 WM 臂之后用）。
set -uo pipefail
for p in $(pgrep -f "watch_close[d]"); do kill -TERM "$p" 2>/dev/null || true; done
sleep 2
cd /home/sudidaren/lightwm_phases/tools
setsid nohup python3 watch_closed.sh >> /mnt/d/lightwm_out/closed_watch.log 2>&1 < /dev/null &
sleep 35
echo "watch=$(pgrep -cf 'watch_close[d]')"
tail -1 /mnt/d/lightwm_out/closed_watch.log
