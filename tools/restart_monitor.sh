#!/usr/bin/env bash
# 重启巡检循环（改了 monitor_status.py 之后要重启才生效——老进程里跑的是
# 旧代码）。脚本名不含 "monitor_status" 字样，避免 pgrep -f 误伤自己。
set -uo pipefail
cd "$(dirname "$0")"
for p in $(pgrep -f "monitor_statu[s]"); do kill "$p" 2>/dev/null || true; done
sleep 2
setsid nohup python3 monitor_status.py --loop 240 \
    >> /mnt/d/lightwm_out/status_all.out 2>&1 </dev/null &
sleep 10
echo "monitor_now=$(pgrep -cf 'monitor_statu[s]' || true)"
tail -1 /mnt/d/lightwm_out/status_all.log
