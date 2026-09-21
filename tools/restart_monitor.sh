#!/usr/bin/env bash
# 重启 monitor_status（本机 run 已切到 closed_gemini_wm）。
# 独立成脚本：命令行里写 pgrep -f "monitor_status"，会匹配到自己的 shell。
set -uo pipefail
for p in $(pgrep -f "monitor_statu[s]"); do kill -TERM "$p" 2>/dev/null || true; done
sleep 3
cd /home/sudidaren/lightwm_phases/tools
setsid nohup python3 monitor_status.py --loop 300 >> /mnt/d/lightwm_out/monitor_status.log 2>&1 < /dev/null &
sleep 30
echo "monitor=$(pgrep -cf 'monitor_statu[s]')"
tail -1 /mnt/d/lightwm_out/status_all.log
