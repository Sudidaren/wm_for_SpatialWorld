#!/usr/bin/env bash
# 后台起剩余 13 条 procthor WM。独立脚本，避免 pgrep 自匹配。
set -uo pipefail
cd /home/sudidaren/lightwm_phases/tools
WORKERS="${WORKERS:-6}" setsid nohup bash run_procthor_wm_left_local.sh \
  >> /mnt/d/lightwm_out/procthor_wm_left.log 2>&1 < /dev/null &
sleep 50
echo "chain=$(pgrep -cf 'run_procthor_wm_left_loca[l]' || true) worker=$(ps -eo cmd | grep -c '[w]ork.run_task') thor=$(ps -eo cmd | grep -c '[t]hor-Linux64')"
tail -6 /mnt/d/lightwm_out/closed_gpt5_wm_procthor.log
free -m | sed -n 2p
