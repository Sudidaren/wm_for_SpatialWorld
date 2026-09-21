#!/usr/bin/env bash
# 后台起本机 GPT-5+WM 剩余批次。独立脚本，避免 pgrep 自匹配。
set -uo pipefail
cd /home/sudidaren/lightwm_phases/tools
WORKERS="${WORKERS:-6}" setsid nohup bash run_gpt5_wm_local.sh run \
  >> /mnt/d/lightwm_out/gpt5_wm_local.log 2>&1 < /dev/null &
sleep 50
echo "chain=$(pgrep -cf 'run_gpt5_wm_loca[l]') worker=$(ps -eo cmd | grep -c '[w]ork.run_task') thor=$(ps -eo cmd | grep -c '[t]hor-Linux64')"
tail -6 /mnt/d/lightwm_out/closed_gpt5_wm.log
free -g | sed -n 2p
