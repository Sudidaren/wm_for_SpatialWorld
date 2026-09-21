#!/usr/bin/env bash
# 后台起本地 GPT-5 ProcTHOR 基线。独立脚本，避免 pgrep 自匹配。
set -uo pipefail
cd /home/sudidaren/lightwm_phases/tools
WORKERS="${WORKERS:-6}" setsid nohup bash run_gpt5_base_procthor_local.sh \
  >> /mnt/d/lightwm_out/gpt5_base_procthor.log 2>&1 < /dev/null &
sleep 50
echo "chain=$(pgrep -cf 'run_gpt5_base_proctho[r]' || true) worker=$(ps -eo cmd | grep -c '[w]ork.run_task') thor=$(ps -eo cmd | grep -c '[t]hor-Linux64')"
tail -8 /mnt/d/lightwm_out/closed_gpt5_base_procthor.log
free -g | sed -n 2p
