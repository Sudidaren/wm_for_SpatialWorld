#!/usr/bin/env bash
# 后台起本机 Gemini+WM 批次（6 worker）。独立脚本，避免 pgrep 自匹配。
set -uo pipefail
cd /home/sudidaren/lightwm_phases/tools
WORKERS=6 setsid nohup bash run_gemini_local.sh full \
  >> /mnt/d/lightwm_out/gemini_local.log 2>&1 < /dev/null &
sleep 45
echo "chain=$(pgrep -cf 'run_gemini_loca[l]') worker=$(ps -eo cmd | grep -c '[w]ork.run_task') thor=$(ps -eo cmd | grep -c '[t]hor-Linux64')"
echo "--- no_proxy 是否带上网关 ---"
grep -m1 'no_proxy=' /proc/$(pgrep -f 'run_gemini_loca[l]' | head -1)/environ 2>/dev/null | tr '\0' '\n' | grep -i no_proxy | head -2
tail -3 /mnt/d/lightwm_out/closed_gemini_wm.log
