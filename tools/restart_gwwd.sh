#!/usr/bin/env bash
# 重启网关看门狗。单独写成脚本是因为：直接在 `bash -c '... pkill -f
# gateway_watchdog ...'` 里写这个模式，远程/本地 shell 自己的命令行里就带
# 这个字样，pkill -f 会连执行命令的 shell 一起杀掉（表现为整条命令 143、
# 什么都没跑成）。脚本名里没有那个字样就不会误伤。
set -uo pipefail
cd "$(dirname "$0")"
for p in $(pgrep -f "gateway_watchdo[g]"); do kill "$p" 2>/dev/null || true; done
sleep 2
setsid nohup python3 gateway_watchdog.py \
    >> /mnt/d/lightwm_out/gateway_watchdog.log 2>&1 </dev/null &
sleep 8
echo "watchdog_now=$(pgrep -cf 'gateway_watchdo[g]' || true)"
tail -3 /mnt/d/lightwm_out/gateway_watchdog.log
