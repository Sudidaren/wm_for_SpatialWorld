#!/usr/bin/env bash
# 三张卡 vLLM 隧道的守护：每 30 秒探一次，断了就重建。
#
# 为什么要它：paramiko 隧道会静默失效 —— **进程还在，但通道已经死了**
# （curl 返回 000）。2026-09-21 16:5x 本机补测就是被这个卡住的：5 个 worker
# 卡在第一次 LLM 调用上 600 秒，而机器负载是 0.02（完全空闲）。
#
# 用法: setsid nohup bash tunnel_keeper.sh >/dev/null 2>&1 &
set -uo pipefail
SWE=/home/sudidaren/spatialworld_eval
LOG=/mnt/d/lightwm_out/tunnel_keeper.log
say() { echo "[tk $(date '+%F %T')] $*" >> "$LOG"; }

start_one() {   # $1=本地端口 $2=主机 $3=端口 $4=密码
    for p in $(pgrep -f "tunnel\.py $1 "); do kill -9 "$p" 2>/dev/null || true; done
    sleep 1
    ( cd "$SWE"
      AUTODL_HOST_OVERRIDE="$2" AUTODL_PORT_OVERRIDE="$3" AUTODL_PW="$4" \
      setsid nohup python3 tunnel.py "$1" 8000 >> "/mnt/d/lightwm_out/tunnel_$1.log" 2>&1 </dev/null & )
}

ok() { curl -s -m 6 -o /dev/null http://127.0.0.1:$1/v1/models 2>/dev/null; }

CARDS=("18000 connect.westd.seetacloud.com 25713 ZkYWHfECEihE"
       "18001 connect.westd.seetacloud.com 33177 tFvwsBM9d5a3"
       "18002 connect.westc.seetacloud.com 47454 l35oYyWKBqhy")

while true; do
    for c in "${CARDS[@]}"; do
        set -- $c
        if ! ok "$1"; then
            say "隧道 $1 断了 → 重建"
            start_one "$1" "$2" "$3" "$4"
            sleep 12
            ok "$1" && say "隧道 $1 已恢复" || say "隧道 $1 仍未通"
        fi
    done
    sleep 30
done
