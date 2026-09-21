#!/usr/bin/env bash
# 跑双 reset 探测（单独成脚本：直接在 bash -c 里写目标进程名，pgrep -f 会
# 匹配到执行命令的那个 shell 自己，把人家的命令当场杀掉）。
set -uo pipefail
SCENE="${1:-FloorPlan14}"
TO="${2:-300}"
export DISPLAY=:99 TMPDIR=/tmp/lightwm_tmp
for p in $(pgrep -f "double_reset_prob[e]"); do kill -9 "$p" 2>/dev/null || true; done
for t in $(pgrep -f "thor-Linux6[4]"); do kill -9 "$t" 2>/dev/null || true; done
sleep 2
setsid nohup /home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python -u \
    /root/double_reset_probe.py "$SCENE" "$TO" \
    > /root/autodl-tmp/dr.log 2>&1 </dev/null &
echo "started $(date +%T)"
