#!/usr/bin/env bash
# 重启卡上的 Xvfb，然后（可选）跑一次双 reset 探测。
#   bash restart_xvfb.sh            # 只重启 + 健康检查
#   bash restart_xvfb.sh FloorPlan14 300   # 重启后测一次双 reset
#
# 单独成脚本的原因：直接在 bash -c 里写 "Xvfb :99"，pgrep -f 会匹配到执行
# 那条命令的 shell 自己，把它当场杀掉（输出直接空白）。
set -uo pipefail
SCENE="${1:-}"
TO="${2:-300}"
export DISPLAY=:99 TMPDIR=/tmp/lightwm_tmp

for p in $(pgrep -f "Xvfb :9[9]"); do kill -9 "$p" 2>/dev/null || true; done
sleep 3
setsid nohup Xvfb :99 -screen 0 1280x1024x24 > /root/autodl-tmp/xvfb.log 2>&1 </dev/null &
sleep 6
echo "xvfb_pid=$(pgrep -cf 'Xvfb :9[9]')"
DISPLAY=:99 xdpyinfo 2>&1 | head -2

if [ -n "$SCENE" ]; then
    for t in $(pgrep -f "thor-Linux6[4]"); do kill -9 "$t" 2>/dev/null || true; done
    for p in $(pgrep -f "double_reset_prob[e]"); do kill -9 "$p" 2>/dev/null || true; done
    sleep 2
    setsid nohup /home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python -u \
        /root/double_reset_probe.py "$SCENE" "$TO" \
        > /root/autodl-tmp/dr.log 2>&1 </dev/null &
    echo "probe started $(date +%T)"
fi
