#!/usr/bin/env bash
# 停掉卡上所有 TVR 相关进程，并把已产出的 runs/ 挪到 /root/tvr_runs_<tag>_<时间>
# （不删，留着对照）。
#
# 单独成脚本的原因：直接在 bash -c 里写进程名，pgrep/pkill -f 会匹配到执行
# 那条命令的 shell 自己，把它当场杀掉（表现为整条命令空白返回）。
set -uo pipefail
TAG="${1:-$(date +%H%M)}"

for p in $(pgrep -f "run_tvr_agen[t]"); do kill -9 "$p" 2>/dev/null || true; done
for p in $(pgrep -f "run_tvr\.sh bot[h]"); do kill -9 "$p" 2>/dev/null || true; done
sleep 2
cd /home/sudidaren/tvrbench_addon
if [ -d runs ] && [ -n "$(ls -A runs 2>/dev/null)" ]; then
    mv runs "/root/tvr_runs_${TAG}"
    echo "旧结果 -> /root/tvr_runs_${TAG}"
fi
mkdir -p runs
echo "残留 addon 进程=$(pgrep -cf 'run_tvr_agen[t]' || true)"
for t in $(pgrep -f "thor-Linux6[4]"); do kill -9 "$t" 2>/dev/null || true; done
sleep 1
echo "残留 unity=$(pgrep -cf 'thor-Linux6[4]' || true)"
