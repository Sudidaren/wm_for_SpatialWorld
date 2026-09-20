#!/usr/bin/env bash
# 停掉卡上的一个评测 run（supervisor + worker + 模拟器），不删结果。
#   bash /root/stop_run.sh wm_gpt5_ai2thor_fail250
set -uo pipefail
RUN="${1:?用法: stop_run.sh <RUN_NAME>}"
PAT="[.]venv/bin/python -u - .* ${RUN} "
P=$(pgrep -f "$PAT" | head -1 || true)
if [ -n "${P:-}" ]; then
    echo "[stop $(date '+%F %T')] 停 supervisor pid=$P"
    kill -TERM "$P" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$P" 2>/dev/null || break; sleep 2; done
    kill -0 "$P" 2>/dev/null && kill -KILL "$P" 2>/dev/null || true
else
    echo "[stop] 没找到 $RUN 的 supervisor"
fi
sleep 3
for q in $(pgrep -f "scripts.ai2thor.work.run_tas[k]"); do
    a=$(ps -o args= -p "$q" 2>/dev/null || true)
    case "$a" in *"$RUN"*) kill -TERM "$q" 2>/dev/null || true ;; esac
done
sleep 6
for q in $(pgrep -f "scripts.ai2thor.work.run_tas[k]"); do
    a=$(ps -o args= -p "$q" 2>/dev/null || true)
    case "$a" in *"$RUN"*) kill -9 "$q" 2>/dev/null || true ;; esac
done
sleep 2
for t in $(pgrep -f "thor-Linux6[4]"); do kill -9 "$t" 2>/dev/null || true; done
sleep 2
echo "[stop] 残留 worker=$(pgrep -cf 'work.run_tas[k]' || true) thor=$(pgrep -cf 'thor-Linux6[4]' || true)"
