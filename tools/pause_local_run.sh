#!/usr/bin/env bash
# 停掉本机的一个评测 run（supervisor + 它名下的 worker + 模拟器）。
# 用于"网关额度用尽"这种暂停，不删任何结果，续跑时 supervisor 会重排
# 未成功的任务。
#
#   bash tools/pause_local_run.sh wm_gemini31pro_fix_rest156
set -uo pipefail
RUN="${1:?用法: pause_local_run.sh <RUN_NAME>}"

PAT="[.]venv/bin/python -u - .* ${RUN} "
P=$(pgrep -f "$PAT" | head -1 || true)
if [ -n "${P:-}" ]; then
    echo "[pause $(date '+%F %T')] 停 supervisor pid=$P"
    kill -TERM "$P" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$P" 2>/dev/null || break; sleep 2; done
    kill -0 "$P" 2>/dev/null && kill -KILL "$P" 2>/dev/null || true
else
    echo "[pause] 没找到 $RUN 的 supervisor"
fi
sleep 3
# worker：argv 里带这个 run 的输出目录；用 [k] 写法避免匹配到自己
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
for t in $(pgrep -f "thor-Linux6[4]"); do
    a=$(ps -o args= -p "$t" 2>/dev/null || true)
    kill -9 "$t" 2>/dev/null || true
done
sleep 2
echo "[pause] 残留 worker=$(pgrep -cf 'work.run_tas[k]' || true) thor=$(pgrep -cf 'thor-Linux6[4]' || true)"
