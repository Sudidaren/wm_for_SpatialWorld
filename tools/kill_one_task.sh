#!/usr/bin/env bash
# 叫停单个任务：supervisor 会重试（max_tries = 1 + retries = 3），
# 所以要把它的重试也一并杀干净，线程才会移去挑下一个任务。
# 用法：bash kill_one_task.sh ai2thor05004
set -uo pipefail
T="${1:?用法: kill_one_task.sh <task_id>}"
# supervisor 的 max_tries = 1 + retries(=2) = 3，所以最多会有 3 次尝试。
# 每杀掉一次它会立刻起下一次；这里持续盯 100 秒，见一次杀一次，
# 三次耗尽后 supervisor 会把它记为 failed_external 并让这个线程去做下一题。
deadline=$(( $(date +%s) + 100 ))
round=0
while [ "$(date +%s)" -lt "$deadline" ]; do
    pids=$(ps -eo pid,args | awk -v t="$T" '$0 ~ /work\.run_task/ && $0 ~ ("--tasks " t " ") {print $1}')
    if [ -n "$pids" ]; then
        round=$((round+1))
        echo "第 $round 次：杀掉 $pids"
        for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
        for t in $(pgrep -f "[t]hor-Linux6[4]"); do
            pp=$(ps -o ppid= -p "$t" 2>/dev/null | tr -d ' ')
            a=$(ps -o args= -p "$pp" 2>/dev/null || true)
            case "$a" in *work.run_task*) ;; *) kill -9 "$t" 2>/dev/null || true ;; esac
        done
    fi
    sleep 3
done
sleep 3
echo "== 复查 =="
echo "还有 ${T} 的进程: $(ps -eo args | grep -c "[w]ork.run_task.*${T}")"
echo "在跑的 worker 总数: $(ps -eo cmd | grep -c '[w]ork.run_task')"
