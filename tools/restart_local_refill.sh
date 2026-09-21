#!/usr/bin/env bash
# 重启本机补测链（用最新的任务清单）。
# 单独成脚本：直接在 bash -c 里写脚本名，pgrep -f 会匹配到执行命令的 shell 自己。
set -uo pipefail
T=/home/sudidaren/lightwm_phases/tools/local_refill_chain.sh
sed -i 's#/tasks/refill_both.txt#/tasks/local_refill_now.txt#' "$T"
for p in $(pgrep -f "local_refill_chai[n]"); do kill -9 "$p" 2>/dev/null || true; done
for q in $(pgrep -f "work.run_tas[k]"); do
    a=$(ps -o args= -p "$q" 2>/dev/null || true)
    case "$a" in *wm_refill_*) kill -9 "$q" 2>/dev/null || true ;; esac
done
sleep 2
for t in $(pgrep -f "thor-Linux6[4]"); do
    pp=$(ps -o ppid= -p "$t" 2>/dev/null | tr -d ' ')
    a=$(ps -o args= -p "$pp" 2>/dev/null || true)
    case "$a" in *work.run_task*) ;; *) kill -9 "$t" 2>/dev/null || true ;; esac
done
grep -n "local_refill_now" "$T" | head -1
setsid nohup bash "$T" >> /mnt/d/lightwm_out/local_refill_chain.log 2>&1 </dev/null &
sleep 30
echo "chain=$(pgrep -cf 'local_refill_chai[n]' || true)  worker=$(pgrep -cf 'work.run_tas[k]' || true)"
tail -3 /mnt/d/lightwm_out/local_refill_chain.log
