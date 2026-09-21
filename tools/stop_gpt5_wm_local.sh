#!/usr/bin/env bash
# 停掉本机 GPT-5+WM 批次（closed_gpt5_wm）。独立脚本，避免 pgrep 自匹配。
set -uo pipefail
for pat in "run_gpt5_wm_loca[l]" "closed_gpt5_w[m]"; do
    for p in $(pgrep -f "$pat"); do kill -TERM "$p" 2>/dev/null || true; done
done
sleep 4
for pat in "run_gpt5_wm_loca[l]" "closed_gpt5_w[m]"; do
    for p in $(pgrep -f "$pat"); do kill -9 "$p" 2>/dev/null || true; done
done
for t in $(pgrep -f "[t]hor-Linux6[4]"); do
    pp=$(ps -o ppid= -p "$t" 2>/dev/null | tr -d ' ')
    a=$(ps -o args= -p "$pp" 2>/dev/null || true)
    case "$a" in *work.run_task*) ;; *) kill -9 "$t" 2>/dev/null || true ;; esac
done
sleep 3
echo "chain=$(pgrep -cf 'run_gpt5_wm_loca[l]' || true) worker=$(ps -eo cmd | grep -c '[w]ork.run_task') thor=$(ps -eo cmd | grep -c '[t]hor-Linux64')"
