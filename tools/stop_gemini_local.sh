#!/usr/bin/env bash
# 停掉本地 Gemini 批次（smoke_gemini_wm / closed_gemini_wm），不动看护与隧道。
set -uo pipefail
for pat in "run_gemini_loca[l]" "smoke_gemini_w[m]" "closed_gemini_w[m]"; do
    for p in $(pgrep -f "$pat"); do kill -TERM "$p" 2>/dev/null || true; done
done
sleep 4
for pat in "run_gemini_loca[l]" "smoke_gemini_w[m]" "closed_gemini_w[m]"; do
    for p in $(pgrep -f "$pat"); do kill -9 "$p" 2>/dev/null || true; done
done
# 父进程已死的 thor
for t in $(pgrep -f "[t]hor-Linux6[4]"); do
    pp=$(ps -o ppid= -p "$t" 2>/dev/null | tr -d ' ')
    a=$(ps -o args= -p "$pp" 2>/dev/null || true)
    case "$a" in *work.run_task*) ;; *) kill -9 "$t" 2>/dev/null || true ;; esac
done
sleep 2
echo "chain=$(pgrep -cf '[r]un_gemini_local' || true) worker=$(ps -eo cmd | grep -c '[w]ork.run_task') thor=$(ps -eo cmd | grep -c '[t]hor-Linux64')"
