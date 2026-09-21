#!/usr/bin/env bash
# 干净停掉本机补测链（wm_refill_*），不动看护/隧道/其他评测。
# 单独成脚本：命令行里直接写目标串，pgrep -f 会匹配到自己的 shell。
set -uo pipefail

echo "== 停之前 =="
pgrep -cf "[l]ocal_refill_chain" || true
pgrep -cf "wm_refill_" || true

# 1) 链本身 + 链里的"臂"进程（内联 python，args 里带 arm 名）
for p in $(pgrep -f "local_refill_chai[n]"); do kill -TERM "$p" 2>/dev/null || true; done
for p in $(pgrep -f "wm_refil[l]_"); do kill -TERM "$p" 2>/dev/null || true; done
sleep 3
for p in $(pgrep -f "local_refill_chai[n]"); do kill -9 "$p" 2>/dev/null || true; done
for p in $(pgrep -f "wm_refil[l]_"); do kill -9 "$p" 2>/dev/null || true; done

# 2) 该链的 worker（只按 run 目录名匹配，不碰别的 run）
for q in $(pgrep -f "[w]ork.run_task"); do
    a=$(ps -o args= -p "$q" 2>/dev/null || true)
    case "$a" in *wm_refill_*) kill -TERM "$q" 2>/dev/null || true ;; esac
done
sleep 5
for q in $(pgrep -f "[w]ork.run_task"); do
    a=$(ps -o args= -p "$q" 2>/dev/null || true)
    case "$a" in *wm_refill_*) kill -9 "$q" 2>/dev/null || true ;; esac
done

# 3) 父进程已死的 thor（模拟器），活的评测的 thor 不动
for t in $(pgrep -f "[t]hor-Linux6[4]"); do
    pp=$(ps -o ppid= -p "$t" 2>/dev/null | tr -d ' ')
    a=$(ps -o args= -p "$pp" 2>/dev/null || true)
    case "$a" in *work.run_task*) ;; *) kill -9 "$t" 2>/dev/null || true ;; esac
done

sleep 3
echo "== 停之后 =="
echo "chain=$(pgrep -cf '[l]ocal_refill_chain' || true)  refill_worker=$(pgrep -f 'wm_refill_' | wc -l)  thor=$(pgrep -cf '[t]hor-Linux64' || true)"
echo "看护仍在: guard=$(pgrep -cf '[l]ocal_guard' || true) monitor=$(pgrep -cf '[m]onitor_status' || true) tunnel=$(pgrep -cf '[t]unnel_keeper' || true) gw=$(pgrep -cf '[g]ateway_watchdog' || true)"
