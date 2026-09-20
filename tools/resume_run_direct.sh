#!/usr/bin/env bash
# 不带代理续跑一个 run（本机 Clash 代理会挂，直连网关是通的）。
#
#   bash resume_run_direct.sh <RUN_NAME> <WORKERS> [SMOKE_TASKS]
#
# 步骤：停 chain/watcher → 停 supervisor → reap 模拟器 → 直连续跑。
# supervisor 的 resume 会把 `completed=null`（failed_external）的行重新排队，
# 并且按 task_id 覆盖写，不会产生重复行。
set -uo pipefail

RUN_NAME="${1:?用法: resume_run_direct.sh <RUN_NAME> <WORKERS> [SMOKE_TASKS]}"
WORKERS="${2:-4}"
SMOKE_TASKS="${3:-}"
SWE=/home/sudidaren/spatialworld_eval
TMPDIR_OVERRIDE_PATH=/tmp/lightwm_tmp

log() { echo "[resume $(date '+%F %T')] $*"; }

# 1) 停掉自动交班/看护，避免重启窗口里误交班
LOCK=/tmp/lightwm_tmp/chain_gpt5.lock
if [ -f "$LOCK" ]; then
    kill -TERM "$(cat "$LOCK")" 2>/dev/null || true
    sleep 2
fi
pkill -f "chain_gpt5_after_gemini" 2>/dev/null || true
pkill -f "watch_wm_chain" 2>/dev/null || true
sleep 2
log "已停 chain/watcher"

# 2) 停 supervisor（先 TERM 让它把 state 落盘）
PAT="venv/bin/python -u - .* ${RUN_NAME} "
PID=$(pgrep -f "$PAT" | head -1)
if [ -n "$PID" ]; then
    log "停 supervisor pid=$PID"
    kill -TERM "$PID" 2>/dev/null || true
    for _ in $(seq 1 15); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
    kill -0 "$PID" 2>/dev/null && { log "TERM 无效，KILL"; kill -KILL "$PID" 2>/dev/null || true; }
fi
sleep 3
# 3) 收掉残留的 worker / 模拟器
pkill -TERM -f "run_task.py --config" 2>/dev/null || true
sleep 5
bash /home/sudidaren/lightwm_phases/tools/reap_sim.sh --workers >>/tmp/lightwm_tmp/reap.log 2>&1 || true
pkill -f thor-Linux64 2>/dev/null || true
sleep 5
log "残留 thor=$(pgrep -cf thor-Linux64 || true) run_task=$(pgrep -cf 'run_task.py --config' || true)"

# 4) 直连续跑（**不设** HTTPS_PROXY/HTTP_PROXY）
if [ -z "$SMOKE_TASKS" ]; then
    SMOKE_TASKS=$(python3 - "$RUN_NAME" <<'PY'
import csv, json, os, sys
run = sys.argv[1]
runs = '/home/sudidaren/spatialworld_eval/runs'

# 本 run 只负责 250 条失败里"没被更早批次跑掉"的那些：逐题看各 run 目录，
# 已经被别的 run（例如 _selection_wm40 里那 40 条）成功/失败判定过的就不再重复跑。
EXCLUDE_FROM = ['wm_gemini31pro_simple40']
handled = set()
for other in EXCLUDE_FROM:
    f = os.path.join(runs, other, 'results.csv')
    if os.path.exists(f):
        for r in csv.DictReader(open(f, encoding='utf-8-sig')):
            if r['Status'] != 'failed_external':
                handled.add(r['Task ID'])

want = [r['Task ID'] for r in csv.DictReader(
    open(os.path.join(runs, 'replay_legacy311_v1/results.csv'), encoding='utf-8-sig'))
    if r['Status'] != 'success' and r['Task ID'] not in handled]
cur = os.path.join(runs, run, 'results.csv')
done = set()
if os.path.exists(cur):
    for r in csv.DictReader(open(cur, encoding='utf-8-sig')):
        if r['Status'] != 'failed_external':      # external 的要重跑
            done.add(r['Task ID'])
print(','.join(t for t in want if t not in done))
PY
)
fi
N=$(echo "$SMOKE_TASKS" | tr ',' '\n' | grep -c . || true)
log "续跑 $RUN_NAME：$N 条，workers=$WORKERS，直连"

cd "$SWE"
TMPDIR_OVERRIDE="$TMPDIR_OVERRIDE_PATH" VLM_FORCE_IPV4=1 VLM_API_TIMEOUT=120 \
SMOKE_TASKS="$SMOKE_TASKS" RUN_NAME="$RUN_NAME" WORKERS="$WORKERS" \
MODEL_NAME="${MODEL_NAME:-gemini-3.1-pro-preview}" \
LLM_API_KEY="${LLM_API_KEY:-$OPENAI_API_KEY}" \
setsid nohup bash run_wm_gemini_ai2thor.sh \
    >>"/mnt/d/lightwm_out/${RUN_NAME}.log" 2>&1 </dev/null &
sleep 30
log "已提交，supervisor=$(pgrep -cf "$PAT" || true)"
