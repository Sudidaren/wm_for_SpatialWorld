#!/usr/bin/env bash
# 等 Gemini+WM 的 210 条跑完 → 清场 → 立刻接上 GPT-5 + WM（同一批 250 条 Gemini 失败任务）。
#
# 目的：不让人盯着的空档浪费算力/额度窗口。幂等：有锁就不重复起；
# 目标 run 已经在跑就直接退出。
#
# 用法：setsid nohup bash chain_gpt5_after_gemini.sh >/dev/null 2>&1 </dev/null &
set -uo pipefail

SWE=/home/sudidaren/spatialworld_eval
RUNS="$SWE/runs"
GEMINI_RUN=wm_gemini31pro_fail210
NEXT_RUN=wm_gpt5_ai2thor_fail250
PY=/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python
LOCK=/tmp/lightwm_tmp/chain_gpt5.lock
LOG=/mnt/d/lightwm_out/chain_${NEXT_RUN}.log
export TMPDIR=/tmp/lightwm_tmp

mkdir -p /mnt/d/lightwm_out "$TMPDIR"
exec >>"$LOG" 2>&1

say() { echo "[chain $(date '+%F %T')] $*"; }

# supervisor 的 argv 形如：
#   <venv>/bin/python -u - <model> <url> wm <RUN_NAME> ai2thor 4
# 只用这个特征匹配，避免把监控命令/日志路径误判成"还在跑"。
sup_pat() { printf '%s' "[.]venv/bin/python -u - .* $1 "; }

# 单实例
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    say "已有 chain 在跑（pid $(cat "$LOCK")），退出"
    exit 0
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

# 目标 run 已在跑就不插手
if pgrep -f "$(sup_pat "$NEXT_RUN")" >/dev/null 2>&1; then
    say "$NEXT_RUN 已在运行，退出"
    exit 0
fi

wait_for() {  # $1 = pgrep 模式, $2 = 最长等待秒
    local t=0
    while pgrep -f "$1" >/dev/null 2>&1; do
        sleep 120; t=$((t + 120))
        if [ "$t" -ge "$2" ]; then say "等待超时（$1）"; return 1; fi
    done
    return 0
}

wait_start() {  # $1 = pgrep 模式, $2 = 最长等待秒：等进程真正出现，避免重复拉起
    local t=0
    until pgrep -f "$1" >/dev/null 2>&1; do
        sleep 10; t=$((t + 10))
        if [ "$t" -ge "$2" ]; then say "警告：$1 未在 ${2}s 内出现"; return 1; fi
    done
    return 0
}

say "等待 $GEMINI_RUN 结束（最长 12 小时）"
wait_for "$(sup_pat "$GEMINI_RUN")" 43200 || true
say "$GEMINI_RUN 进程已退出"

# 中途挂了就补跑一次（supervisor 会跳过已完成）
for attempt in 1 2; do
    PENDING=$("$PY" - "$RUNS" "$GEMINI_RUN" <<'PY'
import csv, os, sys
runs, name = sys.argv[1], sys.argv[2]
base = os.path.join(runs, 'replay_legacy311_v1/results.csv')
want = [r['Task ID'] for r in csv.DictReader(open(base, encoding='utf-8-sig'))
        if r['Status'] != 'success']
done = set()
cur = os.path.join(runs, name, 'results.csv')
if os.path.exists(cur):
    done = {r['Task ID'] for r in csv.DictReader(open(cur, encoding='utf-8-sig'))}
print(','.join(t for t in want if t not in done))
PY
)
    [ -z "$PENDING" ] && { say "$GEMINI_RUN 已全部完成"; break; }
    say "第 $attempt 次补跑，剩 $(echo "$PENDING" | tr ',' '\n' | wc -l) 条"
    cd "$SWE"
    # 不要设代理：本机 Clash 会挂，走代理会 SSL EOF；直连网关是通的（2026-09-20 实测）
    TMPDIR_OVERRIDE="$TMPDIR" VLM_FORCE_IPV4=1 VLM_API_TIMEOUT=120 \
    SMOKE_TASKS="$PENDING" RUN_NAME="$GEMINI_RUN" WORKERS=4 \
    LLM_API_KEY="$OPENAI_API_KEY" setsid nohup bash run_wm_gemini_ai2thor.sh \
        >>"/mnt/d/lightwm_out/${GEMINI_RUN}.log" 2>&1 </dev/null &
    wait_start "$(sup_pat "$GEMINI_RUN")" 180 || true
    sleep 30
    wait_for "$(sup_pat "$GEMINI_RUN")" 21600 || true
done

say "清场：reap_sim.sh"
bash /home/sudidaren/lightwm_phases/tools/reap_sim.sh --workers >>"$LOG" 2>&1 || true
pkill -f thor-Linux64 2>/dev/null || true
sleep 10

# 250 条 Gemini 失败任务（含已经跑过的 40 条），GPT-5 臂用同一批
TASKS=$("$PY" - "$RUNS" <<'PY'
import csv, os, sys
base = os.path.join(sys.argv[1], 'replay_legacy311_v1/results.csv')
print(','.join(r['Task ID'] for r in csv.DictReader(open(base, encoding='utf-8-sig'))
               if r['Status'] != 'success'))
PY
)
N=$(echo "$TASKS" | tr ',' '\n' | wc -l)
say "启动 $NEXT_RUN（$N 条）"

cd "$SWE"
# 6 worker：实测 4 worker 时 16 核只有 ~2.2 负载（瓶颈是 VLM API 延迟，不是本地算力），
# 内存 13.9G 里 4 worker 只占 ~5G。Gemini 那一臂不动（跑着的批次不打断），
# 只让新起的 GPT-5 臂吃满一点。
TMPDIR_OVERRIDE="$TMPDIR" VLM_FORCE_IPV4=1 VLM_API_TIMEOUT=120 \
SMOKE_TASKS="$TASKS" RUN_NAME="$NEXT_RUN" WORKERS=6 \
MODEL_NAME=gpt-5 LLM_API_KEY="$OPENAI_API_KEY" \
setsid nohup bash run_wm_gemini_ai2thor.sh \
    >>"/mnt/d/lightwm_out/${NEXT_RUN}.log" 2>&1 </dev/null &

sleep 45
if pgrep -f "$(sup_pat "$NEXT_RUN")" >/dev/null 2>&1; then
    say "$NEXT_RUN 已启动"
else
    say "警告：$NEXT_RUN 没能起来，见 /mnt/d/lightwm_out/${NEXT_RUN}.log"
fi
