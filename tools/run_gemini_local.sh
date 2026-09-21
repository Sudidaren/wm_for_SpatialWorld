#!/usr/bin/env bash
# 本地只跑 Gemini 这条（2026-09-21 切分：ai2thor 120 + ProCTHOR 20）。
#
# 只需要跑 WM 臂：
#   * Gemini 基线 ai2thor 311 已有（replay_legacy311_v1，重放重判，$0）
#   * Gemini 基线 procthor 127 已有（gemini31pro_procthor127_frozen_v1）
#   * WM 臂在 ai2thor 主清单里已有 69 条现值版结果 → 只补 51 条
#   * WM 臂 procthor 一条都没有 → 全 20 条
# 合计 71 条。
#
# 用法：bash run_gemini_local.sh smoke   # 冒烟 2 条（1 ai2thor + 1 procthor）
#       bash run_gemini_local.sh full    # 正式 71 条
#       bash run_gemini_local.sh chain   # 冒烟 → 过闸门 → 正式
set -uo pipefail
MODE="${1:-full}"
SWE=/home/sudidaren/spatialworld_eval
PLANS=/home/sudidaren/lightwm_phases/plans
LOG=/mnt/d/lightwm_out/gemini_local.log
PY=/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python

export DISPLAY=:0
export TMPDIR_OVERRIDE=/tmp/lightwm_tmp
export LIGHTWM_ROOT=/home/sudidaren/lightwm_phases
export PROCTHOR_DATASET_DIR=/home/sudidaren/.prior/datasets/allenai/procthor-10k/439193522244720b86d8c81cde2e51e3a4d150cf
export AI2THOR_SERVER_TIMEOUT=300
export AI2THOR_START_TIMEOUT=300
# 2026-09-21 19:0x：Windows 上的 Clash(127.0.0.1:7897) 挂了，worker 全卡在
# SYN-SENT 上（负载 0.09、7 分钟零进展）。实测**直连网关可用**（chat 200/3.2s），
# 所以把网关域名从代理里摘出来，绕开那个已死的代理。
export no_proxy="${no_proxy:+$no_proxy,}apic1.ohmycdn.com"
export NO_PROXY="$no_proxy"
export MODEL_NAME="${MODEL_NAME:-gemini-3.1-pro-preview}"
export BASE_URL="${OPENAI_BASE_URL:-https://apic1.ohmycdn.com/v1}"
export LLM_API_KEY="${LLM_API_KEY:-${OPENAI_API_KEY:-}}"
# 非交互 shell 读不到 ~/.bashrc 里的 key，这里补一次（不打印明文）
if [ -z "$LLM_API_KEY" ]; then
    LLM_API_KEY=$(grep -m1 '^export OPENAI_API_KEY=' "$HOME/.bashrc" | cut -d'"' -f2)
    export LLM_API_KEY
fi
[ -n "$LLM_API_KEY" ] || { echo "拿不到 LLM_API_KEY"; exit 2; }

say() { echo "[gem $(date '+%F %T')] $*" | tee -a "$LOG"; }

TASKS=$(tr '\n' ',' < "$PLANS/gemini_local_tasks.txt" | sed 's/,$//')
N=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c .)
SMOKE="$(head -1 "$PLANS/ai2thor_main120.txt"),$(head -1 "$PLANS/procthor_main20.txt")"

run_arm() {   # $1=run 名 $2=任务串
    local run="$1" tasks="$2"
    local n; n=$(printf '%s' "$tasks" | tr ',' '\n' | grep -c .)
    say "=== 跑到 $run：$n 条 ==="
    cd "$SWE"
    MODEL_NAME="$MODEL_NAME" BASE_URL="$BASE_URL" PROFILE=wm RUN_NAME="$run" \
    SCENES=ai2thor,procthor WORKERS="${WORKERS:-5}" SMOKE_TASKS="$tasks" \
    bash run_wm_gemini_ai2thor.sh >> "/mnt/d/lightwm_out/${run}.log" 2>&1
    local rc=$?
    say "=== $run 结束 rc=$rc ==="
    return $rc
}

rows() {   # 读某个 run 的判定行数
    "$PY" -c "
import csv
try:
    print(len(list(csv.DictReader(open('$SWE/runs/$1/results.csv',encoding='utf-8-sig')))))
except Exception:
    print(0)"
}

if [ "$MODE" = "smoke" ]; then
    say "冒烟：$SMOKE"
    run_arm smoke_gemini_wm "$SMOKE"
    say "冒烟判定=$(rows smoke_gemini_wm)"
    exit 0
fi

if [ "$MODE" = "chain" ]; then
    say "冒烟：$SMOKE"
    run_arm smoke_gemini_wm "$SMOKE"
    got=$(rows smoke_gemini_wm)
    say "冒烟判定=$got"
    if [ "${got:-0}" -lt 2 ]; then say "🚨 冒烟没过，不接正式批次"; exit 1; fi
    say "✅ 冒烟通过，接正式批次"
fi

say "开始正式批次：$N 条（ai2thor 51 + procthor 20）model=$MODEL_NAME"
run_arm closed_gemini_wm "$TASKS"
say "结束：判定=$(rows closed_gemini_wm)/$N"
