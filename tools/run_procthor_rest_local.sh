#!/usr/bin/env bash
# 本地补齐 ProcTHOR 的剩余：GPT-5 基线 20 条 + GPT-5+WM 剩下的 13 条。
# 用新 key（2026-09-22 01:3x 换的，$30）。
# 串行跑（procthor 场景重，本机只有 13.9G 内存，两臂不能并发）。
set -uo pipefail
SWE=/home/sudidaren/spatialworld_eval
PLANS=/home/sudidaren/lightwm_phases/plans
LOG=/mnt/d/lightwm_out/procthor_rest.log

export DISPLAY=:0
export TMPDIR_OVERRIDE=/tmp/lightwm_tmp
export LIGHTWM_ROOT=/home/sudidaren/lightwm_phases
export PROCTHOR_DATASET_DIR=/home/sudidaren/.prior/datasets/allenai/procthor-10k/439193522244720b86d8c81cde2e51e3a4d150cf
export AI2THOR_SERVER_TIMEOUT=600
export AI2THOR_START_TIMEOUT=600
export no_proxy="${no_proxy:+$no_proxy,}apic1.ohmycdn.com"
export NO_PROXY="$no_proxy"
export LLM_API_KEY=$(grep -m1 '^export OPENAI_API_KEY=' "$HOME/.bashrc" | cut -d'"' -f2)
[ -n "$LLM_API_KEY" ] || { echo "拿不到 key"; exit 2; }
mkdir -p /tmp/lightwm_tmp

say() { echo "[rest $(date '+%F %T')] $*" | tee -a "$LOG"; }

run_arm() {   # $1=tag $2=profile $3=任务文件
    local tag="$1" prof="$2" list="$3" W="${WORKERS:-6}"
    local tasks; tasks=$(paste -sd, "$list")
    local n; n=$(printf '%s' "$tasks" | tr ',' '\n' | grep -c .)
    say "=== $tag（profile=$prof）$n 条，$W worker ==="
    cd "$SWE"
    MODEL_NAME=gpt-5 BASE_URL=https://apic1.ohmycdn.com/v1 PROFILE="$prof" \
    RUN_NAME="$tag" SCENES=procthor WORKERS="$W" SMOKE_TASKS="$tasks" \
    bash run_wm_gemini_ai2thor.sh >> "/mnt/d/lightwm_out/${tag}.log" 2>&1
    say "=== $tag 结束 rc=$? ==="
}

run_arm closed_gpt5_base_procthor llm "$PLANS/procthor_main20.txt"
run_arm closed_gpt5_wm_procthor wm "$PLANS/procthor_wm_left.txt"
say "全部结束"
