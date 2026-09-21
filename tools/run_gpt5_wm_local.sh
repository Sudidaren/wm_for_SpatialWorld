#!/usr/bin/env bash
# 本地跑 GPT-5 + WM 的剩余任务（2026-09-21 22:1x：三张卡被莫名关停，
# 已拉回 68 条，剩 72 条搬到本机接着跑）。
#
# 与卡上那批的配置逐字段一致，只换了跑的位置；已完成的 68 条不重跑
# （合并出表时两边拼起来）。
#
# 用法：bash run_gpt5_wm_local.sh [worker 数，默认 6]
set -uo pipefail
MODE="${1:-run}"
SWE=/home/sudidaren/spatialworld_eval
PLANS=/home/sudidaren/lightwm_phases/plans
LOG=/mnt/d/lightwm_out/gpt5_wm_local.log
PY=/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python

export DISPLAY=:0
export TMPDIR_OVERRIDE=/tmp/lightwm_tmp
export LIGHTWM_ROOT=/home/sudidaren/lightwm_phases
export PROCTHOR_DATASET_DIR=/home/sudidaren/.prior/datasets/allenai/procthor-10k/439193522244720b86d8c81cde2e51e3a4d150cf
export AI2THOR_SERVER_TIMEOUT=300
export AI2THOR_START_TIMEOUT=300
# 网关直连可用，绕开已经死过的 Windows 代理（19:0x 那次就是卡在这）
export no_proxy="${no_proxy:+$no_proxy,}apic1.ohmycdn.com"
export NO_PROXY="$no_proxy"
export MODEL_NAME=gpt-5
export BASE_URL="${OPENAI_BASE_URL:-https://apic1.ohmycdn.com/v1}"
if [ -z "${LLM_API_KEY:-}" ]; then
    LLM_API_KEY=$(grep -m1 '^export OPENAI_API_KEY=' "$HOME/.bashrc" | cut -d'"' -f2)
    export LLM_API_KEY
fi
[ -n "$LLM_API_KEY" ] || { echo "拿不到 LLM_API_KEY"; exit 2; }

say() { echo "[g5wm $(date '+%F %T')] $*" | tee -a "$LOG"; }

TASKS=$(tr '\n' ',' < "$PLANS/gpt5_wm_todo.txt" | sed 's/,$//')
N=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c .)
W="${WORKERS:-6}"

say "开始：$N 条，$W worker，model=$MODEL_NAME"
cd "$SWE"
MODEL_NAME=gpt-5 BASE_URL="$BASE_URL" PROFILE=wm RUN_NAME=closed_gpt5_wm \
SCENES=ai2thor,procthor WORKERS="$W" SMOKE_TASKS="$TASKS" \
bash run_wm_gemini_ai2thor.sh >> /mnt/d/lightwm_out/closed_gpt5_wm.log 2>&1
rc=$?
say "结束 rc=$rc"
