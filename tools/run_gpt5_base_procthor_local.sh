#!/usr/bin/env bash
# 本地跑 GPT-5 的 ProcTHOR 基线（20 条）。
#
# 为什么要在本地重跑：旧三张卡上跑过一次，20 条全是 failed_external/api_error
# ——踩的就是 top_p 那个 bug（GPT-5 走中转不接受 top_p）。本地已经打了
# provider 层补丁（mllm_base_agent/llm/provider.py），这次正好验证它生效。
#
# 与 WM 臂同一批任务、同一个模型，只差 PROFILE=llm，用来做配对。
set -uo pipefail
SWE=/home/sudidaren/spatialworld_eval
PLANS=/home/sudidaren/lightwm_phases/plans
LOG=/mnt/d/lightwm_out/gpt5_base_procthor.log

export DISPLAY=:0
export TMPDIR_OVERRIDE=/tmp/lightwm_tmp
export LIGHTWM_ROOT=/home/sudidaren/lightwm_phases
export PROCTHOR_DATASET_DIR=/home/sudidaren/.prior/datasets/allenai/procthor-10k/439193522244720b86d8c81cde2e51e3a4d150cf
export AI2THOR_SERVER_TIMEOUT=600
export AI2THOR_START_TIMEOUT=600
export no_proxy="${no_proxy:+$no_proxy,}apic1.ohmycdn.com"
export NO_PROXY="$no_proxy"
if [ -z "${LLM_API_KEY:-}" ]; then
    LLM_API_KEY=$(grep -m1 '^export OPENAI_API_KEY=' "$HOME/.bashrc" | cut -d'"' -f2)
    export LLM_API_KEY
fi
[ -n "$LLM_API_KEY" ] || { echo "拿不到 LLM_API_KEY"; exit 2; }

TASKS=$(paste -sd, "$PLANS/procthor_main20.txt")
N=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c .)
W="${WORKERS:-6}"
echo "[base $(date '+%F %T')] $N 条 procthor 基线，model=gpt-5，worker=$W" | tee -a "$LOG"
mkdir -p /tmp/lightwm_tmp
cd "$SWE"
MODEL_NAME=gpt-5 BASE_URL=https://apic1.ohmycdn.com/v1 PROFILE=llm \
RUN_NAME=closed_gpt5_base_procthor SCENES=procthor WORKERS="$W" \
SMOKE_TASKS="$TASKS" \
bash run_wm_gemini_ai2thor.sh >> /mnt/d/lightwm_out/closed_gpt5_base_procthor.log 2>&1
echo "[base $(date '+%F %T')] 结束 rc=$?" | tee -a "$LOG"
