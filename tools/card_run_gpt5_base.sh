#!/usr/bin/env bash
# 在卡上跑 GPT-5 纯基线的一批（PROFILE=llm，不加载 WM 感知头，不需要 GPU 推理）。
# 用法：bash card_run_gpt5_base.sh <shard编号>
set -uo pipefail
S="${1:?用法: card_run_gpt5_base.sh <shard>}"
SWE=/home/sudidaren/spatialworld_eval
export DISPLAY=:99
export TMPDIR_OVERRIDE=/tmp/lightwm_tmp
export LIGHTWM_ROOT=/home/sudidaren/lightwm_phases
export AI2THOR_SERVER_TIMEOUT=900
export AI2THOR_START_TIMEOUT=900
export PROCTHOR_DATASET_DIR=/root/.prior/datasets/allenai/procthor-10k/439193522244720b86d8c81cde2e51e3a4d150cf
[ -f /root/.llm_key ] && . /root/.llm_key
TASKS=$(paste -sd, "/root/gpt5_base_tasks_s${S}.txt")
N=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c .)
echo "[gpt5 $(date '+%F %T')] shard=$S 任务=$N"
cd "$SWE"
MODEL_NAME=gpt-5 \
BASE_URL=https://apic1.ohmycdn.com/v1 \
PROFILE=llm \
RUN_NAME="main_gpt5_base_s${S}" \
SCENES=ai2thor,procthor \
WORKERS="${WORKERS:-6}" \
SMOKE_TASKS="$TASKS" \
bash run_wm_gemini_ai2thor.sh
echo "[gpt5 $(date '+%F %T')] shard=$S 结束 rc=$?"
