#!/usr/bin/env bash
# 卡上跑 GPT-5 + WM 的一批（PROFILE=wm）。
# 与基线臂只差 profile：同一批任务、同一个模型，用来做配对。
# WM 的感知头要吃显存（约 0.9GB/worker），所以 worker 数由 WORKERS 决定，
# 起之前先看 nvidia-smi 的空闲显存（vLLM 占着大头）。
set -uo pipefail
S="${1:?用法: card_run_gpt5_wm.sh <shard>}"
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
echo "[gpt5wm $(date '+%F %T')] shard=$S 任务=$N workers=${WORKERS:-4}"
cd "$SWE"
MODEL_NAME=gpt-5 \
BASE_URL=https://apic1.ohmycdn.com/v1 \
PROFILE=wm \
RUN_NAME="main_gpt5_wm_s${S}" \
SCENES=ai2thor,procthor \
WORKERS="${WORKERS:-4}" \
SMOKE_TASKS="$TASKS" \
bash run_wm_gemini_ai2thor.sh
echo "[gpt5wm $(date '+%F %T')] shard=$S 结束 rc=$?"
