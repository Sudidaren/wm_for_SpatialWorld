#!/usr/bin/env bash
# 新卡（westb, 3080Ti 12G / 48 核 / 440G）上跑 ProCTHOR 的 GPT-5+WM 臂。
# 只跑主清单里的那 20 条 procthor（ai2thor 留在本机跑）。
#
# 注意：这份 venv 里的 torch 是 CPU 版，所以感知头走 CPU、不吃显存，
# 瓶颈是 48 核 CPU —— 10 worker 是用户指定的。
set -uo pipefail
SWE=/home/sudidaren/spatialworld_eval
PLANS=/home/sudidaren/lightwm_phases/plans
export DISPLAY=:99
export TMPDIR_OVERRIDE=/tmp/lightwm_tmp
export LIGHTWM_ROOT=/home/sudidaren/lightwm_phases
export PROCTHOR_DATASET_DIR=/root/.prior/datasets/allenai/procthor-10k/439193522244720b86d8c81cde2e51e3a4d150cf
export AI2THOR_SERVER_TIMEOUT=900
export AI2THOR_START_TIMEOUT=900
export MODEL_NAME=gpt-5
export BASE_URL=https://apic1.ohmycdn.com/v1
[ -f /root/.llm_key ] && . /root/.llm_key

TASKS=$(paste -sd, /root/gpt5_wm_procthor_tasks.txt)
N=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c .)
W="${WORKERS:-10}"
echo "[procthor $(date '+%F %T')] 任务=$N worker=$W"
mkdir -p /tmp/lightwm_tmp
cd "$SWE"
MODEL_NAME=gpt-5 BASE_URL="$BASE_URL" PROFILE=wm RUN_NAME="main_gpt5_wm_procthor" \
SCENES=procthor WORKERS="$W" SMOKE_TASKS="$TASKS" \
bash run_wm_gemini_ai2thor.sh
echo "[procthor $(date '+%F %T')] 结束 rc=$?"
