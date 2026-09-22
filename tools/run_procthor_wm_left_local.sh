#!/usr/bin/env bash
# 本地把 GPT-5+WM 在 ProcTHOR 上剩下的 13 条跑完（2026-09-22 上午恢复后）。
# 卡上已判 7 条，本地 closed_gpt5_wm_procthor 里的 6 条是欠费时的占位
# （failed_external，completed=None），supervisor 会自动重试。
set -uo pipefail
SWE=/home/sudidaren/spatialworld_eval
PLANS=/home/sudidaren/lightwm_phases/plans
LOG=/mnt/d/lightwm_out/procthor_wm_left.log

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

say() { echo "[wmleft $(date '+%F %T')] $*" | tee -a "$LOG"; }
TASKS=$(paste -sd, "$PLANS/procthor_wm_left.txt")
N=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c .)
W="${WORKERS:-6}"
say "开始：$N 条 procthor WM，$W worker"
cd "$SWE"
MODEL_NAME=gpt-5 BASE_URL=https://apic1.ohmycdn.com/v1 PROFILE=wm \
RUN_NAME=closed_gpt5_wm_procthor SCENES=procthor WORKERS="$W" SMOKE_TASKS="$TASKS" \
bash run_wm_gemini_ai2thor.sh >> /mnt/d/lightwm_out/closed_gpt5_wm_procthor.log 2>&1
say "结束 rc=$?"
