#!/usr/bin/env bash
# 在卡上补跑 procthor 半场（127 条），不动已经跑完的 ai2thor 结果。
#
# 关键点：supervisor 的 state.json 是"全量 438 条"，用 SCENES=procthor 只把
# procthor 放进本次计划，write_results_csv 仍然整份重写 —— 于是 ai2thor 的
# 22 成功 / 252 失败原样保留，procthor 那条从 pending 变成判定。
#
# 用法: bash run_procthor.sh <RUN_NAME> <MODEL_NAME> <WORKERS>
set -uo pipefail
RUN="${1:?用法: run_procthor.sh <RUN_NAME> <MODEL_NAME> <WORKERS>}"
MODEL="${2:?}"
WORKERS="${3:?}"
LOGD=/root/autodl-tmp
SWE=/home/sudidaren/spatialworld_eval

export DISPLAY=:99 TMPDIR_OVERRIDE=/tmp/lightwm_tmp AI2THOR_SERVER_TIMEOUT=300
cd "$SWE"
echo "[procthor $(date '+%F %T')] $RUN model=$MODEL workers=$WORKERS"
MODEL_NAME="$MODEL" BASE_URL=http://127.0.0.1:8000/v1 PROFILE=wm \
RUN_NAME="$RUN" SCENES=procthor WORKERS="$WORKERS" LLM_API_KEY=EMPTY \
setsid nohup bash run_wm_gemini_ai2thor.sh \
    >>"$LOGD/${RUN}_procthor.log" 2>&1 </dev/null &
sleep 25
tail -6 "$LOGD/${RUN}_procthor.log"
