#!/usr/bin/env bash
# 在卡上补跑本 run 里所有 failed_external 的任务（ai2thor + procthor 一起）。
#
# 与 newcard_refill_ai2thor.sh 的区别：那个只管 ai2thor。procthor 的补偿
# 也需要跑，而一个 run 目录同时只能有一个 supervisor（两个会抢写
# results.csv / state.json），所以合并成一次。
#
# 用法: bash refill_failed.sh <RUN_NAME> <MODEL_NAME> [WORKERS]
set -uo pipefail
RUN="${1:?用法: refill_failed.sh <RUN_NAME> <MODEL_NAME> [WORKERS]}"
MODEL="${2:?}"
WORKERS="${3:-3}"
SWE=/home/sudidaren/spatialworld_eval
LOGD=/root/autodl-tmp

export DISPLAY=:99 TMPDIR_OVERRIDE=/tmp/lightwm_tmp
# 云卡软件渲染很慢：一次 reset 实测 ~180s，加上 run_task 里 "init 带 scene"
# 那次（已修成只在场景不同时才做），超时必须给足；默认 600s 的 stall 看门狗
# 会在场景还没 init 完时就把 worker SIGKILL 掉。
export AI2THOR_SERVER_TIMEOUT=900 AI2THOR_START_TIMEOUT=900
export WM_STALL_TIMEOUT=1800 WM_TASK_TIMEOUT=3600

TASKS=$(/root/miniconda3/bin/python - "$SWE/runs/$RUN/results.csv" <<'PY'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
except Exception:
    rows = []
bad = [r["Task ID"] for r in rows if r["Status"] == "failed_external"]
print(",".join(bad))
PY
)
N=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c . || true)
echo "[refill $(date '+%F %T')] $RUN 待补 $N 条（ai2thor+procthor），workers=$WORKERS"
[ "$N" -eq 0 ] && { echo "没有 failed_external，退出"; exit 0; }

cd "$SWE"
MODEL_NAME="$MODEL" BASE_URL=http://127.0.0.1:8000/v1 PROFILE=wm \
RUN_NAME="$RUN" SCENES=ai2thor,procthor WORKERS="$WORKERS" LLM_API_KEY=EMPTY \
SMOKE_TASKS="$TASKS" \
setsid nohup bash run_wm_gemini_ai2thor.sh \
    >>"$LOGD/${RUN}_refill.log" 2>&1 </dev/null &
sleep 25
tail -5 "$LOGD/${RUN}_refill.log"
