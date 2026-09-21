#!/usr/bin/env bash
# 在卡上重跑本 run 里 failed_external 的那些 ai2thor 任务（环境没起来的）。
#
# 背景：2026-09-21 那批 6 worker 并发时，37 条 ai2thor 报
# "Reading from AI2-THOR backend timed out (using 300.0s)"。但把其中几个场景
# （FloorPlan313/314/311/303/30）单独在卡上加载，只要 9~26 秒 —— 说明当时
# 是并发抢资源（6 个 Unity + 6 份 WM 感知 + vLLM）把场景初始化拖爆了，
# 不是场景本身跑不了。所以这里**降并发 + 放长超时**原地重跑：
#   WORKERS 默认 3（原来 6）
#   AI2THOR_SERVER_TIMEOUT / AI2THOR_START_TIMEOUT 都放到 900s
# 任务清单直接从本 run 的 results.csv 里取 failed_external，逐卡各取各的。
#
# 用法: bash refill_ai2thor.sh <RUN_NAME> <MODEL_NAME> [WORKERS]
set -uo pipefail
RUN="${1:?用法: refill_ai2thor.sh <RUN_NAME> <MODEL_NAME> [WORKERS]}"
MODEL="${2:?}"
WORKERS="${3:-3}"
SWE=/home/sudidaren/spatialworld_eval
LOGD=/root/autodl-tmp

export DISPLAY=:99 TMPDIR_OVERRIDE=/tmp/lightwm_tmp
export AI2THOR_SERVER_TIMEOUT=900 AI2THOR_START_TIMEOUT=900
# 注意：render_depth / render_instance_segmentation 本来就由 wm_config_patch
# 在运行时置成 False（WM 只吃 RGB），不需要在这里再管。
#
# 默认 600s 的 stall 看门狗会在场景还没 init 完时就把 worker SIGKILL 掉
# （表现为 "no result json produced"），而 run_task 的 init 带了 scene 时会
# 第二次 env.reset() 重新加载场景，软件渲染下很慢 —— 所以必须一起放长。
export WM_STALL_TIMEOUT=1800 WM_TASK_TIMEOUT=3600

TASKS=$(/root/miniconda3/bin/python - "$SWE/runs/$RUN/results.csv" <<'PY'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
except Exception:
    rows = []
bad = [r["Task ID"] for r in rows
       if r["Task ID"].startswith("ai2thor") and r["Status"] == "failed_external"]
print(",".join(bad))
PY
)
N=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c . || true)
echo "[refill $(date '+%F %T')] $RUN 待补 $N 条，workers=$WORKERS，timeout=900s"
[ "$N" -eq 0 ] && { echo "没有 failed_external，退出"; exit 0; }

cd "$SWE"
MODEL_NAME="$MODEL" BASE_URL=http://127.0.0.1:8000/v1 PROFILE=wm \
RUN_NAME="$RUN" SCENES=ai2thor WORKERS="$WORKERS" LLM_API_KEY=EMPTY \
SMOKE_TASKS="$TASKS" \
setsid nohup bash run_wm_gemini_ai2thor.sh \
    >>"$LOGD/${RUN}_refill.log" 2>&1 </dev/null &
sleep 25
tail -5 "$LOGD/${RUN}_refill.log"
