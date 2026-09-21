#!/usr/bin/env bash
# 用本机渲染补跑某一朵云臂里 failed_external 的任务，LLM 仍走那张卡的 vLLM。
#
# 为什么这么做：云卡是 Xvfb + llvmpipe + OpenGL 软件渲染，`reset` 一次要
# ~180s（本机 1.0s）；即便砍掉那次冗余 reset 并放长超时，最大的几栋房子
# （FloorPlan315 等）仍然起不来。本机是 WSLg + Vulkan，同样的场景 15 秒就起来。
# 所以剩下的收尾放本机跑，模型仍然用对应卡上的 vLLM，保证"模型这一侧"不变。
#
# 用法: bash local_refill_from_cloud.sh <云run名> <本机run名> <模型名> <端口> [workers]
set -uo pipefail
CLOUD_RUN="${1:?用法: local_refill_from_cloud.sh <云run> <本机run> <模型> <端口> [workers]}"
LOCAL_RUN="${2:?}"
MODEL="${3:?}"
PORT="${4:?}"
WORKERS="${5:-4}"
SWE=/home/sudidaren/spatialworld_eval
PULL=/mnt/d/lightwm_out/cloud_pull_20260921

# 拉回来的目录名带卡前缀，例如 30b_wm_q30b_438_v2
SRC="$SWE/runs/$CLOUD_RUN/results.csv"
if [ ! -f "$SRC" ]; then
    SRC=$(ls "$PULL"/*_"$CLOUD_RUN"/results.csv 2>/dev/null | head -1)
fi
[ -n "${SRC:-}" ] && [ -f "$SRC" ] || { echo "找不到 $CLOUD_RUN 的 results.csv"; exit 1; }
echo "[local-refill] 用 $SRC 作为待补清单"

TASKS=$(/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python - "$SRC" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
bad = [r["Task ID"] for r in rows if r["Status"] == "failed_external"]
print(",".join(sorted(bad)))
PY
)
N=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c . || true)
echo "[local-refill] $CLOUD_RUN -> $LOCAL_RUN : $N 条, workers=$WORKERS, vLLM=127.0.0.1:$PORT"
[ "$N" -eq 0 ] && { echo "没有 failed_external"; exit 0; }

export DISPLAY=:0 TMPDIR_OVERRIDE=/tmp/lightwm_tmp
cd "$SWE"
MODEL_NAME="$MODEL" BASE_URL="http://127.0.0.1:$PORT/v1" PROFILE=wm \
RUN_NAME="$LOCAL_RUN" SCENES=ai2thor,procthor WORKERS="$WORKERS" LLM_API_KEY=EMPTY \
SMOKE_TASKS="$TASKS" \
setsid nohup bash run_wm_gemini_ai2thor.sh \
    > "/mnt/d/lightwm_out/${LOCAL_RUN}.log" 2>&1 </dev/null &
sleep 30
tail -6 "/mnt/d/lightwm_out/${LOCAL_RUN}.log"
