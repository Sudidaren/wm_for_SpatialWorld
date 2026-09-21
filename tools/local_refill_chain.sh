#!/usr/bin/env bash
# 本机跑"补测"：ai2thor 那批环境失败的（云上起不来）+ procthor 的 30 条同分布采样。
# LLM 走对应卡上的 vLLM（隧道 18000/18001/18002），渲染在本机。
#
# 三臂串行（本机 16 核，不能并发）：
#   30b  -> 18000   qwen3vl-30b
#   8b   -> 18001   qwen3vl-8b
#   kimi -> 18002   kimivl-a3b
set -uo pipefail
SWE=/home/sudidaren/spatialworld_eval
TASKS=$(paste -sd, /home/sudidaren/tvrbench_addon/tasks/local_refill_now.txt)
N=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c .)
LOG=/mnt/d/lightwm_out/local_refill_chain.log
export DISPLAY=:0 TMPDIR_OVERRIDE=/tmp/lightwm_tmp LIGHTWM_ROOT=/home/sudidaren/lightwm_phases
export PROCTHOR_DATASET_DIR=/home/sudidaren/.prior/datasets/allenai/procthor-10k/439193522244720b86d8c81cde2e51e3a4d150cf

say() { echo "[refill $(date '+%F %T')] $*" >> "$LOG"; }
say "开始：每臂 $N 条（ai2thor 补测 + procthor30）"

run_arm() {   # $1=tag $2=model $3=port
    say "=== 臂 $1 ($2 @ $3) ==="
    cd "$SWE"
    MODEL_NAME="$2" BASE_URL="http://127.0.0.1:$3/v1" PROFILE=wm \
    RUN_NAME="wm_refill_$1" SCENES=ai2thor,procthor WORKERS=5 LLM_API_KEY=EMPTY \
    SMOKE_TASKS="$TASKS" \
    bash run_wm_gemini_ai2thor.sh >> "/mnt/d/lightwm_out/refill_$1.log" 2>&1
    say "=== 臂 $1 完成 ==="
}

run_arm 30b  qwen3vl-30b 18000
run_arm 8b   qwen3vl-8b  18001
run_arm kimi kimivl-a3b  18002
say "全部完成"
