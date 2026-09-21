#!/usr/bin/env bash
# 在卡上并行跑 TVR 任务集（addon 的 run_tvr_agent.py 本身是串行的，所以这里
# 把任务列表切成 N 片、起 N 个进程）。
#
# 用法: bash run_tvr.sh <base|wm|both> <模型名> <片数> [all|stage1|stages1-2|文件]
#   both = 先把 base 全跑完，再自动接着跑 wm（同一个卡不并发两臂）
set -uo pipefail
ARM="${1:?用法: run_tvr.sh <base|wm> <模型> <片数> [all|stage1|文件]}"
MODEL="${2:?}"
SHARDS="${3:-4}"
SPEC="${4:-all}"
ADDON=/home/sudidaren/tvrbench_addon
export DISPLAY=:99 TMPDIR=/tmp/lightwm_tmp LIGHTWM_ROOT=/home/sudidaren/lightwm_phases
export PROCTHOR_DATASET_DIR=/root/.prior/datasets/allenai/procthor-10k/439193522244720b86d8c81cde2e51e3a4d150cf
export AI2THOR_SERVER_TIMEOUT=900 AI2THOR_START_TIMEOUT=900
cd "$ADDON"

# 任务清单：优先用行会里的清单文件，否则列出 tasks/ 下所有 tvr_*
case "$SPEC" in
  all)        LIST=$(cat tasks/final_all.txt) ;;
  stage[0-9]) LIST=$(cat "tasks/final_${SPEC}.txt") ;;
  stages1-2)  LIST=$(cat tasks/final_stage1.txt tasks/final_stage2.txt) ;;
  *)      LIST=$(cat "tasks/$SPEC" 2>/dev/null || printf '%s' "$SPEC" | tr ',' '\n') ;;
esac
N=$(printf '%s\n' "$LIST" | grep -c . || true)
echo "[tvr $(date '+%F %T')] arm=$ARM model=$MODEL shards=$SHARDS 任务 $N 条"

launch_arm() {   # $1=base|wm
    local arm="$1" wmflag=""
    [ "$arm" = "wm" ] && wmflag="--wm"
    for s in $(seq 0 $((SHARDS - 1))); do
        local IDS CSV C
        IDS=$(printf '%s\n' "$LIST" | awk -v s="$s" -v n="$SHARDS" 'NR % n == s')
        C=$(printf '%s\n' "$IDS" | grep -c . || true)
        [ "$C" -eq 0 ] && continue
        CSV=$(printf '%s' "$IDS" | tr '\n' ',' | sed 's/,$//')
        setsid nohup /home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python -u \
            run_tvr_agent.py --tasks "$CSV" --model "$MODEL" \
            --base-url http://127.0.0.1:8000/v1 --api-key EMPTY $wmflag \
            --width 800 --height 600 --outdir "runs/tvr_${arm}_${MODEL}_s${s}" \
            >> "/root/autodl-tmp/tvr_${arm}_${MODEL}_s${s}.log" 2>&1 </dev/null &
        echo "  [$arm] 片 $s: $C 条"
        sleep 3
    done
}

if [ "$ARM" = "both" ]; then
    launch_arm base
    echo "等 base 收尾…"
    while pgrep -f "run_tvr_agent.py --tasks" > /dev/null; do sleep 60; done
    echo "[tvr $(date '+%F %T')] base 跑完，接 wm"
    launch_arm wm
else
    launch_arm "$ARM"
fi
sleep 20
echo "在跑的 addon 进程: $(pgrep -cf run_tvr_agent)"
