#!/usr/bin/env bash
# 卡上跑 5 臂消融（方案 B 的任务清单：诊断层 + 官方 39 条的并集）。
#
#   bash card_run_ablation_planb.sh <模型名> [arms...]
#
# 5 个臂（隔离"哪条通道真在被读"）：
#   base      不开 WM                                     —— 对照
#   wm        现状全开                                    —— 上限
#   noinject  WM 照跑但不注入 prompt（WM_NO_INJECT=1）    —— 证明增益来自文本
#   notarget  关目标物位置提示（WM_TARGET_HINT=0）        —— 主通道（占提示量 67%）
#   nomem     只报当前帧、不留记忆（LIGHTWM_MEMORY_FRAMES=0）—— 感知 vs 记忆
#
# 全跑卡上自有 vLLM → token $0，只花卡时。
set -uo pipefail
MODEL="${1:?用法: card_run_ablation_planb.sh <model> [arms...]}"
shift || true
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(base wm noinject notarget nomem)

SWE=/home/sudidaren/spatialworld_eval
ADDON=/home/sudidaren/tvrbench_addon
LIST=/root/ablation_tasks.txt          # 由控制端按模型上传
export DISPLAY=:99 TMPDIR_OVERRIDE=/tmp/lightwm_tmp LIGHTWM_ROOT=/home/sudidaren/lightwm_phases
export AI2THOR_SERVER_TIMEOUT=900 AI2THOR_START_TIMEOUT=900
export PROCTHOR_DATASET_DIR=/root/.prior/datasets/allenai/procthor-10k/439193522244720b86d8c81cde2e51e3a4d150cf

OFF=$(paste -sd, "$LIST")
TVR=$(paste -sd, "$ADDON/tasks/ablation_tvr.txt")
echo "[abl $(date '+%F %T')] 模型=$MODEL 官方+诊断 $(grep -c . $LIST) 条 + TVR $(grep -c . $ADDON/tasks/ablation_tvr.txt) 条"

arm_env() {
    case "$1" in
        base)     echo "" ;;
        noinject) echo "WM_NO_INJECT=1" ;;
        notarget) echo "WM_TARGET_HINT=0" ;;
        nomem)    echo "LIGHTWM_MEMORY_FRAMES=0" ;;
        *)        echo "" ;;
    esac
}

for arm in "${ARMS[@]}"; do
    envs=$(arm_env "$arm")
    echo "[abl $(date '+%F %T')] === arm=$arm model=$MODEL envs='$envs' ==="
    ( cd "$SWE"
      if [ "$arm" = base ]; then PROF=llm; else PROF=wm; fi
      env MODEL_NAME="$MODEL" BASE_URL=http://127.0.0.1:8000/v1 PROFILE="$PROF" \
          RUN_NAME="abl_${arm}_${MODEL}" SCENES=ai2thor,procthor WORKERS=3 \
          LLM_API_KEY=EMPTY SMOKE_TASKS="$OFF" $envs \
          bash run_wm_gemini_ai2thor.sh ) >> "/root/autodl-tmp/abl_${arm}_${MODEL}_official.log" 2>&1
    ( cd "$ADDON"
      WMFLAG=""; [ "$arm" != base ] && WMFLAG="--wm"
      env MODEL_NAME="$MODEL" $envs \
        /home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python -u run_tvr_agent.py \
        --tasks "$TVR" --model "$MODEL" --base-url http://127.0.0.1:8000/v1 \
        --api-key EMPTY $WMFLAG --width 800 --height 600 \
        --outdir "runs/abl_${arm}_${MODEL}_tvr" ) >> "/root/autodl-tmp/abl_${arm}_${MODEL}_tvr.log" 2>&1
    echo "[abl $(date '+%F %T')] === arm=$arm 跑完 ==="
done
echo "[abl $(date '+%F %T')] 全部 ${#ARMS[@]} 个臂跑完"
