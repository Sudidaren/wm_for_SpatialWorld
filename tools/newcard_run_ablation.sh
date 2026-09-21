#!/usr/bin/env bash
# 在卡上跑一轮消融：官方 39 条（ai2thor 27 + procthor 12）+ TVR 20 条，共 59 条。
#
#   bash run_ablation.sh <model> [arms...]      默认跑全部 5 个臂
#
# 5 个臂（"哪条通道真在被读"决定的，不是拍脑袋）：
#   base      不开 WM                                  —— 对照
#   wm        现状全开                                 —— 上限
#   noinject  WM 照跑但不说话（WM_NO_INJECT=1）        —— 证明增益来自提示文本
#   notarget  关目标物位置提示（WM_TARGET_HINT=0）     —— 占总提示量 67% 的主通道
#   nomem     只报当前帧、不留记忆（LIGHTWM_MEMORY_FRAMES=0）—— 记忆 vs 现见
#
# 全跑在卡上自有 vLLM 上 → token $0，只花卡时。
set -uo pipefail
MODEL="${1:?用法: run_ablation.sh <model> [arms...]}"
shift || true
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(base wm noinject notarget nomem)

SWE=/home/sudidaren/spatialworld_eval
ADDON=/home/sudidaren/tvrbench_addon
export DISPLAY=:99 TMPDIR_OVERRIDE=/tmp/lightwm_tmp LIGHTWM_ROOT=/home/sudidaren/lightwm_phases
export AI2THOR_SERVER_TIMEOUT=900 AI2THOR_START_TIMEOUT=900
export PROCTHOR_DATASET_DIR=/root/.prior/datasets/allenai/procthor-10k/439193522244720b86d8c81cde2e51e3a4d150cf

OFF=$(paste -sd, "$ADDON/tasks/ablation_official.txt")
TVR=$(paste -sd, "$ADDON/tasks/ablation_tvr.txt")

arm_env() {   # $1=arm -> 打印额外环境变量
    case "$1" in
        base)     echo "" ;;          # base 的 PROFILE 在下面单独设成 llm
        noinject) echo "WM_NO_INJECT=1" ;;
        notarget) echo "WM_TARGET_HINT=0" ;;
        nomem)    echo "LIGHTWM_MEMORY_FRAMES=0" ;;
        *)        echo "" ;;
    esac
}

for arm in "${ARMS[@]}"; do
    envs=$(arm_env "$arm")
    echo "[abl $(date '+%F %T')] === arm=$arm model=$MODEL envs='$envs' ==="
    # ---- 官方 ai2thor + procthor（走 supervisor）----
    ( cd "$SWE"
      if [ "$arm" = base ]; then PROF=llm; else PROF=wm; fi
      env MODEL_NAME="$MODEL" BASE_URL=http://127.0.0.1:8000/v1 PROFILE="$PROF" \
          RUN_NAME="abl_${arm}_${MODEL}" SCENES=ai2thor,procthor WORKERS=3 \
          LLM_API_KEY=EMPTY SMOKE_TASKS="$OFF" $envs \
          bash run_wm_gemini_ai2thor.sh ) >> "/root/autodl-tmp/abl_${arm}_${MODEL}_official.log" 2>&1
    # ---- TVR（走 addon）----
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
