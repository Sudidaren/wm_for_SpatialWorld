#!/usr/bin/env bash
# 消融实验统一入口：
#   PROFILE=llm  -> 纯模型基线（无 WM）
#   PROFILE=wm   -> WM 接入（PLAN 字段已废弃：子目标分解 2026-09-17 删除）
#
# 用法：
#   MODEL_NAME=qwen3vl-8b BASE_URL=http://127.0.0.1:18001/v1 PLAN=off \
#     PROFILE=llm RUN_NAME=qwen3vl8b_438_v1 SCENES=ai2thor,procthor WORKERS=4 \
#     bash run_ablation.sh
set -euo pipefail
cd "$(dirname "$0")"

MODEL_NAME="${MODEL_NAME:-qwen3vl-30b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:18000/v1}"
PLAN="${PLAN:-off}"
PROFILE="${PROFILE:-wm}"
RUN_NAME="${RUN_NAME:-ablation_run}"
SCENES="${SCENES:-ai2thor,procthor}"
WORKERS="${WORKERS:-4}"

export LIGHTWM_ROOT=/home/sudidaren/lightwm_phases
export LIGHTWM_STORAGE_ROOT=/home/sudidaren/lightwm_phases
export LIGHTWM_DETECTOR=rfdetr_small_depth
export LIGHTWM_DETECTOR_PATH="$LIGHTWM_ROOT/checkpoints/rfdetr_small_228094/checkpoint_best_total.pth"
export PERCEPTION_CKPT="$LIGHTWM_ROOT/checkpoints/small_objects_20260910/dense_depth_best.pt"
export LIGHTWM_OBJ_THR=0.40
export LIGHTWM_ZOOM=0
# 历史体积优化（对基线与 WM 条件一致生效）：
#   WM_HISTORY_TURNS  近端历史轮数（默认沿用冻结配置的 29）
#   WM_HISTORY_MAX_SIDE 历史图片最长边（>0 时转 JPEG 下采样）
export WM_HISTORY_TURNS="${WM_HISTORY_TURNS:-29}"
export WM_HISTORY_MAX_SIDE="${WM_HISTORY_MAX_SIDE:-0}"
echo "[ablation] history_turns=$WM_HISTORY_TURNS history_max_side=$WM_HISTORY_MAX_SIDE"

echo "[ablation] model=$MODEL_NAME url=$BASE_URL profile=$PROFILE plan=$PLAN run=$RUN_NAME workers=$WORKERS"

exec /home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python -u - \
        "$MODEL_NAME" "$BASE_URL" "$PROFILE" "$RUN_NAME" "$SCENES" "$WORKERS" <<'PY'
import os
import sys

sys.path.insert(0, '/home/sudidaren/spatialworld_eval')
import eval_config as cfg
import config_builder as cb
import supervisor

model_name, base_url, profile, run_name, scenes_s, workers_s = sys.argv[1:7]
scenes = [s for s in scenes_s.split(',') if s]
workers = int(workers_s)

if profile == 'wm':
    import wm_config_patch
    wm_config_patch.install(cb)          # 感知后端 + WM 提示开关
    cfg.PERCEPTION_CKPT = (f'{wm_config_patch.WM_ROOT}/checkpoints'
                           '/small_objects_20260910/dense_depth_best.pt')

# 历史窗口覆盖（冻结文件不改，运行时打补丁）
_hist = int(os.environ.get('WM_HISTORY_TURNS', '29') or 29)
_orig_ctx = cb._apply_context_block


def _apply_context_block_patched(data):
    _orig_ctx(data)
    data.setdefault('context_management', {})
    data['context_management']['short_term_history_window_size'] = _hist


cb._apply_context_block = _apply_context_block_patched

cfg.SCENES = scenes
cfg.PROFILE = 'wingman_llm' if profile == 'wm' else 'llm'
cfg.LLM_PRESET = 'qwen3.5'
cfg.LLM_OVERRIDES = {
    'provider': 'openai',
    'model_name': model_name,
    'base_url': base_url,
    # 本地 vLLM 不校验 key，所以卡上的臂一直用占位符。走外部网关（例如
    # WM+Gemini）时必须带真 key：LLM_API_KEY 优先，其次 OPENAI_API_KEY。
    'api_key': (os.environ.get('LLM_API_KEY')
                or os.environ.get('OPENAI_API_KEY')
                or 'EMPTY'),
}
cfg.HEADLESS = False
if os.environ.get('HEADLESS') == '1':
    # 云上无 X 显示：走 CloudRendering（config_builder 会自动设 env.platform）
    cfg.HEADLESS = True
cfg.VERIFY_GOLDEN_REPLAY = False
cfg.WORKERS = workers
cfg.RESUME_SKIP_DECIDED = True
cfg.RUN_NAME = run_name

supervisor.main([
    '--scenes', ','.join(scenes),
    '--profile', 'wingman_llm' if profile == 'wm' else 'llm',
    '--no-headless' if os.environ.get('HEADLESS') != '1' else '--headless',
    '--no-verify',
    '--workers', str(workers),
    *[a for t in (os.environ.get('SMOKE_TASKS') or '').split(',') if t
      for a in ('--task', t)],
])
PY
