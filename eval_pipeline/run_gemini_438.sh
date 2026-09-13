#!/usr/bin/env bash
# Pure Gemini-3.1-Pro-Preview on AI2-THOR + ProcTHOR (438 tasks).
# VirtualHome is excluded until its local Unity launcher issue is fixed.
set -euo pipefail
cd "$(dirname "$0")"
exec /home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python -u - <<'PY'
import sys
sys.path.insert(0, '/home/sudidaren/spatialworld_eval')
import eval_config as cfg
import supervisor

cfg.SCENES = ['ai2thor', 'procthor']
cfg.PROFILE = 'llm'
cfg.LLM_PRESET = 'gemini-3.1-pro'      # model_name=gemini-3.1-pro-preview
cfg.HEADLESS = False                    # local WSLg display
cfg.VERIFY_GOLDEN_REPLAY = False
cfg.WORKERS = 3
cfg.RESUME_SKIP_DECIDED = True
cfg.RUN_NAME = 'gemini31pro_ai2thor_procthor_438_v1'

supervisor.main([
    '--scenes', 'ai2thor,procthor',
    '--profile', 'llm',
    '--no-headless',
    '--no-verify',
    '--workers', '3',
])
PY
