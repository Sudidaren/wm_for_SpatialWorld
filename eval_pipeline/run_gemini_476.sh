#!/usr/bin/env bash
# Pure Gemini-3.1-Pro-Preview, household 476 tasks, local machine.
# Usage: OPENAI_API_KEY=... OPENAI_BASE_URL=... bash run_gemini_476.sh
set -euo pipefail
cd "$(dirname "$0")"
exec /home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python -u - <<'PY'
import sys
sys.path.insert(0, '/home/sudidaren/spatialworld_eval')
import eval_config as cfg
import supervisor

cfg.SCENES = ['ai2thor', 'procthor', 'virtualhome']
cfg.PROFILE = 'llm'
cfg.LLM_PRESET = 'gemini-3.1-pro'      # model_name=gemini-3.1-pro-preview
cfg.HEADLESS = False                    # local WSLg display; CloudRendering needs credentials
cfg.VERIFY_GOLDEN_REPLAY = False        # batch mode: no golden re-run (saves ~2x sim time)
cfg.WORKERS = 3
cfg.RESUME_SKIP_DECIDED = True
cfg.RUN_NAME = 'gemini31pro_household476_v1'

supervisor.main([
    '--scenes', 'ai2thor,procthor,virtualhome',
    '--profile', 'llm',
    '--no-headless',
    '--no-verify',
    '--workers', '3',
])
PY
