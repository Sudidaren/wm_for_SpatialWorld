#!/usr/bin/env bash
# Freeze verification: golden replay on three tasks must all be Completed=true.
set -euo pipefail
cd "$(dirname "$0")"
RUN_DIR="runs/verify_freeze_$(date +%Y%m%d_%H%M%S)"
/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python -u - "$RUN_DIR" <<'PY'
import sys
run_dir = sys.argv[1]
sys.path.insert(0, '/home/sudidaren/spatialworld_eval')
import eval_config as cfg
import supervisor
cfg.PROFILE = 'rl'
cfg.RL_POLICY_IMPORT = ''
cfg.RL_CHECKPOINT = ''
cfg.SCENES = ['ai2thor', 'procthor']
cfg.HEADLESS = False
cfg.VERIFY_GOLDEN_REPLAY = False
cfg.WORKERS = 1
cfg.RESUME_SKIP_DECIDED = False
cfg.RUN_NAME = run_dir.split('/')[-1]
supervisor.main(['--scenes', 'ai2thor,procthor', '--profile', 'rl',
                 '--task', 'ai2thor05523', '--task', 'ai2thor02436',
                 '--task', 'procthor304',
                 '--no-headless', '--no-verify', '--workers', '1'])
PY
/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python - "$RUN_DIR/results.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8-sig')))
bad = [r['Task ID'] for r in rows if (r['Completed'] or '').lower() != 'true']
print(f'verified {len(rows)} tasks; non-true: {bad}')
if bad or len(rows) != 3:
    raise SystemExit('FREEZE VERIFICATION FAILED')
print('FREEZE VERIFICATION PASSED')
PY
