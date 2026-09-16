#!/bin/bash
# cqa1：Qwen3-VL-8B + WM，只跑 AI2-THOR 311 条
for p in $(pgrep -f 'cloud_orchestrator_v[34][.]sh'); do kill "$p" 2>/dev/null; done
sleep 2
cd /root
RUNS_DIR=/root/autodl-tmp/runs_local ORCH_LOG=/root/orchestrator_v2.log \
ORCH_DISPLAY=:99 TOTAL=311 SCENES=ai2thor \
setsid nohup bash /root/cloud_orchestrator_v4.sh \
  'qwen3vl-8b|http://127.0.0.1:8000/v1|off|wm|wm_ai2thor311_qwen3vl8b_v1|5' \
  > /root/orch_wm_ai2thor_cqa1.out 2>&1 < /dev/null &
