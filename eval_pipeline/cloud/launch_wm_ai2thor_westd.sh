#!/bin/bash
# westd：Kimi-VL-A3B + WM，只跑 AI2-THOR 311 条
for p in $(pgrep -f 'cloud_orchestrator_v[34][.]sh'); do kill "$p" 2>/dev/null; done
sleep 2
cd /root
RUNS_DIR=/root/autodl-tmp/runs_local ORCH_LOG=/root/orchestrator_v2.log \
ORCH_DISPLAY=:99 TOTAL=311 SCENES=ai2thor \
setsid nohup bash /root/cloud_orchestrator_v4.sh \
  'kimivl-a3b|http://127.0.0.1:8000/v1|off|wm|wm_ai2thor311_kimivl_a3b_v1|5' \
  > /root/orch_wm_ai2thor_westd.out 2>&1 < /dev/null &
