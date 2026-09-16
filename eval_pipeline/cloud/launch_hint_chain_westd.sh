#!/bin/bash
# westd (Kimi-VL-A3B): two stages, run sequentially.
#   stage 1 = pure WM (target hint OFF), resume wm_ai2thor311_kimivl_a3b_v1
#   stage 2 = WM + target hint (WM_TARGET_HINT=1) -> wmhint_ai2thor311_kimivl_a3b_v1
for p in $(pgrep -f 'cloud_orchestrator_v[34][.]sh'); do kill "$p" 2>/dev/null; done
sleep 3
setsid nohup bash -c '
cd /root
export RUNS_DIR=/root/autodl-tmp/runs_local
export ORCH_LOG=/root/orchestrator_v2.log
export ORCH_DISPLAY=:99
export TOTAL=311 SCENES=ai2thor
bash /root/cloud_orchestrator_v4.sh \
  "kimivl-a3b|http://127.0.0.1:8000/v1|off|wm|wm_ai2thor311_kimivl_a3b_v1|5" \
  >> /root/orch_chain_westd.log 2>&1
export WM_TARGET_HINT=1
bash /root/cloud_orchestrator_v4.sh \
  "kimivl-a3b|http://127.0.0.1:8000/v1|off|wm|wmhint_ai2thor311_kimivl_a3b_v1|5" \
  >> /root/orch_chain_westd.log 2>&1
' > /root/orch_chain_westd.out 2>&1 < /dev/null &
