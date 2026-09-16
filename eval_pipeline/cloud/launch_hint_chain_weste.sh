#!/bin/bash
# weste (30B): two stages, run sequentially.
#   stage 1 = pure WM (target hint OFF), resume wm_ai2thor311_qwen3vl30b_v1
#   stage 2 = WM + target hint (WM_TARGET_HINT=1) -> wmhint_ai2thor311_qwen3vl30b_v1
# weste MUST keep HEADLESS=1, otherwise ai2thor picks Linux64 and starts a
# 769MB download at 16KB/s (which looks exactly like a hung run).
for p in $(pgrep -f 'cloud_orchestrator_v[34][.]sh'); do kill "$p" 2>/dev/null; done
sleep 3
setsid nohup bash -c '
cd /root
export RUNS_DIR=/root/autodl-tmp/runs_local
export ORCH_LOG=/root/orchestrator_v2.log
export TOTAL=311 SCENES=ai2thor HEADLESS=1
bash /root/cloud_orchestrator_v4.sh \
  "qwen3vl-30b|http://127.0.0.1:8000/v1|off|wm|wm_ai2thor311_qwen3vl30b_v1|6" \
  >> /root/orch_chain_weste.log 2>&1
export WM_TARGET_HINT=1
bash /root/cloud_orchestrator_v4.sh \
  "qwen3vl-30b|http://127.0.0.1:8000/v1|off|wm|wmhint_ai2thor311_qwen3vl30b_v1|6" \
  >> /root/orch_chain_weste.log 2>&1
' > /root/orch_chain_weste.out 2>&1 < /dev/null &
