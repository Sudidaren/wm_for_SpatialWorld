#!/bin/bash
# weste：Qwen3-VL-30B + WM，只跑 AI2-THOR 311 条（PLAN=off，目标提示/状态查询都关）
# 注意：weste 必须 HEADLESS=1 -> config_builder 才会把 env.platform 设成 CloudRendering；
# 否则 ai2thor 默认选 Linux64，而这张卡没有 Linux64 构建，会去下 769MB（16KB/s，直接卡死）。
for p in $(pgrep -f 'cloud_orchestrator_v[34][.]sh'); do kill "$p" 2>/dev/null; done
sleep 2
cd /root
RUNS_DIR=/root/autodl-tmp/runs_local ORCH_LOG=/root/orchestrator_v2.log \
TOTAL=311 SCENES=ai2thor HEADLESS=1 \
setsid nohup bash /root/cloud_orchestrator_v4.sh \
  'qwen3vl-30b|http://127.0.0.1:8000/v1|off|wm|wm_ai2thor311_qwen3vl30b_v1|6' \
  > /root/orch_wm_ai2thor_weste.out 2>&1 < /dev/null &
