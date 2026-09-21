#!/usr/bin/env bash
# 等基线分片跑完后，在三张卡上启动 GPT-5 + WM 的分片。
# 用法：bash launch_gpt5_wm.sh [每卡 worker 数，默认 4]
set -uo pipefail
cd /home/sudidaren/lightwm_phases/tools
W="${1:-4}"
CARDS=(30b 8b kimi)
for i in 0 1 2; do
    tag="${CARDS[$i]}"
    echo "=== $tag shard $i ==="
    NEWCARD=$tag timeout 120 python3 newcard.py run \
      "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"
    NEWCARD=$tag timeout 120 python3 newcard.py put card_run_gpt5_wm.sh /root/card_run_gpt5_wm.sh
    NEWCARD=$tag timeout 120 python3 newcard.py run \
      "cd /root && grep -c 'PROFILE=wm' /root/card_run_gpt5_wm.sh && setsid nohup env WORKERS=$W bash /root/card_run_gpt5_wm.sh $i >> /root/autodl-tmp/main_gpt5_wm_s$i.log 2>&1 </dev/null & sleep 25; tail -6 /root/autodl-tmp/main_gpt5_wm_s$i.log"
done
