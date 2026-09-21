#!/usr/bin/env bash
# 在三张卡上启动 GPT-5 基线分片（6 worker/卡）
set -uo pipefail
cd /home/sudidaren/lightwm_phases/tools
CARDS=(30b 8b kimi)
for i in 0 1 2; do
    tag="${CARDS[$i]}"
    echo "=== 启动 $tag shard $i ==="
    NEWCARD=$tag timeout 120 python3 newcard.py run \
      "cd /root && chmod +x /root/card_run_gpt5_base.sh && setsid nohup bash /root/card_run_gpt5_base.sh $i >> /root/autodl-tmp/main_gpt5_base_s$i.log 2>&1 </dev/null & sleep 20; echo '--- 日志 ---'; head -12 /root/autodl-tmp/main_gpt5_base_s$i.log; echo '--- worker ---'; pgrep -cf work.run_task"
done
