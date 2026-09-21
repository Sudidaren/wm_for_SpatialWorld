#!/usr/bin/env bash
# 把 GPT-5 基线批次铺到三张已经就绪的卡上并启动。
# 用现有卡的理由：它们已经装好环境（仓库 + AI-THOR + eval_sets + Xvfb :99），
# 而且 208 核 / 1TB 内存只用了 3-6 个 worker，空得多。
set -uo pipefail
cd /home/sudidaren/lightwm_phases/tools
P=/home/sudidaren/lightwm_phases/plans/shards

# 1) key 落成一个 600 的文件（不进命令行、不回显）
umask 077
KEY=$(grep -m1 '^export OPENAI_API_KEY=' "$HOME/.bashrc" | cut -d'"' -f2)
[ -n "$KEY" ] || { echo "拿不到 key"; exit 2; }
printf 'export LLM_API_KEY=%s\n' "$KEY" > /tmp/.llm_env
chmod 600 /tmp/.llm_env

CARDS=(30b 8b kimi)
for i in 0 1 2; do
    tag="${CARDS[$i]}"
    echo "=== 铺卡 $tag (shard $i) ==="
    NEWCARD=$tag timeout 120 python3 newcard.py put /tmp/.llm_env /root/.llm_key
    NEWCARD=$tag timeout 120 python3 newcard.py put "$P/gpt5_base_s${i}.txt" "/root/gpt5_base_tasks_s${i}.txt"
    NEWCARD=$tag timeout 120 python3 newcard.py put card_run_gpt5_base.sh /root/card_run_gpt5_base.sh
    # 看护白名单：否则 30 分钟后它会把这个内联 python 的 supervisor 当"卡死脚本"杀掉
    NEWCARD=$tag timeout 120 python3 newcard.py run \
      "sed -i 's/\"tensorboard\", \"multiprocessing.resource_tracker\")/\"tensorboard\", \"multiprocessing.resource_tracker\", \"main_gpt5\")/' /root/guard.py; grep -c main_gpt5 /root/guard.py; for p in \$(pgrep -f 'guard\\.py'); do kill -TERM \$p 2>/dev/null; done; sleep 2; setsid nohup /root/miniconda3/bin/python /root/guard.py >> /root/autodl-tmp/guard.log 2>&1 </dev/null & sleep 2; pgrep -cf 'guard\\.py'"
done
rm -f /tmp/.llm_env
echo "=== 铺完 ==="
