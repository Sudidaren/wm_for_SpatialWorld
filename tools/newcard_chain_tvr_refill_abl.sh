#!/usr/bin/env bash
# 卡上的一条龙队列（用户 2026-09-21 14:2x 定）：
#   1. 等 TVR base→wm 那条链跑完（run_tvr.sh both 起的）
#   2. 跑 438 批的补测（ai2thor + procthor 的 failed_external）
#   3. 跑消融（5 臂 × 59 条，全在自有 vLLM 上，token $0）
#
# 用法: bash chain_tvr_refill_abl.sh <RUN_NAME> <MODEL> <TVR_SHARDS>
set -uo pipefail
RUN="${1:?用法: chain_tvr_refill_abl.sh <RUN_NAME> <MODEL> <TVR_SHARDS>}"
MODEL="${2:?}"
SHARDS="${3:-4}"
LOG=/root/autodl-tmp/chain_all.log

say() { echo "[chain $(date '+%F %T')] $*" >> "$LOG"; }
say "启动：run=$RUN model=$MODEL shards=$SHARDS"

# 1) 等 TVR（base 和 wm 都跑完再动手）
#
# 顺序很重要：**先**等 run_tvr.sh 这个"调度者"消失，**再**等 addon 进程消失。
# 反过来会有个致命的空窗：base 刚跑完、run_tvr.sh 还没把 wm 的 shard 拉起来
# 的那一瞬间，addon 进程数是 0，等待条件立刻为真 —— 2026-09-21 14:26 就是
# 这么把补测和消融提前放出来的。
while pgrep -f "run_tvr\.sh bot[h]" > /dev/null; do sleep 60; done
while pgrep -f "run_tvr_agent.p[y]" > /dev/null; do sleep 60; done
say "TVR 链已结束"

# 2) 补测
bash /root/refill_failed.sh "$RUN" "$MODEL" 3 >> "$LOG" 2>&1
say "补测完成"

# 3) 消融
bash /root/run_ablation.sh "$MODEL" >> "$LOG" 2>&1
say "消融完成，全部收工"
