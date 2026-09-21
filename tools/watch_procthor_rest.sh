#!/usr/bin/env bash
# 常驻看护：procthor 两臂进度 + API key 是否又被限流。
# 401 就地把两条线停掉（harness 第 4 条：key 失效 → 暂停该批并汇报）。
set -uo pipefail
LOG=/mnt/d/lightwm_out/watch_procthor_rest.log
BASE_RUN=/home/sudidaren/spatialworld_eval/runs/closed_gpt5_base_procthor
WM_RUN=/home/sudidaren/spatialworld_eval/runs/closed_gpt5_wm_procthor
KEY=$(grep -m1 '^export OPENAI_API_KEY=' "$HOME/.bashrc" | cut -d'"' -f2)

say() { echo "[watch $(date '+%F %T')] $*" | tee -a "$LOG"; }
n() { /home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python -c "import csv
try: print(len(list(csv.DictReader(open('$1/results.csv',encoding='utf-8-sig')))))
except Exception: print(0)"; }

say "开始看护（procthor 两臂）"
last=""
while true; do
    b=$(n "$BASE_RUN"); w=$(n "$WM_RUN")
    code=$(timeout 40 curl -s --noproxy '*' -o /dev/null -w "%{http_code}" \
        https://apic1.ohmycdn.com/v1/chat/completions \
        -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
        -d '{"model":"gpt-5","messages":[{"role":"user","content":"ok"}],"max_tokens":5}' 2>/dev/null)
    line="基线 $b/20 · WM $w/13 · key=$code"
    [ "$line" != "$last" ] && { say "$line"; last="$line"; }
    if [ "$code" = "401" ]; then
        say "🚨 key 又被限流（401），停两条线"
        for pat in "run_procthor_rest_loca[l]" "closed_gpt5_base_proctho[r]" "closed_gpt5_wm_proctho[r]" "work.run_tas[k]"; do
            for p in $(pgrep -f "$pat"); do kill -9 "$p" 2>/dev/null || true; done
        done
        sleep 3
        for p in $(pgrep -f "[t]hor-Linux6[4]"); do kill -9 "$p" 2>/dev/null || true; done
        say "已停。等用户给新 key / 提额"
        exit 1
    fi
    if [ "$b" -ge 20 ] && [ "$w" -ge 13 ]; then
        say "✅ 两臂都到齐（$b/20 + $w/13）"
        exit 0
    fi
    sleep 300
done
