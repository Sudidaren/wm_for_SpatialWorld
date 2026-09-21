#!/usr/bin/env bash
# 闭源主批次一览：本机 Gemini+WM、三张卡的 GPT-5 基线分片。
set -uo pipefail
cd /home/sudidaren/lightwm_phases/tools

echo "时间: $(date '+%F %T')"
echo "--- 本机 Gemini+WM (closed_gemini_wm) ---"
/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python - <<'PY'
import csv, collections, os
f='/home/sudidaren/spatialworld_eval/runs/closed_gemini_wm/results.csv'
try:
    r=list(csv.DictReader(open(f,encoding='utf-8-sig')))
    c=collections.Counter(x['Environment'] for x in r)
    s=sum(1 for x in r if str(x.get('Success')).lower()=='true')
    print(f'  判定 {len(r)}/71  {dict(c)}  成功 {s} ({100*s/max(len(r),1):.0f}%)')
except Exception as e: print('  还没有结果', e)
PY
echo "  负载: $(uptime | sed 's/.*load average: //')"

echo "--- 云上 GPT-5 基线（3 分片，共 134 条）---"
tot=0
CARDS=(30b 8b kimi)
for i in 0 1 2; do
    tag=${CARDS[$i]}
    n=$(NEWCARD=$tag timeout 60 python3 newcard.py run \
      "/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python -c \"import csv;print(len(list(csv.DictReader(open('/home/sudidaren/spatialworld_eval/runs/main_gpt5_base_s$i/results.csv',encoding='utf-8-sig')))))\" 2>/dev/null || echo 0" 2>/dev/null | tail -1 | tr -d '\r')
    n=${n:-0}
    tot=$((tot+n))
    printf "  %-4s shard %d: 判定 %s\n" "$tag" "$i" "$n"
done
echo "  合计 $tot/134"
