#!/usr/bin/env bash
# 等两边跑完 → 重出两张主表 → 提交并推 GitHub。
set -uo pipefail
LOG=/mnt/d/lightwm_out/finalize_and_push.log
PY=/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python
ET=/home/sudidaren/eval_tables
REPO=/home/sudidaren/lightwm_phases
BASE_RUN=/home/sudidaren/spatialworld_eval/runs/closed_gpt5_base_procthor
WM_RUN=/home/sudidaren/spatialworld_eval/runs/closed_gpt5_wm_procthor

say() { echo "[fin $(date '+%F %T')] $*" | tee -a "$LOG"; }
count() { "$PY" -c "import csv,sys
try: print(len(list(csv.DictReader(open(sys.argv[1]+'/results.csv',encoding='utf-8-sig')))))
except Exception: print(0)" "$1"; }

say "等 procthor 两臂跑完（基线 20 + WM 13）"
for i in $(seq 1 360); do
    b=$(count "$BASE_RUN")
    w=$(count "$WM_RUN")
    if [ $((i % 10)) -eq 1 ]; then say "  基线 $b/20  WM $w/13"; fi
    if [ "${b:-0}" -ge 20 ] && [ "${w:-0}" -ge 13 ]; then say "两臂到齐"; break; fi
    sleep 60
done

say "重出两张主表"
cd "$ET"
"$PY" build_main_table.py --mode sample >/dev/null 2>&1 && say "  main_table.md OK"
"$PY" build_main_table.py --mode full >/dev/null 2>&1 && say "  main_table_full.md OK"
"$PY" build_main_table_tex.py >/dev/null 2>&1 && say "  tex OK"

say "拷进仓库并推送"
mkdir -p "$REPO/eval_tables"
cp "$ET"/build_main_table.py "$ET"/build_main_table_tex.py "$ET"/main_table.md "$ET"/main_table_full.md "$ET"/main_table.tex "$ET"/main_table_snippet.tex "$REPO/eval_tables/" 2>/dev/null
cd "$REPO"
git add -A eval_tables plans docs tools patches >/dev/null 2>&1
git -c user.name=codex -c user.email=codex@local commit -q -m "Main table on the common 120+20 sample, plus each model full batch" >/dev/null 2>&1 && say "  已提交" || say "  没有改动"
git push origin master >> "$LOG" 2>&1 && say "推送完成" || say "推送失败"
