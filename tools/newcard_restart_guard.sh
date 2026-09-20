#!/usr/bin/env bash
# 重启卡上看护（改了孤儿判定之后需要重新起一遍）。
#
# 注意：不要把这行命令直接拼成 `newcard.py run '...pkill -f guard.py...'`，
# 远程 shell 自己的 cmdline 里就带 "guard.py" 这个字样，pkill -f 会连自己
# 一起匹配掉（当场把执行命令的 shell 杀掉，表现为"输出空白、看护没起来"）。
# 所以固定走这个脚本文件，脚本名里不含那个字样。
set -uo pipefail

G=/root/guard.py
LOG=/root/autodl-tmp/guard.log
PY=/root/miniconda3/bin/python

cp "$LOG" "${LOG}.bak.$(date +%s)" 2>/dev/null || true
for p in $(pgrep -f "guard\.p[y]"); do kill "$p" 2>/dev/null || true; done
sleep 2
setsid nohup "$PY" "$G" >>"$LOG" 2>&1 </dev/null &
sleep 3
echo "guard_now=$(pgrep -cf "guard\.p[y]" || true)"
md5sum "$G"
tail -3 "$LOG"
