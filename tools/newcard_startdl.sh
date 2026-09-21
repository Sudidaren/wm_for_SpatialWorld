#!/bin/bash
# Start (or restart) a ModelScope download on a card, in the background.
# ASCII only: this file travels through paramiko and non-ASCII gets mangled.
#
#   bash startdl.sh <repo_id> <dest_dir> <log> [jobs]
set -u
REPO="$1"; DEST="$2"; LOG="$3"; JOBS="${4:-6}"
PY=/root/miniconda3/bin/python

# kill any previous downloader for this dest (bracket trick: do not match self)
for pat in "[f]astdl.py" "[m]odelscope"; do
    ps -eo pid,args --no-headers | grep -E "$pat" | awk '{print $1}' | xargs -r kill -9
done
sleep 1

mkdir -p "$DEST"
cd /root
setsid nohup "$PY" /root/fastdl.py "$REPO" "$DEST" "$JOBS" > "$LOG" 2>&1 < /dev/null &
sleep 20
echo "=== running? ==="
ps -eo args --no-headers | grep -c "[f]astdl.py"
echo "=== log ==="
tail -6 "$LOG"
echo "=== on disk ==="
du -sh "$DEST"
