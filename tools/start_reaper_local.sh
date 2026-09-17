#!/bin/bash
# Run the stray-process reaper as a local background daemon.
# Kept in a file (not passed on the command line) so that the pkill pattern can
# never match the invoking shell.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${INTERVAL:-120}"
MAX_AGE="${MAX_AGE:-8}"
LOG="${LOG:-/tmp/local_reaper.log}"
pkill -f 'reap_stray[s].py' 2>/dev/null
sleep 1
setsid nohup python3 "$ROOT/tools/reap_strays.py" --loop --interval "$INTERVAL" \
    --max-age "$MAX_AGE" --log "$LOG" > /dev/null 2>&1 < /dev/null &
sleep 4
echo -n "reaper 进程数: "
ps -eo args | grep -c 'reap_stray[s]'
