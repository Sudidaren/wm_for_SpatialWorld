#!/usr/bin/env bash
# Keep the arm queue alive when the orchestrator is the thing that dies.
#
# The watchdog covers process hygiene and the orchestrator covers a dead
# supervisor, but nothing covered "the orchestrator itself is gone": on
# 2026-09-19 it was killed by hand and the batch sat idle for six minutes with
# three orphaned workers still holding simulators.  This loop is that missing
# layer.  It does nothing clever -- if the orchestrator is not running, it
# re-runs the launch script, and the orchestrator skips everything already
# finalized.
#
#   LAUNCH=/root/arm_launch.sh INTERVAL=90 MAX_RESTARTS=12 \
#     setsid nohup bash tools/card_keepalive.sh &
#
# Disable with:  touch /root/keepalive.off
# Log:           /root/autodl-tmp/logs/keepalive.log

set -u
LAUNCH="${LAUNCH:-/root/arm_launch.sh}"
INTERVAL="${INTERVAL:-90}"
MAX_RESTARTS="${MAX_RESTARTS:-12}"
LOG="${LOG:-/root/autodl-tmp/logs/keepalive.log}"
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

PIDFILE="${PIDFILE:-/root/keepalive.pid}"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log "keepalive start (launch=$LAUNCH interval=${INTERVAL}s max=${MAX_RESTARTS} pid=$$)"
restarts=0
while true; do
  sleep "$INTERVAL"
  if [ -f /root/keepalive.off ]; then
    log "keepalive.off present -> stop"
    exit 0
  fi
  if [ ! -f "$LAUNCH" ]; then
    log "no launch script at $LAUNCH -> stop"
    exit 0
  fi
  # awk, not pgrep: this script names the orchestrator script, so a pattern
  # match would find itself.
  if [ -n "$(ps -eo pid=,args= | awk '/cloud_orchestrator_v4\.sh [a-z0-9]/{print $1}')" ]; then
    continue
  fi

  # The orchestrator can also exit on purpose, when every stage is finished.
  # The arms it owns are listed in /root/keepalive.arms, so "finished" is a
  # question about those runs only -- not about every run dir on the box.
  still_open=$(/root/miniconda3/bin/python - <<'PY' 2>/dev/null || echo 1
import json, os
arms = [a.strip() for a in open('/root/keepalive.arms') if a.strip()]
try:
    total = int(open('/root/keepalive.total').read().strip())
except Exception:
    total = 311
open_n = 0
for a in arms:
    p = f'/root/autodl-tmp/runs_local/{a}/state.json'
    if not os.path.exists(p):
        open_n += 1
        continue
    try:
        tasks = json.load(open(p)).get('tasks', [])
    except Exception:
        open_n += 1
        continue
    done = sum(1 for t in tasks
               if str(t.get('completed', '')).lower() in ('true', 'false')
               or (t.get('status') == 'failed_external'
                   and int(t.get('attempts') or 0) >= 3))
    if done < total:
        open_n += 1
print(open_n)
PY
)
  if [ "${still_open:-0}" = "0" ]; then
    log "every arm in keepalive.arms is finished -> stop"
    exit 0
  fi
  if [ ! -f /root/keepalive.arms ]; then
    log "no /root/keepalive.arms -> stop (nothing to finish)"
    exit 0
  fi
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    log "max restarts ($MAX_RESTARTS) reached -> stop (inspect by hand)"
    exit 0
  fi

  restarts=$((restarts + 1))
  log "orchestrator missing, $still_open run dir(s) unfinished -> restart #$restarts"
  setsid nohup bash "$LAUNCH" > /root/orch_keepalive.out 2>&1 < /dev/null &
  sleep 20
  if [ -n "$(ps -eo pid=,args= | awk '/cloud_orchestrator_v4\.sh [a-z0-9]/{print $1}')" ]; then
    log "  restart #$restarts ok"
  else
    log "  restart #$restarts did not come up (see /root/orch_keepalive.out)"
    tail -5 /root/orch_keepalive.out 2>/dev/null | sed 's/^/    /'
  fi
done
