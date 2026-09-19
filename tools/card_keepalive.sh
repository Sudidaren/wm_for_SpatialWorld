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

# How many of this card's arms still have work left.  Only the runs listed in
# /root/keepalive.arms count: the boxes also hold historical batches the paper
# cites, and those must never make this loop think work is outstanding.
still_open() {
  /root/miniconda3/bin/python - <<'PY' 2>/dev/null || echo 1
import json, os
try:
    arms = [a.strip() for a in open('/root/keepalive.arms') if a.strip()]
except Exception:
    print(1); raise SystemExit
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
}

log "keepalive start (launch=$LAUNCH interval=${INTERVAL}s max=${MAX_RESTARTS} pid=$$)"
restarts=0
vllm_bad=0
stack_restarts=0
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
  # Finished cards must stop before the vLLM check: a card that is done has
  # no reason to keep a model server, and restarting the stack for it would
  # spin forever.
  if [ "$(still_open)" = "0" ]; then
    log "every arm in keepalive.arms is finished -> stop"
    exit 0
  fi
  # -- vLLM health -------------------------------------------------------
  # A dead or OOM-killed server does not stop the arm: every remaining task
  # just records a model failure, and the batch looks like a result.  Card A
  # lost its engine to CUDA OOM at 14:18 and kept "running" for 30 minutes.
  vllm_ok=1
  if ! curl -sf -m 6 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    vllm_ok=0
    if pgrep -f "[v]llm serve" >/dev/null 2>&1; then
      vllm_bad=$((vllm_bad + 1))
      log "vLLM process alive, endpoint down (check $vllm_bad/4; a first load takes ~4 min)"
      [ "$vllm_bad" -lt 4 ] && continue
      log "vLLM unresponsive for $vllm_bad checks -> restart the whole stack"
    else
      log "vLLM process is gone -> restart the whole stack"
    fi
  else
    vllm_bad=0
  fi

  if [ "$vllm_ok" = "0" ]; then
    if [ "$stack_restarts" -ge "$MAX_RESTARTS" ]; then
      log "max stack restarts ($MAX_RESTARTS) reached -> stop (inspect by hand)"
      exit 0
    fi
    stack_restarts=$((stack_restarts + 1))
    # stop the control plane and the arms, then let arm_launch.sh bring the
    # whole thing back (it starts vLLM, waits for the endpoint, then the queue)
    #
    # The vLLM process has to go too, even when it looks alive: an API server
    # whose engine died still holds its GPU allocation, and the fresh server
    # then refuses to start ("Free memory 66.75/94.97 GiB is less than desired
    # 0.78") -- that is exactly how card E sat in a restart loop.
    for p in $(pgrep -f "[v]llm serve"); do
      log "  killing stale vLLM pid=$p (engine dead, still holding GPU)"
      kill -TERM "$p" 2>/dev/null
    done
    for p in $(ps -eo pid=,args= | awk '/cloud_orchestrator_v4\.sh [a-z0-9]/{print $1}'); do
      kill -TERM "$p" 2>/dev/null
    done
    sleep 8
    for p in $(pgrep -f "[v]llm serve"); do kill -9 "$p" 2>/dev/null; done
    # wait for the GPU to actually come back before launching another server
    for _ in $(seq 1 12); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      [ -z "$used" ] && break
      [ "$used" -lt 2000 ] && break
      sleep 5
    done
    log "  GPU after cleanup: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)"
    for arm in $(cat /root/keepalive.arms 2>/dev/null); do
      for p in $(pgrep -f "[p]ython -u -"); do
        if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "RUN_NAME=$arm"; then
          log "  stopping arm $arm (supervisor pid=$p)"
          for c in $(pgrep -P "$p" 2>/dev/null); do
            for g in $(pgrep -P "$c" 2>/dev/null); do kill -9 "$g" 2>/dev/null; done
            kill -9 "$c" 2>/dev/null
          done
          kill -9 "$p" 2>/dev/null
        fi
      done
    done
    sleep 3
    log "  stack restart #$stack_restarts"
    setsid nohup bash "$LAUNCH" > /root/orch_keepalive.out 2>&1 < /dev/null &
    sleep 30
    vllm_bad=0
    continue
  fi

  # awk, not pgrep: this script names the orchestrator script, so a pattern
  # match would find itself.
  if [ -n "$(ps -eo pid=,args= | awk '/cloud_orchestrator_v4\.sh [a-z0-9]/{print $1}')" ]; then
    continue
  fi

  still_open=$(still_open)
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
