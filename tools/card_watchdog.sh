#!/usr/bin/env bash
# Process hygiene for an evaluation card.  Run this *next to* the arm queue.
#
# Why it exists: the previous batch died because workers from earlier attempts
# stayed around -- zombies plus orphaned ai2thor/Unity children piled up until
# the box ran out of memory and the supervisor could not start anything.  The
# rule this script enforces is simple: at any moment, the only evaluation
# processes on the box are the ones whose environment says they belong to the
# arm that is currently supposed to be running.
#
# It never kills anything by name pattern -- only by PID, after checking the
# process's own environment (`/proc/<pid>/environ`), which is why it cannot
# take out the watchdog itself or an unrelated job.
#
#   RUN_NAME=<arm> bash tools/card_watchdog.sh &
#
# Log: /root/autodl-tmp/logs/watchdog.log

set -u
RUN_NAME="${RUN_NAME:-}"
INTERVAL="${INTERVAL:-60}"
LOG="${LOG:-/root/autodl-tmp/logs/watchdog.log}"
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

# A pidfile, because every pgrep pattern for "this script" also matches the
# command line of whoever is grepping -- that has silently disabled checks
# three times today.  A pid is unambiguous.
PIDFILE="${PIDFILE:-/root/watchdog.pid}"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log "watchdog start (arm=${RUN_NAME:-<none>} interval=${INTERVAL}s pid=$$)"

prev_zombie=0
while true; do
  # -- X11: the harness runs ai2thor on the Linux64 platform, which needs a
  # live display.  If Xvfb dies, every task fails as "failed_external"
  # (measured: 160/160 before this check existed), so keep it alive here.
  if ! pgrep -f "[X]vfb ${DISPLAY:-:99}" >/dev/null 2>&1; then
    log "Xvfb ${DISPLAY:-:99} not running -> restart"
    setsid nohup Xvfb "${DISPLAY:-:99}" -screen 0 1280x1024x24 \
      >> /root/autodl-tmp/logs/xvfb.log 2>&1 < /dev/null &
    sleep 3
    pgrep -f "[X]vfb ${DISPLAY:-:99}" >/dev/null && log "  xvfb back up" \
      || log "  xvfb FAILED to restart"
  fi

  # -- zombies: count, and never touch their parent ----------------------
  # 2026-09-19, on card D: this block used to SIGKILL "the parent of a zombie
  # if its environment says RUN_NAME", and at 12:40:16 it found two zombies and
  # killed the supervisor of a live 311-task arm.  Two zombies cost nothing --
  # they are dead processes whose exit status has not been collected yet, and
  # the python parent collects them on its next wait().  What actually eats a
  # box is an orphaned Unity build, which is reaped further down by PID.
  # So: report, never signal.  (A zombie's parent that is *itself* dead is
  # handled by the orphan rules below.)
  zombies=$(ps -eo stat= | grep -c '^Z' || true)
  if [ "$zombies" -ge 50 ]; then
    log "WARNING: $zombies zombies -- parents: $(ps -eo ppid=,stat= | awk '$2 ~ /^Z/ {print $1}' | sort -u | tr '\n' ' ')"
  fi

  # -- orphaned workers: a run_task whose supervisor is gone ---------------
  # Its result can never be collected, and it keeps a simulator alive, so it
  # is dead work either way.  (The supervisor runs it as a direct child, so
  # ppid 1 here means the supervisor exited.)
  for p in $(pgrep -f "[r]un_task" 2>/dev/null); do
    ppid=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)
    if [ -z "$ppid" ] || [ "$ppid" = "1" ]; then
      log "orphaned worker pid=$p (ppid=${ppid:-?}) -> kill"
      kill -9 "$p" 2>/dev/null
    fi
  done

  # -- strays: our evaluation processes that are NOT the current arm ------
  for p in $(pgrep -f "[p]ython -u -" 2>/dev/null); do
    env_run=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^RUN_NAME=//p')
    [ -z "$env_run" ] && continue                  # not ours
    if [ -n "$RUN_NAME" ] && [ "$env_run" != "$RUN_NAME" ]; then
      log "stray arm '$env_run' (pid=$p) -> kill"
      for c in $(pgrep -P "$p" 2>/dev/null); do
        for g in $(pgrep -P "$c" 2>/dev/null); do kill -9 "$g" 2>/dev/null; done
        kill -9 "$c" 2>/dev/null
      done
      kill -9 "$p" 2>/dev/null
    fi
  done

  # -- ai2thor / Unity children with no live python parent ----------------
  # (these are the ones that eat tens of GB when orphaned)
  for p in $(pgrep -f "[t]hor-CloudRendering|[t]hor-Linux64" 2>/dev/null); do
    ppid=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)
    if [ -z "$ppid" ] || [ "$ppid" = "1" ] || ! kill -0 "$ppid" 2>/dev/null; then
      log "orphaned simulator pid=$p (ppid=$ppid) -> kill"
      kill -9 "$p" 2>/dev/null
    fi
  done

  # -- resources ----------------------------------------------------------
  mem_avail=$(free -g | awk '/^Mem:/{print $7}')
  gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  log "ok zombies=$zombies mem_avail=${mem_avail}G gpu_used=${gpu_mem:-?}M procs=$(pgrep -fc '[p]ython -u -' || true)"
  if [ "$mem_avail" -lt 20 ]; then
    log "WARNING: available RAM ${mem_avail}G"
  fi
  # -- memory floor -------------------------------------------------------
  # The box dying is worse than a retried task: an OOM kill takes the
  # supervisor with it and leaves orphans holding simulators.  Below the
  # floor, drop the *newest* worker (the one that has done the least), whose
  # task the supervisor records as an external failure and retries.  One per
  # cycle, so the floor is approached slowly and never as a cull.
  if [ "$mem_avail" -lt 8 ]; then
    # newest = smallest elapsed time
    victim=$(for p in $(pgrep -f "[r]un_task" 2>/dev/null); do
               et=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')
               [ -n "$et" ] && echo "$et $p"
             done | sort -n | head -1 | awk '{print $2}')
    if [ -n "$victim" ]; then
      log "EMERGENCY mem_avail=${mem_avail}G -> stop newest worker pid=$victim"
      for c in $(pgrep -P "$victim" 2>/dev/null); do kill -9 "$c" 2>/dev/null; done
      kill -9 "$victim" 2>/dev/null
      sleep 5
      log "  after kill: mem_avail=$(free -g | awk '/^Mem:/{print $7}')G"
    fi
  fi
  prev_zombie=$zombies
  sleep "$INTERVAL"
done
