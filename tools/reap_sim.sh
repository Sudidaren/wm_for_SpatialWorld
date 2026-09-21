#!/usr/bin/env bash
# Kill leftover simulator processes on this machine, by PID.
#
# Why this is a file and not a one-liner: a shell command that mentions the
# process name it is hunting matches its own command line, so `pgrep -f ...`
# finds the shell that is running it and the kill loop terminates the caller
# (this bit the session four times in one afternoon).  A script's command line
# is just "bash reap_sim.sh", so nothing here can match itself.
#
#   bash tools/reap_sim.sh            # kill leftover Unity
#   bash tools/reap_sim.sh --workers  # also kill orphaned run_task workers
set -u

me=$$
killed=0

kill_matching() {
  local pattern="$1" label="$2" p
  for p in $(pgrep -f "$pattern" 2>/dev/null); do
    [ "$p" = "$me" ] && continue
    # never kill our own ancestors (the shell that launched this script)
    if [ "$p" = "$PPID" ]; then continue; fi
    kill -9 "$p" 2>/dev/null && { echo "  killed $label pid=$p"; killed=$((killed + 1)); }
  done
}

kill_matching "releases/thor-Linux64" "unity"
kill_matching "releases/thor-CloudRendering" "unity-cloud"

if [ "${1:-}" = "--workers" ]; then
  for p in $(pgrep -f "scripts.ai2thor.work.run_task" 2>/dev/null); do
    [ "$p" = "$me" ] && continue
    ppid=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)
    if [ -z "$ppid" ] || [ "$ppid" = "1" ]; then
      kill -9 "$p" 2>/dev/null && { echo "  killed orphan worker pid=$p"; killed=$((killed + 1)); }
    fi
  done
fi

# --run NAME: stop one evaluation arm (its supervisor plus everything under
# it).  Matched through /proc/<pid>/environ, so it finds exactly the arm and
# never a shell that merely mentions the name.
if [ "${1:-}" = "--run" ] && [ -n "${2:-}" ]; then
  run_name="$2"
  for p in $(pgrep -f "python -u" 2>/dev/null); do
    [ "$p" = "$me" ] && continue
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "RUN_NAME=$run_name"; then
      echo "  stopping arm $run_name (supervisor pid=$p)"
      for c in $(pgrep -P "$p" 2>/dev/null); do
        for g in $(pgrep -P "$c" 2>/dev/null); do kill -9 "$g" 2>/dev/null; done
        kill -9 "$c" 2>/dev/null
      done
      kill -9 "$p" 2>/dev/null
      killed=$((killed + 1))
    fi
  done
fi

echo "reaped $killed process(es)"
