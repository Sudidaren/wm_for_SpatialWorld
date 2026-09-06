#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Safe AI2-THOR task launcher: process hygiene + cgroup memory guard + retry.
#
# Why: heavy tasks (world model + depth/seg rendering) can push the WSL VM to
# its memory ceiling; the kernel OOM-killer then kills init.scope and the whole
# WSL instance disappears (terminal flash-crash).  Running the task inside its
# own cgroup makes the kernel kill *only the task*, so WSL survives and the
# task can be retried.
#
# Usage (run from the SpatialWorld repo root):
#   scripts/ai2thor/safe_run.sh envs/ai2thor/.venv/bin/python -u \
#       envs/ai2thor/validate_occupancy.py
#   scripts/ai2thor/safe_run.sh envs/ai2thor/.venv/bin/python -u \
#       scripts/ai2thor/run_benchmark.py --config ... 
#
# Environment overrides:
#   SAFE_RUN_MEM_MAX    cgroup anon-memory ceiling    (default 7G)
#   SAFE_RUN_SWAP_MAX   cgroup swap ceiling           (default 3G)
#   SAFE_RUN_MAX_TRIES  retries after a task failure  (default 3)
#   SAFE_RUN_LOCK       lock dir for single-instance  (default /tmp/ai2thor_safe_run.lock)
# ---------------------------------------------------------------------------
set -u

MEM_MAX="${SAFE_RUN_MEM_MAX:-7G}"
SWAP_MAX="${SAFE_RUN_SWAP_MAX:-3G}"
MAX_TRIES="${SAFE_RUN_MAX_TRIES:-3}"
BACKOFF_BASE="${SAFE_RUN_BACKOFF_BASE:-5}"
LOCK_DIR="${SAFE_RUN_LOCK:-/tmp/ai2thor_safe_run.lock}"

log() { echo "[safe_run] $*"; }

# ---------------------------------------------------------------------------
# Single Unity instance lock (PID-aware so a hard crash cannot wedge it)
# ---------------------------------------------------------------------------
if mkdir "$LOCK_DIR" 2>/dev/null; then
  echo $$ > "$LOCK_DIR/pid"
  trap 'rm -f "$LOCK_DIR/pid"; rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
else
  old_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || echo 0)"
  if [ "$old_pid" -gt 0 ] 2>/dev/null && kill -0 "$old_pid" 2>/dev/null; then
    log "another task instance is already running (pid $old_pid); aborting"
    exit 3
  fi
  log "removing stale lock (pid $old_pid not alive)"
  rm -f "$LOCK_DIR/pid"
  rmdir "$LOCK_DIR" 2>/dev/null || true
  mkdir "$LOCK_DIR" || { log "cannot acquire lock at $LOCK_DIR"; exit 3; }
  echo $$ > "$LOCK_DIR/pid"
  trap 'rm -f "$LOCK_DIR/pid"; rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
fi

# ---------------------------------------------------------------------------
# Kill leftover simulator processes (the memory-peak source last time)
# ---------------------------------------------------------------------------
cleanup_stale() {
  log "cleaning stale simulator processes..."
  pkill -9 -f 'thor-' 2>/dev/null
  pkill -9 -f 'Xvfb' 2>/dev/null
  pkill -9 -f 'linux_exec' 2>/dev/null
  pkill -9 -f 'CarlaUE4' 2>/dev/null
  sleep 1
}

# ---------------------------------------------------------------------------
# Run once, inside a cgroup when possible
# ---------------------------------------------------------------------------
run_once() {
  local py="$1"
  shift
  local guard_level=0
  if command -v systemd-run >/dev/null 2>&1; then
    if systemctl --user show-environment >/dev/null 2>&1; then
      guard_level=1
    elif systemctl show-environment >/dev/null 2>&1; then
      guard_level=2
    fi
  fi
  if [ "$guard_level" -eq 1 ]; then
    log "launching under user cgroup: MemoryMax=$MEM_MAX MemorySwapMax=$SWAP_MAX"
    systemd-run --user --scope --quiet \
      -p "MemoryMax=$MEM_MAX" -p "MemorySwapMax=$SWAP_MAX" \
      -- "$py" "$@"
    return $?
  fi
  if [ "$guard_level" -eq 2 ]; then
    log "launching under system cgroup: MemoryMax=$MEM_MAX MemorySwapMax=$SWAP_MAX"
    systemd-run --scope --quiet \
      -p "MemoryMax=$MEM_MAX" -p "MemorySwapMax=$SWAP_MAX" \
      -- "$py" "$@"
    return $?
  fi
  log "systemd bus unavailable; running directly (no memory guard)"
  "$py" "$@"
  return $?
}

if [ $# -lt 2 ]; then
  echo "usage: $0 <python> <script.py> [args...]" >&2
  exit 2
fi

PY="$1"
shift
ARGS=("$@")

cleanup_stale

attempt=1
while :; do
  log "attempt $attempt/$MAX_TRIES: $PY ${ARGS[*]}"
  run_once "$PY" "${ARGS[@]}"
  code=$?
  if [ "$code" -eq 0 ]; then
    log "task finished OK"
    exit 0
  fi
  if [ "$attempt" -ge "$MAX_TRIES" ]; then
    log "task FAILED after $MAX_TRIES attempts (last exit=$code)"
    exit "$code"
  fi
  wait_sec=$((BACKOFF_BASE * attempt * attempt))
  log "task exit=$code; retrying in ${wait_sec}s (exponential backoff)"
  cleanup_stale
  sleep "$wait_sec"
  attempt=$((attempt + 1))
done
