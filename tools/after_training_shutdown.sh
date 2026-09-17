#!/bin/bash
# Watch the cloud box: when training is finished, stop every job we own and
# power the instance off, so an idle GPU never keeps billing.
#
# Install with:  setsid nohup bash tools/after_training_shutdown.sh >/dev/null 2>&1 &
# Progress is appended to /root/autoshutdown.log.
#
# The AutoDL container holds no CAP_SYS_BOOT, so in-container shutdown may be
# refused by the platform.  The script always tries, and records the outcome so
# a refused shutdown is visible instead of silently leaving the box idle.
set -u

TRAIN_PATTERN="${TRAIN_PATTERN:-train_depth_v2.py}"
WAIT_STEPS="${WAIT_STEPS:-960}"   # 960 x 30 s = 8 h cap
LOG="${LOG:-/root/autoshutdown.log}"

exec >>"$LOG" 2>&1
echo "==== $(date -Is) watcher start (pattern=$TRAIN_PATTERN) ===="

for _ in $(seq 1 "$WAIT_STEPS"); do
  pgrep -f "[t]rain_depth_v2.py" >/dev/null || break
  sleep 30
done
echo "$(date -Is) trainer finished; final metrics:"
grep -E '^epoch|new best' /root/train_depth_v2.log 2>/dev/null | tail -12

# Stop everything we own; leave nothing spinning on the GPU/CPU.
pkill -f '[r]eap_strays.py' 2>/dev/null
pkill -f '[v]llm serve' 2>/dev/null
pkill -f '[V]LLM::EngineCore' 2>/dev/null
ps -eo pid,args | grep -E '[r]un_task|[c]ollect_|[o]rchestrator|[c]hain|[r]un_ablation' \
  | awk '{print $1}' | xargs -r kill -TERM 2>/dev/null
sleep 5
ps -eo pid,args | grep '[r]un_task' | awk '{print $1}' | xargs -r kill -KILL 2>/dev/null
sync
echo "$(date -Is) our processes stopped"

for cmd in "shutdown -h now" "poweroff -f" "halt -f"; do
  echo "$(date -Is) try: $cmd"
  $cmd
  echo "  rc=$?  $(date -Is)"
  sleep 5
done

echo "$(date -Is) INSTANCE_STILL_UP -- the platform refused in-container power-off;"
echo "  power it off from the AutoDL console (/panel or the web page)."
