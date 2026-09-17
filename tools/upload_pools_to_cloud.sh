#!/bin/bash
# Upload the non-evaluation collection pools to the training box.
#
# The upload is resumable by file size, so it can be re-run while a collection
# is still adding episodes; already-transferred files are skipped.
#
# Usage:  bash tools/upload_pools_to_cloud.sh [--wait]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ "${1:-}" = "--wait" ]; then
  while pgrep -f 'collect_proctho[r].py' > /dev/null; do sleep 60; done
fi

setsid nohup python3 "$ROOT/tools/upload_pools.py" > /tmp/upload_pools.log 2>&1 < /dev/null &
sleep 10
echo -n "uploader 进程: "
pgrep -fc 'upload_pool[s].py' || true
