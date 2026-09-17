#!/bin/bash
# Pack the local non-eval pools and push them to the training box in one stream.
#
# Per-file SFTP runs at ~0.7 MB/s because every 200 KB PNG pays a round trip;
# a single tar stream reaches ~16 MB/s.  The upload is therefore: tar locally ->
# sftp one file -> untar remotely -> delete both tarballs.
#
# Usage:  bash tools/sync_pools_to_cloud.sh [--wait-for-collection]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAR=/mnt/d/lightwm_pools.tar
REMOTE_TAR=/root/autodl-tmp/lightwm_pools.tar

if [ "${1:-}" = "--wait-for-collection" ]; then
  while pgrep -f 'collect_proctho[r].py' > /dev/null || pgrep -f 'collect_coverag[e].py' > /dev/null; do
    sleep 60
  done
fi

echo "[sync] packing (cov2 + procthor2) ..."
tar cf "$TAR" -C /mnt/d lightwm_data_cov2 lightwm_data_procthor2
echo "[sync] packed: $(du -h "$TAR" | cut -f1)"

setsid nohup python3 "$ROOT/tools/push_tar.py" --tar "$TAR" --remote "$REMOTE_TAR" \
    > /tmp/sync_pools.log 2>&1 < /dev/null &
sleep 8
echo -n "[sync] uploader 进程: "
pgrep -fc 'push_ta[r].py' || true
