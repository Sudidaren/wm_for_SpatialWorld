#!/usr/bin/env bash
# Apply LightWM runtime overlay onto an official SpatialWorld checkout
# (base commit f47b1e0) and verify byte-for-byte equality via sha256.
set -euo pipefail

usage() { echo "usage: $0 --repo <spatialworld_dir> --overlay <overlay_dir>"; exit 1; }

REPO=""; OVERLAY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --overlay) OVERLAY="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$REPO" ] && [ -n "$OVERLAY" ] || usage
REPO="$(cd "$REPO" && pwd)"
OVERLAY="$(cd "$OVERLAY" && pwd)"

cd "$REPO"
if [ "$(git rev-parse HEAD 2>/dev/null)" != "f47b1e091985a7f141db8d93c6822b42663cc1da" ]; then
  echo "!! SpatialWorld HEAD is not official f47b1e0; overlay is diffed against it."
  echo "   run: git fetch origin && git checkout f47b1e0"
  exit 1
fi

echo "== applying overlay =="
for d in mllm_base_agent experiments evaluation envs scripts; do
  if [ -d "$OVERLAY/$d" ]; then
    cp -r "$OVERLAY/$d"/. "$REPO/$d"/
  fi
done
if [ -f "$OVERLAY/.gitignore" ]; then
  cp "$OVERLAY/.gitignore" "$REPO/.gitignore"
fi

echo "== verifying sha256 (41 files) =="
sha256sum -c "$OVERLAY/MANIFEST.sha256"

echo "OK: runtime files identical to LightWM local (base f47b1e0)."
