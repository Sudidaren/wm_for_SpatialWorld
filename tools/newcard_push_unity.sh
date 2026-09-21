#!/bin/bash
# Upload the AI2-THOR Linux64 Unity build from this box to a card.
# The card's own downloader pulls it at ~55 KB/s (≈4 h); from here it is
# ~4 min.  Same commit hash on both sides, so this is the identical build.
#
#   bash newcard_push_unity.sh <card>
set -u
CARD="$1"
TGZ=/mnt/d/eval_card_payload/thor_linux64_build.tar.gz
N="python3 /home/sudidaren/lightwm_phases/tools/newcard.py --card $CARD"

echo "[unity] $CARD: upload $(du -h $TGZ | cut -f1)"
$N put "$TGZ" /root/thor.tgz || { echo "[unity] $CARD: upload FAILED"; exit 1; }
echo "[unity] $CARD: unpack"
$N run "mkdir -p /root/.ai2thor/releases && tar xzf /root/thor.tgz -C /root/.ai2thor/releases && rm -f /root/thor.tgz && du -sh /root/.ai2thor/releases/*/ && ls /root/.ai2thor/releases/ | head -3"
