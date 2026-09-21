#!/bin/bash
# Upload the WM code payload to one new card and unpack it into /home/sudidaren,
# which is the layout the eval scripts expect (same paths as the local box).
#
#   bash newcard_push_payload.sh <30b|8b|kimi>
set -u
CARD="$1"
P=/mnt/d/eval_card_payload
N="python3 /home/sudidaren/lightwm_phases/tools/newcard.py --card $CARD"

echo "[push] $CARD: mkdir + upload"
$N run "mkdir -p /home/sudidaren /root/autodl-tmp/logs" >/dev/null
for t in lightwm_phases.tar spatialworld.tar spatialworld_eval.tar; do
    echo "[push] $CARD: $t ($(du -h $P/$t | cut -f1))"
    $N put "$P/$t" "/root/$t" || { echo "[push] $CARD: $t FAILED"; exit 1; }
done

echo "[push] $CARD: unpack"
$N run "cd /home/sudidaren && for t in lightwm_phases spatialworld spatialworld_eval; do tar xf /root/\$t.tar; done; rm -f /root/*.tar; ls /home/sudidaren | tr '\n' ' '; echo; du -sh /home/sudidaren/lightwm_phases /home/sudidaren/SpatialWorld /home/sudidaren/spatialworld_eval"
