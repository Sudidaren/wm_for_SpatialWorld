#!/bin/bash
# Bring up a working :99 display on a card.
# Note: xdpyinfo comes from x11-utils -- without it every display check reads
# "BAD" even when Xvfb is fine.  That cost a round trip today.
set -u
export DISPLAY=:99
mkdir -p /root/autodl-tmp/logs
apt-get install -y -qq xvfb x11-utils >/dev/null 2>&1
echo "Xvfb: $(command -v Xvfb || echo missing)   xdpyinfo: $(command -v xdpyinfo || echo missing)"
if [ ! -e /tmp/.X11-unix/X99 ]; then
    setsid nohup Xvfb :99 -screen 0 1280x1024x24 \
        > /root/autodl-tmp/logs/xvfb.log 2>&1 < /dev/null &
    for _ in $(seq 1 15); do
        [ -e /tmp/.X11-unix/X99 ] && break
        sleep 1
    done
fi
echo "sockets: $(ls /tmp/.X11-unix/ 2>/dev/null | tr '\n' ' ')"
if DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then
    echo "DISPLAY :99 OK  ($(DISPLAY=:99 xdpyinfo | grep -m1 dimensions))"
else
    echo "DISPLAY :99 BAD"
    tail -5 /root/autodl-tmp/logs/xvfb.log
fi
