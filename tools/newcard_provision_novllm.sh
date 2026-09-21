#!/bin/bash
# Provision a card that runs an API model (GPT-5) instead of a local vLLM:
# Xvfb, eval python deps, the eval venv, and the ai2thor smoke test.
# No vLLM, no model weights -- GPT-5 is reached over HTTPS, and the only
# thing on the GPU is the WM's perception head (RF-DETR + monocular depth).
# ASCII only: this travels through paramiko.
set -u
LOG=/root/autodl-tmp/provision.log
exec >>"$LOG" 2>&1
PIP=/root/miniconda3/bin/pip
PY=/root/miniconda3/bin/python
VENV=/home/sudidaren/SpatialWorld/envs/ai2thor/.venv
TS=https://pypi.tuna.tsinghua.edu.cn/simple
echo "================ provision (no vllm) start $(date '+%F %T')"
mkdir -p /root/autodl-tmp/logs

echo "--- [1/5] Xvfb + xdpyinfo"
apt-get install -y -qq xvfb x11-utils >/dev/null 2>&1
export DISPLAY="${DISPLAY:-:99}"
if [ ! -e /tmp/.X11-unix/X99 ]; then
    setsid nohup Xvfb :99 -screen 0 1280x1024x24 \
        > /root/autodl-tmp/logs/xvfb.log 2>&1 < /dev/null &
    for _ in $(seq 1 15); do [ -e /tmp/.X11-unix/X99 ] && break; sleep 1; done
fi
DISPLAY=:99 xdpyinfo >/dev/null 2>&1 \
    && echo "  display OK ($(DISPLAY=:99 xdpyinfo | grep -m1 dimensions))" \
    || { echo "  display BAD"; tail -5 /root/autodl-tmp/logs/xvfb.log; }

echo "--- [2/5] eval python deps (conda base, tsinghua mirror)"
$PIP install -q -i "$TS" ai2thor==5.0.0 rfdetr supervision \
    opencv-python-headless transformers openai python-dotenv pyyaml paramiko \
    2>&1 | tail -3

echo "--- [3/5] eval venv (--system-site-packages)"
[ -x "$VENV/bin/python" ] || $PY -m venv --system-site-packages "$VENV"
$VENV/bin/python -c "import ai2thor, rfdetr, supervision, cv2, transformers, openai; \
print('  eval deps OK, ai2thor', ai2thor.__version__)"

echo "--- [4/5] gateway reachability (GPT-5 lives at the far end)"
for label in direct turbo; do
    if [ "$label" = turbo ]; then source /etc/network_turbo >/dev/null 2>&1; fi
    code=$(timeout 25 curl -s -o /dev/null -w '%{http_code}' \
        https://apic1.ohmycdn.com/v1/models 2>/dev/null)
    echo "  $label: http=$code"
done

echo "--- [5/5] ai2thor smoke (first run fetches the Unity build)"
$VENV/bin/python - <<'PY'
import time
from ai2thor.controller import Controller
t0 = time.time()
c = Controller(scene="FloorPlan1", platform="Linux64", width=300, height=300,
               server_timeout=900.0, start_unity_process=True)
print("  controller up in %.0fs" % (time.time() - t0))
print("  frame", c.step("Pass").frame.shape)
c.stop()
PY
echo "================ provision (no vllm) done $(date '+%F %T')"
