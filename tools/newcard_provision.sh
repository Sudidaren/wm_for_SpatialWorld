#!/bin/bash
# Provision one card: Xvfb, python deps, the eval venv, vLLM, then the ai2thor
# smoke test (which downloads the ~2 GB Unity build on first use).
# ASCII only -- this travels through paramiko.
#
#   bash newcard_provision.sh
set -u
LOG=/root/autodl-tmp/provision.log
exec >>"$LOG" 2>&1
PIP=/root/miniconda3/bin/pip
PY=/root/miniconda3/bin/python
VENV=/home/sudidaren/SpatialWorld/envs/ai2thor/.venv
TS=https://pypi.tuna.tsinghua.edu.cn/simple
echo "================ provision start $(date '+%F %T')"
mkdir -p /root/autodl-tmp/logs

echo "--- [1/5] Xvfb"
export DISPLAY="${DISPLAY:-:99}"
if ! xdpyinfo >/dev/null 2>&1; then
    setsid nohup Xvfb "$DISPLAY" -screen 0 1280x1024x24 \
        > /root/autodl-tmp/logs/xvfb.log 2>&1 < /dev/null &
    for _ in $(seq 1 20); do xdpyinfo >/dev/null 2>&1 && break; sleep 1; done
fi
if xdpyinfo >/dev/null 2>&1; then echo "  display OK on $DISPLAY"
else echo "  display BAD"; tail -5 /root/autodl-tmp/logs/xvfb.log; fi

echo "--- [2/5] eval python deps (conda base, tsinghua mirror)"
$PIP install -q -i "$TS" ai2thor==5.0.0 rfdetr supervision \
    opencv-python-headless transformers openai python-dotenv pyyaml paramiko \
    2>&1 | tail -3

echo "--- [3/5] eval venv (--system-site-packages so it sees the above)"
[ -x "$VENV/bin/python" ] || $PY -m venv --system-site-packages "$VENV"
$VENV/bin/python -c "import ai2thor, rfdetr, supervision, cv2, transformers, openai; \
print('  eval deps OK, ai2thor', ai2thor.__version__)"

echo "--- [4/5] vLLM"
$PIP install -q -i "$TS" vllm 2>&1 | tail -3
$PIP show vllm 2>/dev/null | head -2
$PY -c "import vllm, torch; print('  vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)" 2>&1 | tail -2

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
echo "================ provision done $(date '+%F %T')"
