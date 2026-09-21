#!/bin/bash
# A stale ai2thor smoke test can sit in Build.download() holding the
# releases/*.lock file for hours (it downloads the Unity build at ~55 KB/s
# when the build directory is not yet valid).  Every worker then blocks on
# that fcntl lock and the card looks "stuck with 0 frames".
#
# This kills the stale holders + workers, clears the locks, and leaves the
# (already uploaded) build in place so the next Controller skips the download.
set -u
echo "--- stale long-running python (the lock holder) ---"
ps -eo pid,etimes,args --no-headers \
    | awk '$2 > 600 && /\.venv\/bin\/python -$|miniconda3\/bin\/python -$/ {print $1, $2}' \
    | while read -r pid secs; do echo "  kill $pid (${secs}s)"; kill -9 "$pid" 2>/dev/null; done
echo "--- workers / thor ---"
pgrep -f '[w]ork.run_task' | xargs -r kill -9
pgrep -f '[t]hor-Linux64' | xargs -r kill -9
sleep 3
echo "--- clear the build locks ---"
rm -f /root/.ai2thor/tmp/*.lock /root/.ai2thor/releases/*.lock
ls /root/.ai2thor/tmp/ 2>/dev/null | wc -l | xargs echo "  tmp 剩余条目:"
echo "--- build present? ---"
ls -d /root/.ai2thor/releases/thor-Linux64-*/ 2>/dev/null
echo -n "  可执行文件: "
ls /root/.ai2thor/releases/thor-Linux64-*/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917 >/dev/null 2>&1 && echo yes || echo NO
echo "--- 现在起一个 smoke 验证（应在 ~20s 起来）---"
export DISPLAY=:99
timeout 180 /home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python - <<'PY'
import time
from ai2thor.controller import Controller
t0 = time.time()
c = Controller(scene="FloorPlan1", platform="Linux64", width=300, height=300,
               server_timeout=120.0, start_unity_process=True)
print("  controller up in %.0fs" % (time.time() - t0))
print("  frame", c.step("Pass").frame.shape)
c.stop()
PY
