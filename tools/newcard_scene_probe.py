#!/usr/bin/env python3
"""在卡上单独加载一个 ai2thor 场景并计时，用来判断"环境创建超时"是
场景本身太慢，还是并发抢资源。

    DISPLAY=:99 /home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python \
        /root/scene_probe.py FloorPlan315 900

输出一行：scene / 秒数 / OK 或 异常。
"""

from __future__ import annotations

import sys
import time

from ai2thor.controller import Controller


def main() -> None:
    scene = sys.argv[1] if len(sys.argv) > 1 else "FloorPlan315"
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 900.0
    t0 = time.time()
    try:
        c = Controller(scene=scene, width=800, height=600, gridSize=0.25,
                       visibilityDistance=1.0, server_timeout=timeout)
        dt = time.time() - t0
        print(f"{scene} {dt:.1f}s OK", flush=True)
        # 再走一步，确认渲染也活着
        ev = c.step("RotateRight")
        print(f"{scene} step_ok={bool(ev.metadata.get('lastActionSuccess'))}",
              flush=True)
        c.stop()
    except Exception as exc:                       # noqa: BLE001
        print(f"{scene} {time.time() - t0:.1f}s FAIL {type(exc).__name__}: {exc}",
              flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
