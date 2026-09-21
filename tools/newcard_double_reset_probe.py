#!/usr/bin/env python3
"""复现 harness 的调用序列：同一个 Controller 连续 reset 两次同一个场景。

`scripts/ai2thor/work/run_task.py` 是这么写的：

    observation = env.reset(task_description, scene=task_scene)      # (1)
    print("📁 Loading init from task folder: ...")
    init_data, init_scene = load_init_actions_from_folder(...)
    if init_scene:
        observation = env.reset(task_description, scene=task_scene)  # (2)

而每个任务目录里的 init.json 都带 "scene" 字段，所以每个任务都会走 (2)。
卡上观测到 worker 的日志正好停在 (1) 之后的那个 print 上 —— 也就是说卡在 (2)。

    python3 newcard_double_reset_probe.py FloorPlan14 300
"""

from __future__ import annotations

import sys
import time

from ai2thor.controller import Controller


def main() -> None:
    scene = sys.argv[1] if len(sys.argv) > 1 else "FloorPlan14"
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    t0 = time.time()
    c = Controller(scene=scene, width=800, height=600, gridSize=0.25,
                   visibilityDistance=1.0, server_timeout=timeout)
    print(f"controller up {time.time() - t0:.1f}s", flush=True)
    t1 = time.time()
    c.reset(scene)
    print(f"reset #1 {time.time() - t1:.1f}s", flush=True)
    t2 = time.time()
    c.reset(scene)
    print(f"reset #2 {time.time() - t2:.1f}s  total {time.time() - t0:.1f}s",
          flush=True)
    c.stop()


if __name__ == "__main__":
    main()
