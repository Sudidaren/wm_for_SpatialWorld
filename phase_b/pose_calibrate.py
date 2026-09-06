"""Offline calibration of move/rotation magnitudes for dead reckoning.

Replays navigation actions in AI2-THOR and measures the *actual* deltas the
simulator executed (sim pose is used ONLY as an offline measurement ruler;
the runtime never reads it). Outputs the systematic bias to fold into
move_magnitudes / rotation constants.

Usage (GPU not required, simulator needed):
  python phase_b/pose_calibrate.py [--scene FloorPlan1]
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="FloorPlan1")
    args = ap.parse_args()

    import ai2thor.controller
    c = ai2thor.controller.Controller(
        scene=args.scene, width=800, height=600,
        visibilityDistance=20, gridSize=0.25)

    def pose():
        a = c.last_event.metadata["agent"]
        return (a["position"]["x"], a["position"]["z"],
                float(a["rotation"]["y"]))

    results = {"MoveAhead": {}, "MoveBack": {}, "MoveLeft": {},
               "MoveRight": {}, "RotateLeft": {}, "RotateRight": {}}

    def run_move(name, mag):
        c.step(dict(action="Teleport", position={"x": 0.0, "y": 0.0,
                                                 "z": 0.0},
                    rotation={"x": 0, "y": 0, "z": 0}, horizon=0,
                    standing=True))
        c.step(dict(action="RotateRight", degrees=90))
        x0, z0, _ = pose()
        ev = c.step(dict(action=name, moveMagnitude=mag))
        if not ev.metadata.get("lastActionSuccess"):
            return None
        x1, z1, _ = pose()
        return math.hypot(x1 - x0, z1 - z0)

    def run_rot(name):
        c.step(dict(action="Teleport", position={"x": 0.0, "y": 0.0,
                                                 "z": 0.0},
                    rotation={"x": 0, "y": 0, "z": 0}, horizon=0,
                    standing=True))
        _, _, y0 = pose()
        ev = c.step(dict(action=name, degrees=90))
        if not ev.metadata.get("lastActionSuccess"):
            return None
        _, _, y1 = pose()
        d = (y1 - y0) % 360
        return d if d <= 180 else d - 360

    for name in ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight"):
        for mag in (0.25, 0.5, 1.0):
            vals = [run_move(name, mag) for _ in range(3)]
            vals = [v for v in vals if v is not None]
            if vals:
                results[name][mag] = (float(np.mean(vals)),
                                      float(np.std(vals)))
    for name in ("RotateLeft", "RotateRight"):
        vals = [run_rot(name) for _ in range(3)]
        vals = [v for v in vals if v is not None]
        if vals:
            results[name][90.0] = (float(np.mean(vals)),
                                   float(np.std(vals)))

    c.stop()
    print("calibration results (actual displacement / rotation):")
    for k, v in results.items():
        for mag, (mean, std) in sorted(v.items()):
            print(f"  {k} request={mag}: mean={mean:.4f} std={std:.4f}")
    np.save("/tmp/pose_calibration.npy", results, allow_pickle=True)


if __name__ == "__main__":
    main()
