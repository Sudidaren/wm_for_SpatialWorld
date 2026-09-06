#!/usr/bin/env python3
"""Quick probe: what moves succeed from the start pose in FloorPlan17."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ai2thor.controller import Controller


def pose(meta):
    a = meta["agent"]["position"]
    r = meta["agent"]["rotation"]
    return a["x"], a["z"], r["y"]


def main():
    c = Controller(
        scene="FloorPlan17",
        width=256,
        height=192,
        gridSize=0.25,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )
    c.reset(scene="FloorPlan17")
    print("after reset:", pose(c.last_event.metadata))
    c.step(action="LookDown", degrees=30)
    print("after LookDown:", pose(c.last_event.metadata))

    for mag in (0.25, 1.0):
        for action in ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight"):
            ev = c.step(action=action, moveMagnitude=mag)
            ok = ev.metadata["lastActionSuccess"]
            print(f"yaw0 {action}({mag}): ok={ok} -> {pose(ev.metadata)}")
            # step back to origin
            rev = {
                "MoveAhead": "MoveBack",
                "MoveBack": "MoveAhead",
                "MoveLeft": "MoveRight",
                "MoveRight": "MoveLeft",
            }[action]
            ev2 = c.step(action=rev, moveMagnitude=mag)
            print(f"   return {rev}: ok={ev2.metadata['lastActionSuccess']} -> {pose(ev2.metadata)}")

    # replicate golden: RotateRight(90) then MoveRight(1)
    c.step(action="RotateRight", degrees=90)
    ev = c.step(action="MoveRight", moveMagnitude=1)
    print("golden path RotateRight90 + MoveRight(1):", ev.metadata["lastActionSuccess"], "->", pose(ev.metadata))
    c.step(action="MoveRight", moveMagnitude=0.25)
    c.step(action="MoveAhead", moveMagnitude=0.5)
    print("after golden 4 moves:", pose(c.last_event.metadata))
    ev = c.step(action="MoveRight", moveMagnitude=0.5)
    print("golden MoveRight(0.5):", ev.metadata["lastActionSuccess"], "->", pose(ev.metadata))
    c.stop()


if __name__ == "__main__":
    main()
