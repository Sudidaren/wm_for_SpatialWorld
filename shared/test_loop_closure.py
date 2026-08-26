"""Deterministic unit test for the loop-closure CORRECTION MATH.

The fingerprint-detection side is covered by eval_loop_closure.py on real
episodes.  Here we verify the invariant that matters:

    after a confident revisit, world_pose(odom_at_revisit) == matched place

with a fully controlled state (no image noise, no drift randomness).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from loop_closure import LoopClosure, PlaceNode  # noqa: E402
from spatial_memory import SpatialMemory  # noqa: E402


def test_correction_math():
    lc = LoopClosure(SpatialMemory(), device="cpu")
    lc.places.append(PlaceNode(np.array([1.0, 0.0]), np.zeros(2), 0.0,
                               np.zeros(2), 0.0, 0))
    lc.places.append(PlaceNode(np.array([0.0, 1.0]), np.array([0.0, 2.0]),
                               0.0, np.array([0.0, 2.0]), 0.0, 5))

    odom_pos = {"x": 0.3, "z": 2.4, "y": 0.9}
    before = lc.world_pose(odom_pos, 0.0)[0]
    assert lc.n_closure == 0
    lc._close_loop(np.array([1.0, 0.0]), odom_pos, 0.0, 12)
    assert lc.n_closure == 1, "expected one closure"
    after = lc.world_pose(odom_pos, 0.0)[0]
    print(f"world pose before closure: ({before['x']:.2f}, {before['z']:.2f})")
    print(f"world pose after closure:  ({after['x']:.2f}, {after['z']:.2f}) "
          f"(should be the start place (0, 0))")
    assert abs(after["x"]) < 1e-6 and abs(after["z"]) < 1e-6, \
        "after closure the current pose must map to the matched place"
    n_before = lc.n_closure
    lc._close_loop(np.array([0.0, 1.0]), {"x": 0.4, "z": 2.5, "y": 0.9},
                   0.0, 13)
    assert lc.n_closure == n_before, "low-sim frame must not fire a closure"
    print("OK")


if __name__ == "__main__":
    test_correction_math()
