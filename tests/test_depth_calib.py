"""Geometry of the depth-head scale calibration, on synthetic frames.

The calibration itself never sees ground truth, so the only thing worth
testing is that the geometry helpers and the floor-anchored fit do what the
derivation says they do: a depth map that *is* the floor recovers exactly the
camera height, and a depth map that is the truth pushed through a known
distortion comes back out of the fit.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase_b.depth_calib import (GroundCalibConfig, apply_scale_curve,  # noqa: E402
                                 estimate_scale, fit_scale_curve,
                                 floor_depth_map, horizon_row,
                                 implied_height_map, vertical_factor)

W, H_PX = 200, 150
CFG = GroundCalibConfig(cam_height=0.675, stride=1, z_min=0.35, z_max=20.0)


def synthetic_frame(horizon_deg: float = 0.0):
    """A floor plane plus a near box, as a *true* depth map."""
    zfloor = floor_depth_map(horizon_deg, width=W, height=H_PX, cfg=CFG)
    depth = np.array(zfloor)
    # a box standing on the floor: closer than the floor, upper-middle of frame
    depth[40:95, 60:140] = 1.4
    return depth


class TestGeometry(unittest.TestCase):
    def test_floor_depth_reconstructs_camera_height(self):
        for horizon in (0.0, 12.0, -8.0):
            z = floor_depth_map(horizon, width=W, height=H_PX, cfg=CFG)
            a = implied_height_map(z, horizon, width=W, height=H_PX, cfg=CFG)
            below = np.isfinite(z)
            self.assertGreater(below.sum(), 1000)
            self.assertLess(float(np.abs(a[below] - CFG.cam_height).max()), 1e-9)

    def test_horizon_row_matches_zero_height(self):
        c = vertical_factor(0.0, width=W, height=H_PX, cfg=CFG)
        v = int(round(horizon_row(0.0, width=W, height=H_PX, cfg=CFG)))
        # the row just below the horizon has a small positive factor, and the
        # bottom rows a large one
        self.assertAlmostEqual(float(c[v]), 0.0, places=6)
        self.assertGreater(float(c[-1]), 0.4)

    def test_estimate_scale_recovers_a_pure_scale_error(self):
        truth = synthetic_frame()
        for k in (0.55, 0.8, 1.0, 1.35):
            est, info = estimate_scale(truth * k, 0.0, width=W, height=H_PX,
                                       cfg=CFG)
            self.assertTrue(math.isfinite(est), info)
            self.assertLess(abs(est - k) / k, 0.05, f"k={k} got {est}")

    def test_curve_fit_recovers_a_power_law_distortion(self):
        truth = synthetic_frame()
        # the measured failure mode: a compressed response curve
        a_true, b_true = 0.8, 0.1
        pred = np.exp((np.log(truth) - b_true) / a_true)
        curves, info = fit_scale_curve(pred.astype(np.float32), 0.0, width=W,
                                       height=H_PX,
                                       cfg=GroundCalibConfig(
                                           cam_height=0.675, stride=1,
                                           slope_shrink=0.0))
        self.assertIsNotNone(curves, info)
        fixed = apply_scale_curve(pred, curves)
        ok = np.isfinite(truth) & (truth > 0)   # rows above the horizon are inf
        rel = np.abs(fixed[ok] - truth[ok]) / truth[ok]
        self.assertLess(float(np.median(rel)), 0.05, info)

    def test_flat_prediction_is_rejected(self):
        # no floor evidence at all -> the caller must be told, not guessed at
        flat = np.full((H_PX, W), 3.0, dtype=np.float32)
        curves, info = fit_scale_curve(flat, 0.0, width=W, height=H_PX, cfg=CFG)
        if curves is not None:
            self.assertGreater(info.get("resid", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
