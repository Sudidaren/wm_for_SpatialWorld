"""Unit tests for the 3-D observability coverage primitive (OCM)."""

from __future__ import annotations

import unittest

from shared.geometry import observed_bands


def key(x, z, cell=0.25):
    return (int(round(x / cell)), int(round(z / cell)))


AGENT = {"x": 0.0, "y": 0.9, "z": 0.0}


class OCMTest(unittest.TestCase):
    def test_level_view_covers_front_mid_but_not_back(self):
        cov = observed_bands(AGENT, yaw=0.0, horizon=0.0)
        self.assertIn("mid", cov.get(key(0.0, 1.0), set()))
        self.assertNotIn(key(0.0, -1.0), cov)  # behind the camera

    def test_yaw_90_covers_positive_x(self):
        cov = observed_bands(AGENT, yaw=90.0, horizon=0.0)
        self.assertIn("mid", cov.get(key(1.0, 0.0), set()))

    def test_look_down_sees_low_near(self):
        cov = observed_bands(AGENT, yaw=0.0, horizon=60.0)
        self.assertIn("low", cov.get(key(0.0, 0.5), set()))

    def test_occlusion_blocks_far_cell(self):
        wall = key(0.0, 0.5)
        cov_clear = observed_bands(AGENT, yaw=0.0, horizon=0.0)
        cov_wall = observed_bands(AGENT, yaw=0.0, horizon=0.0,
                                  occupancy_cells={wall})
        far = key(0.0, 1.0)
        self.assertIn("mid", cov_clear.get(far, set()))
        self.assertNotIn(far, cov_wall)


if __name__ == "__main__":
    unittest.main()
