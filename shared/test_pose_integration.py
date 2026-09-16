"""Unit tests for dead-reckoning integration + landmark pose correction.

Run: python -m unittest shared/test_pose_integration.py

2026-09-16: realigned with the no-cheat WM runtime.
  * the module is loaded from the official SpatialWorld checkout; its
    ``self_observation`` sibling must be importable, so that directory goes on
    sys.path first;
  * ``WorldModel(...)`` no longer takes ``targets=`` (there is no task-truth
    channel any more);
  * ``_integrate_pose(action, moved)`` takes the *frame-difference* outcome
    (``False`` = the view did not change = blocked), not a simulator error
    string;
  * the ``_horizon`` (LookUp/LookDown pitch tracking) feature was removed, and
    so were its tests;
  * added a regression test for the action-magnitude resolution bug that used
    to make every move integrate the wrong distance.
"""

from __future__ import annotations

import os
import sys
import unittest
import importlib.util

_WM_DIR = os.environ.get(
    "LIGHTWM_WM_DIR",
    "/home/sudidaren/SpatialWorld/mllm_base_agent/agent")
if _WM_DIR not in sys.path:
    sys.path.insert(0, _WM_DIR)          # for world_model's flat fallback import

_SPEC = importlib.util.spec_from_file_location(
    "lightwm_world_model", os.path.join(_WM_DIR, "world_model.py"))
wm_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wm_mod)


def make_wm():
    return wm_mod.WorldModel(
        move_magnitudes={
            "MoveAhead": 0.5, "MoveBack": 0.5,
            "MoveLeft": 0.5, "MoveRight": 0.5,
            "MoveSmall": 0.25, "MoveMedium": 0.5, "MoveLarge": 1.0,
        })


def act(name, **kw):
    d = {"action_name": name}
    d.update(kw)
    return d


class PoseIntegrationTest(unittest.TestCase):
    def test_rotate_right_increases_yaw(self):
        wm = make_wm()
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        wm._integrate_pose(act("RotateRight", degrees=90), None)
        self.assertAlmostEqual(wm._pose[3], 90.0)

    def test_rotate_left_decreases_yaw(self):
        wm = make_wm()
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        wm._integrate_pose(act("RotateLeft", degrees=90), None)
        self.assertAlmostEqual(wm._pose[3], 270.0)

    def test_move_ahead_after_right_turn_goes_plus_x(self):
        wm = make_wm()
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        wm._integrate_pose(act("RotateRight", degrees=90), None)
        wm._integrate_pose(act("MoveAhead", magnitude=0.5), None)
        self.assertAlmostEqual(wm._pose[0], 0.5)
        self.assertAlmostEqual(wm._pose[2], 0.0)

    def test_move_left_right_signs(self):
        wm = make_wm()
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        wm._integrate_pose(act("MoveRight", magnitude=0.5), None)
        self.assertAlmostEqual(wm._pose[0], 0.5)
        wm._integrate_pose(act("MoveLeft", magnitude=0.5), None)
        self.assertAlmostEqual(wm._pose[0], 0.0)

    def test_failed_move_does_not_move_and_marks_blocked(self):
        wm = make_wm()
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        # moved=False is the frame-difference verdict, not a simulator message
        wm._integrate_pose(act("MoveAhead", magnitude=0.5), False)
        self.assertEqual(wm._pose[:3], [0.0, 0.0, 0.0])
        self.assertIn((0, 1), wm._blocked_from_moves)

    def test_bare_move_uses_small_magnitude(self):
        """Regression: a bare MoveAhead is 0.25 m, not move_ahead_magnitude."""
        wm = make_wm()
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        wm._integrate_pose(act("MoveAhead"), True)
        self.assertAlmostEqual(wm._pose[2], 0.25)

    def test_granularity_resolution(self):
        wm = make_wm()
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        wm._integrate_pose(act("MoveAhead", granularity="Large"), True)
        self.assertAlmostEqual(wm._pose[2], 1.0)
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        wm._integrate_pose(act("MoveAhead", granularity="Small"), True)
        self.assertAlmostEqual(wm._pose[2], 0.25)

    def test_look_actions_do_not_move_the_agent(self):
        wm = make_wm()
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        wm._integrate_pose(act("LookUp", degrees=30), None)
        wm._integrate_pose(act("LookDown", degrees=30), None)
        self.assertEqual(wm._pose, [0.0, 0.0, 0.0, 0.0])

    def test_look_actions_track_camera_horizon(self):
        """The pitch must be recorded even though it does not translate the
        agent: every unprojection depends on it (79% of the AI2-THOR tasks
        start with a LookDown).  AI2-THOR sign convention: +=down."""
        wm = make_wm()
        wm._horizon = 0.0
        wm._integrate_pose(act("LookDown"), None)         # default 30 deg
        self.assertAlmostEqual(wm._horizon, 30.0)
        wm._integrate_pose(act("LookUp"), None)
        self.assertAlmostEqual(wm._horizon, 0.0)
        wm._integrate_pose(act("LookDown", degrees=45), None)
        self.assertAlmostEqual(wm._horizon, 45.0)
        for _ in range(4):
            wm._integrate_pose(act("LookDown"), None)
        self.assertAlmostEqual(wm._horizon, 60.0)         # clamped
        for _ in range(6):
            wm._integrate_pose(act("LookUp"), None)
        self.assertAlmostEqual(wm._horizon, -60.0)

    def test_camera_sits_one_camera_height_above_the_agent(self):
        wm = make_wm()
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        pos, rot = wm._resolve_pose(None, None)
        self.assertAlmostEqual(wm._camera_xyz(pos, rot)[1], wm_mod.CAMERA_Y)
        self.assertIn("horizon", rot)

    def test_landmark_correction_removes_drift(self):
        wm = make_wm()
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        stored = [(3.0, 1.0, 0.0), (3.0, 1.0, 2.0), (5.0, 1.0, 1.0)]
        drift = (0.4, 0.2)
        measured = [(x + drift[0], y, z + drift[1]) for x, y, z in stored]
        wm._correct_pose_with_landmarks(
            [(s, m) for s, m in zip(stored, measured)])
        self.assertAlmostEqual(wm._pose[0], -drift[0], places=2)
        self.assertAlmostEqual(wm._pose[2], -drift[1], places=2)

    def test_landmark_correction_bounded(self):
        wm = make_wm()
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        stored = [(3.0, 1.0, 0.0), (3.0, 1.0, 2.0), (5.0, 1.0, 1.0)]
        measured = [(x + 0.8, y, z) for x, y, z in stored]  # 0.8 m > bound
        wm._correct_pose_with_landmarks(
            [(s, m) for s, m in zip(stored, measured)])
        self.assertEqual(wm._pose[:3], [0.0, 0.0, 0.0])

    def test_similarity_translation(self):
        import numpy as np
        src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        dst = src + np.array([0.5, -0.25])
        sc, R, t = wm_mod._similarity2d(src, dst)
        self.assertAlmostEqual(sc, 1.0)
        self.assertAlmostEqual(t[0], 0.5)
        self.assertAlmostEqual(t[1], -0.25)


if __name__ == "__main__":
    unittest.main()
