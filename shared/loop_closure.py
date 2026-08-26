"""Loop closure for the object-centric spatial memory.

The world model's odometry (action-log dead reckoning) accumulates drift over
long trajectories.  This module adds the missing correction mechanism:

  1. place fingerprint : normalized DINOv2 CLS embedding of the current view
  2. revisit detection : cosine similarity against stored place keyframes
  3. drift correction  : keep a world-frame offset (odom -> world); when a
     revisit is detected, re-align the offset so the remembered place's world
     position matches, then translate all anchors by the delta.

The wrapped SpatialMemory stays unchanged (it always receives world-frame
poses); the offset lives here.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spatial_memory import SpatialMemory  # noqa: E402

# ---------------------------------------------------------------------------
# Place fingerprint (DINOv2 CLS, frozen)
# ---------------------------------------------------------------------------

_fingerprinter = None


def _get_fingerprinter(device: str = "cpu"):
    global _fingerprinter
    if _fingerprinter is None:
        from transformers import Dinov2Model
        model = Dinov2Model.from_pretrained("facebook/dinov2-small")
        for p in model.parameters():
            p.requires_grad_(False)
        _fingerprinter = model.eval().to(device)
    return _fingerprinter


def fingerprint(rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    """rgb: (H, W, 3) uint8/float -> normalized MEAN PATCH embedding (384,).

    The CLS token collapses for textureless/random views (sim ~0.97), so it
    is a bad place fingerprint; the mean of patch tokens retains more spatial
    information and separates revisits better on real episodes.
    """
    import torch
    model = _get_fingerprinter(device)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    im = np.asarray(rgb, dtype=np.float32)
    if im.shape[:2] != (224, 224):
        from PIL import Image
        im = np.asarray(Image.fromarray(im.astype(np.uint8)).resize((224, 224)),
                        dtype=np.float32)
    # DINOv2 normalization (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = torch.from_numpy((im / 255.0 - mean) / std).permute(2, 0, 1)
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(pixel_values=x).last_hidden_state[0]      # (1+N, 384)
    patch = out[1:].mean(0).cpu().numpy().astype(np.float64)  # (384,)
    n = np.linalg.norm(patch)
    return patch / max(n, 1e-9)


# ---------------------------------------------------------------------------
# Place keyframes + loop closure
# ---------------------------------------------------------------------------

class PlaceNode:
    """A place keyframe.  `world_pos` is kept in the SAME world frame as the
    anchor map: every loop-closure correction transforms it too, so it never
    becomes a stale reference."""

    __slots__ = ("fp", "odom_pos", "odom_yaw", "world_pos", "world_yaw",
                 "step")

    def __init__(self, fp, odom_pos, odom_yaw, world_pos, world_yaw, step):
        self.fp = fp
        self.odom_pos = np.asarray(odom_pos, dtype=float)
        self.odom_yaw = odom_yaw
        self.world_pos = np.asarray(world_pos, dtype=float)
        self.world_yaw = world_yaw
        self.step = step


def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _apply_offset(pos_xy: np.ndarray, off_t: np.ndarray, off_yaw: float
                  ) -> np.ndarray:
    return _rot(off_yaw) @ np.asarray(pos_xy, dtype=float) + off_t


class LoopClosure:
    """Wraps a SpatialMemory; corrects the odom->world offset on revisits."""

    def __init__(self, memory: Optional[SpatialMemory] = None,
                 sim_thr: float = 0.95, min_pose_gap: float = 0.3,
                 min_step_gap: int = 3, keyframe_dist: float = 0.6,
                 device: str = "cpu"):
        self.mem = memory or SpatialMemory()
        self.off_t = np.zeros(2)
        self.off_yaw = 0.0
        self.places: List[PlaceNode] = []
        self.sim_thr = sim_thr
        self.min_pose_gap = min_pose_gap
        self.min_step_gap = min_step_gap
        self.keyframe_dist = keyframe_dist
        self.device = device
        self.n_closure = 0
        self.closure_steps: List[int] = []
        self.corrections: List[tuple] = []  # (step, R, p_cur, p_node)
        self.last_place_step = -10 ** 9

    # -- pose helpers -------------------------------------------------------
    def world_pose(self, odom_pos: dict, odom_yaw: float):
        p = _apply_offset(np.array([odom_pos["x"], odom_pos["z"]]),
                          self.off_t, self.off_yaw)
        return {"x": float(p[0]), "z": float(p[1]), "y": odom_pos.get("y", 0.9)},\
            odom_yaw + self.off_yaw

    def world_pose_off(self, odom_pos: dict, odom_yaw: float,
                       off_t, off_yaw) -> dict:
        """World pose under an explicit offset (for eval snapshots)."""
        p = _apply_offset(np.array([odom_pos["x"], odom_pos["z"]]),
                          off_t, off_yaw)
        return {"x": float(p[0]), "z": float(p[1]), "y": odom_pos.get("y", 0.9)}

    def observe(self, rgb, detections, odom_pos: dict, odom_yaw: float,
                depth=None, step: int = 0, use_fp: bool = True) -> List[Dict]:
        """odom_pos/yaw come from the action-log odometry (drifting)."""
        wpos, wyaw = self.world_pose(odom_pos, odom_yaw)
        self.mem.update(detections, wpos, wyaw, depth, step)
        if not use_fp or rgb is None:
            return self.mem.summary(wpos, wyaw)
        fp = fingerprint(rgb, self.device)
        self._keyframe(fp, odom_pos, odom_yaw, step)
        self._close_loop(fp, odom_pos, odom_yaw, step)
        return self.mem.summary(wpos, wyaw)

    # -- keyframe management ------------------------------------------------
    def _keyframe(self, fp, odom_pos, odom_yaw, step):
        xy = np.array([odom_pos["x"], odom_pos["z"]])
        if self.places:
            last = self.places[-1]
            if np.linalg.norm(xy - last.odom_pos) < self.keyframe_dist and \
                    abs(step - last.step) < 4:
                return
        self.places.append(PlaceNode(fp, xy, float(odom_yaw),
                                     _apply_offset(xy, self.off_t,
                                                   self.off_yaw),
                                     float(odom_yaw) + self.off_yaw, step))

    # -- revisit detection + correction -------------------------------------
    def _close_loop(self, fp, odom_pos, odom_yaw, step):
        if len(self.places) < 2:
            return
        xy = np.array([odom_pos["x"], odom_pos["z"]])
        best, best_sim = None, self.sim_thr
        for node in self.places[:-1]:
            if step - node.step < self.min_step_gap:
                continue
            if np.linalg.norm(xy - node.odom_pos) < self.min_pose_gap:
                continue  # odometry thinks we're still near the same place
            s = float(np.dot(fp, node.fp))
            if s > best_sim:
                best, best_sim = node, s
        if best is None:
            return
        # Single-offset correction (v1):
        #   - where we currently think we are (world frame)
        p_cur = _apply_offset(xy, self.off_t, self.off_yaw)
        #   - where the matched place really is.  Nodes are true-world
        #     references and are NEVER transformed.
        p_node = best.world_pos
        delta = p_node - p_cur
        if float(np.linalg.norm(delta)) < 0.05:
            return
        # update the odom->world offset for future observations; anchors keep
        # their near-true positions and new observations associate with them
        self.off_t = self.off_t + delta
        self.n_closure += 1
        self.closure_steps.append(step)
        self.corrections.append((step, delta))
