"""AI2-THOR camera geometry (validated against episode data + official docs).

Conventions (all unit tests in test_geometry.py):
  - world frame: x right, y up, z ... AI2-THOR.  Agent position is the body
    root; the camera sits at ``agent.position + (0, CAMERA_Y, 0)`` with
    CAMERA_Y = 0.675 (ai2thor initialization cameraY default).
  - yaw: rotation.y; forward = (sin(yaw), cos(yaw)) in (x, z)  [validated by
    multi-view triangulation on episode data].
  - depth: perspective-correct forward distance in meters (PNG stores mm).
  - image: u right, v down; principal point at center.

This module is the single source of truth for projection; everything else
(perception heads, spatial memory, scene graph, occupancy completion) uses it.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

CAMERA_Y = 0.675


def camera_position(agent_pos: dict) -> np.ndarray:
    return np.array(
        [agent_pos["x"], agent_pos["y"] + CAMERA_Y, agent_pos["z"]], dtype=float
    )


def camera_basis(yaw: float, horizon: float = 0.0
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (forward, right, up) 3-vectors in the world frame.

    horizon: AI2-THOR cameraHorizon in degrees; negative = looking up,
    positive = looking down (ManipulaTHOR doc convention).
    """
    rad = math.radians(yaw)
    fwd_h = np.array([math.sin(rad), 0.0, math.cos(rad)])
    right = np.array([math.cos(rad), 0.0, -math.sin(rad)])
    up = np.array([0.0, 1.0, 0.0])
    if abs(horizon) > 1e-6:
        th = math.radians(-horizon)  # negative horizon -> pitch up
        c, s = math.cos(th), math.sin(th)
        fwd = fwd_h * c + up * s
        up_axis = up * c - fwd_h * s
    else:
        fwd, up_axis = fwd_h, up
    return fwd, right, up_axis


def intrinsics(width: int = 800, height: int = 600, fov: float = 60.0):
    fx = (width / 2.0) / math.tan(math.radians(fov) / 2.0)
    return fx, fx, width / 2.0, height / 2.0


def project(
    world: np.ndarray,
    agent_pos: dict,
    yaw: float,
    horizon: float = 0.0,
    width: int = 800,
    height: int = 600,
    fov: float = 60.0,
) -> Optional[Tuple[float, float]]:
    """Project a world point into the image. Returns (u, v) or None if behind."""
    fx, fy, cx, cy = intrinsics(width, height, fov)
    cam = camera_position(agent_pos)
    fwd, right, up = camera_basis(yaw, horizon)
    d = np.asarray(world, dtype=float) - cam
    x_rel = float(np.dot(d, right))
    y_rel = float(np.dot(d, up))
    z_rel = float(np.dot(d, fwd))
    if z_rel <= 0.05:
        return None
    return cx + fx * x_rel / z_rel, cy - fy * y_rel / z_rel


def unproject(
    u: float,
    v: float,
    depth: float,
    agent_pos: dict,
    yaw: float,
    horizon: float = 0.0,
    width: int = 800,
    height: int = 600,
    fov: float = 60.0,
) -> np.ndarray:
    """Unproject one pixel to a world point. depth is forward distance (m)."""
    fx, fy, cx, cy = intrinsics(width, height, fov)
    cam = camera_position(agent_pos)
    fwd, right, up = camera_basis(yaw, horizon)
    x_rel = (u - cx) * depth / fx
    y_rel = (cy - v) * depth / fy
    return cam + right * x_rel + up * y_rel + fwd * depth


def unproject_mask(
    us: np.ndarray,
    vs: np.ndarray,
    depth: np.ndarray,
    agent_pos: dict,
    yaw: float,
    horizon: float = 0.0,
    width: int = 800,
    height: int = 600,
    fov: float = 60.0,
) -> np.ndarray:
    """Unproject many pixels at once. Returns (N, 3) world points."""
    fx, fy, cx, cy = intrinsics(width, height, fov)
    cam = camera_position(agent_pos)
    fwd, right, up = camera_basis(yaw, horizon)
    z = depth.astype(np.float64)
    x_rel = (us - cx) * z / fx
    y_rel = (cy - vs) * z / fy
    out = np.empty((len(us), 3), dtype=np.float64)
    out[:, 0] = cam[0] + right[0] * x_rel + up[0] * y_rel + fwd[0] * z
    out[:, 1] = cam[1] + right[1] * x_rel + up[1] * y_rel + fwd[1] * z
    out[:, 2] = cam[2] + right[2] * x_rel + up[2] * y_rel + fwd[2] * z
    return out


# Bands approximate the physical layers objects live on in household scenes.
DEFAULT_BAND_HEIGHTS = {"low": 0.25, "mid": 1.05, "high": 1.95}


def observed_bands(
    agent_pos: dict,
    yaw: float,
    horizon: float = 0.0,
    occupancy_cells=frozenset(),
    *,
    fov: float = 60.0,
    vertical_fov: float = 60.0,
    max_dist: float = 3.0,
    cell_size: float = 0.25,
    band_heights: Optional[dict] = None,
) -> dict:
    """3-D observability coverage of one egocentric view.

    Returns {(ix, iz): set(band)} for every grid cell whose representative
    band height falls inside the view frustum (yaw FOV x horizon FOV) and is
    not occluded by an occupancy cell along the line of sight.

    Conventions follow the rest of this module: horizon negative = looking up,
    yaw forward = (sin, cos); agent body root + CAMERA_Y is the eye position.

    This is the "did I actually look there?" primitive of the 3-D curiosity
    map: a visited cell with an unobserved high/low band stays curious.
    """
    cam = camera_position(agent_pos)
    fwd, right, up = camera_basis(yaw, horizon)
    bands = band_heights or DEFAULT_BAND_HEIGHTS
    occ = set(occupancy_cells or ())

    # candidate cells inside the horizontal frustum
    cx, cz = cam[0], cam[2]
    half_h = math.tan(math.radians(fov) / 2.0)
    half_v = math.tan(math.radians(vertical_fov) / 2.0)
    center_ang = -float(horizon)  # negative horizon = looking up
    half_v_deg = math.degrees(math.atan(half_v))
    r = max(1, int(math.ceil(max_dist / cell_size)))
    ox, oz = int(round(cx / cell_size)), int(round(cz / cell_size))
    out: dict = {}
    for ix in range(ox - r, ox + r + 1):
        for iz in range(oz - r, oz + r + 1):
            key = (ix, iz)
            if key in occ:
                continue  # the cell itself is an obstacle
            wx = ix * cell_size
            wz = iz * cell_size
            dx = wx - cx
            dz = wz - cz
            x_rel = dx * right[0] + dz * right[2]
            z_rel = dx * fwd[0] + dz * fwd[2]
            if z_rel <= 0.05:
                continue
            dist = math.hypot(dx, dz)
            if dist > max_dist:
                continue
            if abs(x_rel) > half_h * z_rel:
                continue  # outside horizontal FOV
            seen = set()
            for band, h in bands.items():
                y_rel = h - cam[1]
                ang = math.degrees(math.atan2(y_rel, z_rel))
                if abs(ang - center_ang) > half_v_deg:
                    continue  # outside vertical FOV
                if _line_blocked(cx, cz, wx, wz, occ, cell_size):
                    continue
                seen.add(band)
            if seen:
                out[key] = seen
    return out


def _line_blocked(ax, az, bx, bz, occ, cell_size=0.25) -> bool:
    """True if any occupancy cell lies between agent and target cell."""
    dist = math.hypot(bx - ax, bz - az)
    steps = max(1, int(round(dist / cell_size)))
    for i in range(1, steps):
        t = i / steps
        c = (int(round((ax + (bx - ax) * t) / cell_size)),
             int(round((az + (bz - az) * t) / cell_size)))
        if c in occ:
            return True
    return False


def rel_direction(
    world: np.ndarray,
    agent_pos: dict,
    yaw: float,
    horizon: float = 0.0,
) -> Tuple[np.ndarray, float, float]:
    """Relative (right, up, forward) components, distance, and (yaw_deg, pitch_deg)
    of a world point from the agent's camera. Used for hint rendering:
    direction (left/right/front/back), height band (up/down), distance band.
    """
    cam = camera_position(agent_pos)
    fwd, right, up = camera_basis(yaw, horizon)
    d = np.asarray(world, dtype=float) - cam
    rx = float(np.dot(d, right))
    ry = float(np.dot(d, up))
    rz = float(np.dot(d, fwd))
    dist = float(np.linalg.norm(d))
    yaw_deg = math.degrees(math.atan2(rx, max(rz, 1e-6)))
    pitch_deg = math.degrees(math.atan2(ry, math.hypot(rx, rz)))
    return np.array([rx, ry, rz]), dist, (yaw_deg, pitch_deg)
