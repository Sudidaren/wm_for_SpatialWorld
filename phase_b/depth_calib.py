"""Metric depth self-calibration from the floor plane.

Why this exists
---------------
The monocular depth head is accurate *in the domain it was trained on* but its
metric scale collapses on unseen rooms (measured: predictions come out at
0.59-0.62x the true distance on the held-out SpatialWorld rooms).  A global
scale error is exactly the kind of failure a *geometric* cue can remove,
because the camera height above the floor is a known constant of the rig
(``CAMERA_Y = 0.675 m``) and the floor is the one surface every room shares.

The estimator
-------------
For a pixel ``(u, v)`` with forward-axis depth ``z`` the world height of the
unprojected point is ``p_y = h - a`` where::

    a = ((v - cy) * z / fy) * cos(pitch) - z * sin(pitch)

depends only on the pixel, the depth and the known camera pitch.  ``a`` is
therefore the camera height that *would* put that point on the floor.  For a
pixel that really is on the floor ``a == h``; every other surface sits above
the floor and returns ``a < h``.

If the head's depth is globally wrong by a factor ``k`` (``z_pred = k * z_true``)
then every ``a`` is scaled by the same ``k``, so the floor's sharp peak moves
from ``h`` to ``k * h`` and::

    k = a_peak / h        ->        d_corrected = d_pred / k

Neither GT depth, nor object labels, nor simulator state is involved: this is
built from the camera intrinsics, the agent's own pitch/height (proprioception
the baseline also has) and the depth *prediction* itself.  Nothing here trains
on the evaluation rooms in any form.

The peak is taken as the *topmost strongly supported* peak rather than a plain
quantile: every horizontal surface (floor, table tops, counters) produces its
own sharp peak, and the floor is always the highest one because all the other
surfaces are above it.  Requiring mass and spatial support keeps a handful of
badly predicted pixels from dragging the estimate upward.

Measured status (2026-09-18, see ``results/depth_head_ab_20260918/
SCALE_CALIBRATION_20260918.md`` for the full tables)
--------------------------------------------------------------------
* The correction itself is worth a lot: applied with a *known* floor mask it
  takes the held-out room error from 0.72-1.07 m to 0.10-0.19 m, which is far
  below anything a single global scale can reach (a per-frame global scale
  fitted from GT is *worse* than no correction at all, because the head's
  error is distance dependent, not multiplicative).
* Finding the floor automatically is the unsolved half.  In the evaluation
  rooms the floor covers a median of only 3.8 % of the pixels (train rooms:
  7.1 %), so its histogram peak is a weak shoulder on a flat background and
  every detector tried here - topmost supported peak, histogram argmax,
  physics-constrained consensus, depth-map upper envelope, bump
  deconvolution, ADE20K floor segmentation - lands 10-50 % away from the true
  level.  :func:`fit_scale_curve` therefore reports ``resid``/``support`` so a
  caller can *gate* on it and keep the raw depth when the floor evidence is
  too thin.
* Nothing in this module is trained on, or tuned on, the evaluation rooms:
  the floor labels come from the camera height plus the predicted depth, and
  the intrinsic/height constants were verified separately against recorded
  object positions (vertical 60 deg FOV, camera 0.675 m, both good to ~1.5 %).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

#: AI2-THOR camera height above the agent root (``initializationParameters``).
CAMERA_Y = 0.675
#: ... which is NOT the camera's height above the floor.  Measured on the
#: collected episodes (unprojecting ground-truth depth):
#:
#:   * pixels whose depth puts them 1.576 m below the camera land at world
#:     y ~= 0.00  -> that is the floor (1.576 = agent y 0.901 + CAMERA_Y);
#:   * pixels 0.675 m below the camera land at world y ~= 0.92 -> counter tops.
#:
#: So ``GroundCalibConfig.cam_height`` must be read as "how far below the
#: camera the plane we anchor on sits", and anchoring on the *floor* needs
#: 1.576 m on this corpus, not 0.675 m.  The 0.675 default is kept because it
#: is the value every measurement in results/depth_head_ab_20260918 was made
#: with, and that plane (counter tops / tables) is the one that is actually
#: visible in most kitchen frames -- but it is a scene measurement, not a rig
#: constant, and the code must not pretend otherwise.
FLOOR_BELOW_CAMERA_DEFAULT = 1.576
#: AI2-THOR ``fieldOfView`` default.  Unity semantics: VERTICAL.
FOV_DEG = 60.0


@dataclass
class GroundCalibConfig:
    #: distance from the camera down to the plane the anchor uses.  This is
    #: NOT a rig constant: 0.675 m below the camera is the counter-top plane
    #: on the collected corpus, while the true floor sits 1.576 m below it
    #: (see the module-level note).  Anchoring on the floor means setting this
    #: to ``agent_y + CAMERA_Y``.
    cam_height: float = CAMERA_Y
    fov: float = FOV_DEG
    #: "vertical" (correct for AI2-THOR) or "horizontal" (legacy convention in
    #: shared/geometry.py).  Kept switchable so the choice can be measured.
    convention: str = "vertical"
    stride: int = 4
    z_min: float = 0.35
    z_max: float = 8.0
    #: plausible range for the recovered scale k = a_peak / cam_height
    min_scale: float = 0.20
    max_scale: float = 2.50
    bin_m: float = 0.02
    smooth_bins: int = 3
    #: a candidate peak must hold this fraction of the biggest peak's mass ...
    peak_rel_mass: float = 0.30
    #: ... and this fraction of every sampled pixel.
    peak_abs_mass: float = 0.030
    #: fraction of all sampled pixels within +/- tol_m of the peak that counts
    #: as "floor support"; below ``min_support`` the frame is declared
    #: uncalibratable and the caller keeps the raw prediction.
    support_tol_m: float = 0.06
    min_support: float = 0.05
    #: the supporting pixels must sit below the horizon row by this fraction
    #: of the image height (the floor is never above the horizon).
    min_below_horizon: float = 0.03
    # ---- floor-anchored curve fit -------------------------------------
    #: bootstrap: keep pixels whose implied height is this close to the peak
    floor_tol_m: float = 0.10
    #: refinement: keep pixels whose implied height is this close to the true
    #: camera height once the depth map has been corrected
    floor_tol_ref_m: float = 0.08
    refine_iters: int = 2
    curve_bins: int = 24
    curve_min_per_bin: int = 20
    min_floor_samples: int = 400
    max_curve_samples: int = 60000
    #: plausible exponent of ``true = pred**slope * exp(offset)``
    slope_min: float = 0.40
    slope_max: float = 1.70
    #: 0 = free slope, 1 = force slope 1 (pure scale).  Pulling the free fit
    #: slightly toward a pure scale keeps a narrow floor band from inventing a
    #: steep exponent out of noise.
    slope_shrink: float = 0.25
    #: reject the frame when the fitted curve does not actually explain the
    #: floor labels (median |log residual| in log metres)
    max_curve_residual: float = 0.12


def intrinsics(width: int, height: int, fov: float = FOV_DEG,
               convention: str = "vertical") -> Tuple[float, float, float, float]:
    if convention == "vertical":
        f = (height / 2.0) / math.tan(math.radians(fov) / 2.0)
    elif convention == "horizontal":
        f = (width / 2.0) / math.tan(math.radians(fov) / 2.0)
    else:
        raise ValueError(f"unknown FOV convention: {convention!r}")
    return f, f, width / 2.0, height / 2.0


def implied_heights(
    depth: np.ndarray,
    horizon_deg: float,
    *,
    width: int,
    height: int,
    cfg: Optional[GroundCalibConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(a, rows, cols)`` for the sampled pixels (see module doc)."""
    cfg = cfg or GroundCalibConfig()
    fx, fy, cx, cy = intrinsics(width, height, cfg.fov, cfg.convention)
    stride = max(1, int(cfg.stride))
    vs, us = np.mgrid[0:height:stride, 0:width:stride]
    z = np.asarray(depth, dtype=np.float64)[vs, us]
    ok = np.isfinite(z) & (z > cfg.z_min) & (z < cfg.z_max)
    if not np.any(ok):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    th = math.radians(-float(horizon_deg or 0.0))
    yr = (vs[ok] - cy) * z[ok] / fy
    a = yr * math.cos(th) - z[ok] * math.sin(th)
    return a, vs[ok], us[ok]


def horizon_row(horizon_deg: float, *, width: int, height: int,
                cfg: Optional[GroundCalibConfig] = None) -> float:
    """Image row of the horizon line for a known camera pitch."""
    cfg = cfg or GroundCalibConfig()
    _, fy, _, cy = intrinsics(width, height, cfg.fov, cfg.convention)
    th = math.radians(-float(horizon_deg or 0.0))
    # a == 0 exactly on the horizon: (v-cy)*inf... solve a=0 for the row at a
    # large-but-finite distance is ill conditioned, so use the pitch directly.
    return cy + fy * math.tan(th)


def _peak_of(a: np.ndarray, cfg: GroundCalibConfig) -> Tuple[float, float, np.ndarray]:
    """Topmost strongly supported peak of the implied-height histogram."""
    lo = max(1e-3, cfg.cam_height * cfg.min_scale)
    hi = cfg.cam_height * cfg.max_scale
    edges = np.arange(lo, hi + cfg.bin_m, cfg.bin_m)
    if len(edges) < 4:
        return float("nan"), 0.0, edges
    counts, edges = np.histogram(a, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    if counts.max() <= 0:
        return float("nan"), 0.0, centres
    n = max(1, len(a))
    # Smoothing widens a peak; "mass" below is therefore reported on the
    # *unsmoothed* profile so that the fractions keep their plain meaning.
    hist = counts.astype(np.float64)
    k = max(1, int(cfg.smooth_bins))
    if k > 1:
        hist = np.convolve(hist, np.ones(k) / k, mode="same")
    # local maxima
    is_peak = np.ones(len(hist), dtype=bool)
    is_peak[1:-1] = (hist[1:-1] >= hist[:-2]) & (hist[1:-1] >= hist[2:])
    strong = hist >= cfg.peak_rel_mass * hist.max()
    cand = np.nonzero(is_peak & strong)[0]
    if len(cand) == 0:
        return float("nan"), 0.0, centres
    best = int(cand[-1])                      # topmost strong peak = floor
    # report the mass of the whole (unsmoothed) band around the peak
    band = counts[np.abs(centres - centres[best]) <= 2 * cfg.bin_m].sum()
    return float(centres[best]), float(band / n), centres


def vertical_factor(horizon_deg: float, *, width: int, height: int,
                    cfg: Optional[GroundCalibConfig] = None) -> np.ndarray:
    """Per-row factor ``c(v)`` with ``world_height = cam_height - depth * c(v)``.

    A pixel is on the floor exactly when ``depth * c(v) == cam_height``, so
    ``cam_height / c(v)`` *is* the true depth of the floor at that row -- a
    label the estimator can use without any ground truth image.
    """
    cfg = cfg or GroundCalibConfig()
    _, fy, _, cy = intrinsics(width, height, cfg.fov, cfg.convention)
    th = math.radians(-float(horizon_deg or 0.0))
    v = np.arange(int(height), dtype=np.float64)
    return math.cos(th) * (v - cy) / fy - math.sin(th)


def implied_height_map(depth: np.ndarray, horizon_deg: float, *, width: int,
                       height: int, cfg: Optional[GroundCalibConfig] = None
                       ) -> np.ndarray:
    c = vertical_factor(horizon_deg, width=width, height=height, cfg=cfg)
    return np.asarray(depth, dtype=np.float64) * c[:, None]


def floor_depth_map(horizon_deg: float, *, width: int, height: int,
                    cfg: Optional[GroundCalibConfig] = None) -> np.ndarray:
    """Depth the floor plane *would* have at every pixel row (metres).

    This is derived, not recorded: it is ``cam_height / c(v)`` with the camera
    height a rigid-body constant, the intrinsics fixed, and the pitch read from
    the agent's own pose.  No simulator depth image, no annotation and no
    segmentation takes part -- the module only ever sees the RGB frame, the
    intrinsics, the known camera height and the agent's pitch.  The name says
    "floor" because it is only meaningful *for a pixel that really lies on the
    floor*; identifying those pixels is the other half of the problem (see
    :func:`implied_height_map`, which runs the same relation backwards from the
    predicted depth).
    """
    cfg = cfg or GroundCalibConfig()
    c = vertical_factor(horizon_deg, width=width, height=height, cfg=cfg)
    out = np.full((int(height), int(width)), np.inf, dtype=np.float64)
    ok = c > 1e-3
    out[ok, :] = cfg.cam_height / c[ok, None]
    return out


def estimate_scale(
    depth: np.ndarray,
    horizon_deg: float,
    *,
    width: int,
    height: int,
    cfg: Optional[GroundCalibConfig] = None,
) -> Tuple[float, Dict]:
    """Recover ``k`` with ``depth_pred ~= k * depth_true`` (NaN if undecidable).

    ``k < 1`` means the head compressed the scene (predictions too small).
    """
    cfg = cfg or GroundCalibConfig()
    a, rows, cols = implied_heights(depth, horizon_deg, width=width,
                                    height=height, cfg=cfg)
    info: Dict = {"samples": int(len(a))}
    if len(a) < 200:
        info["reason"] = "too_few_samples"
        return float("nan"), info
    peak, mass, centres = _peak_of(a, cfg)
    if not math.isfinite(peak) or mass < cfg.peak_abs_mass:
        info.update({"reason": "no_peak", "peak": peak, "peak_mass": mass})
        return float("nan"), info
    near = np.abs(a - peak) < cfg.support_tol_m
    support = float(near.mean())
    row_med = float(np.median(rows[near])) if np.any(near) else float("nan")
    hrow = horizon_row(horizon_deg, width=width, height=height, cfg=cfg)
    below = row_med - hrow
    info.update({"peak": peak, "peak_mass": mass, "support": support,
                 "row_median": row_med, "horizon_row": hrow,
                 "below_horizon": below, "cols_used": int(cols.size)})
    if support < cfg.min_support:
        info["reason"] = "weak_support"
        return float("nan"), info
    if not math.isfinite(below) or below < cfg.min_below_horizon * height:
        info["reason"] = "above_horizon"
        return float("nan"), info
    k = peak / cfg.cam_height
    if not (cfg.min_scale < k < cfg.max_scale):
        info["reason"] = "out_of_range"
        return float("nan"), info
    info["scale"] = float(k)
    return float(k), info


class GroundCalibrator:
    """Online scale tracker: per-frame estimate, pooled over recent frames.

    ``k`` is a property of the room and of the head's domain shift, not of a
    single frame, so pooling the per-frame estimates of the last ``window``
    frames cancels the estimation noise while still adapting when the agent
    walks into a room with different lighting/appearance.
    """

    def __init__(self, cfg: Optional[GroundCalibConfig] = None,
                 window: int = 12, min_frames: int = 3,
                 fallback: float = 1.0, ema: float = 0.35):
        self.cfg = cfg or GroundCalibConfig()
        self.window = int(window)
        self.min_frames = int(min_frames)
        self.fallback = float(fallback)
        self.ema = float(ema)
        self.history: list = []
        self.k: float = float(fallback)
        self.n_est: int = 0

    def update(self, depth: np.ndarray, horizon_deg: float, *,
               width: int, height: int) -> Tuple[float, Dict]:
        k, info = estimate_scale(depth, horizon_deg, width=width, height=height,
                                 cfg=self.cfg)
        if math.isfinite(k):
            self.history.append(k)
            del self.history[:-self.window]
            self.n_est += 1
            target = float(np.median(self.history))
            self.k = (target if self.n_est < self.min_frames
                      else (1 - self.ema) * self.k + self.ema * target)
        info = {**info, "k": self.k, "n_est": self.n_est,
                "history": len(self.history)}
        return self.k, info

    # -- floor-anchored curve ------------------------------------------------
    def calibrate(self, depth: np.ndarray, horizon_deg: float, *,
                  width: int, height: int) -> Tuple[np.ndarray, Dict]:
        """Return ``(calibrated_depth, info)`` for one frame."""
        curves, info = fit_scale_curve(depth, horizon_deg, width=width,
                                       height=height, cfg=self.cfg)
        if curves is None:
            return np.asarray(depth, dtype=np.float32), info
        history = self.__dict__.setdefault("curve_history", [])
        history.append(curves)
        del history[:-self.window]
        self.n_est += 1
        if len(history) >= self.min_frames:
            pooled = {
                "a": float(np.median([c["a"] for c in history])),
                "b": float(np.median([c["b"] for c in history])),
                "frames": len(history),
            }
            # a is a shape parameter (stable); only the offset is smoothed, and
            # only partially, so a real relocalisation still shows up fast.
            curves = {**curves, "b": (1 - self.ema) * curves["b"]
                      + self.ema * pooled["b"], "pooled_b": pooled["b"],
                      "pool_frames": len(history)}
        return apply_scale_curve(depth, curves), {**info, **curves}


def _weighted_logfit(x: np.ndarray, y: np.ndarray, cfg: GroundCalibConfig
                     ) -> Tuple[float, float, float, int]:
    """Robust ``y = a * x + b`` over quantile bins of ``x``.

    Returns ``(a, b, residual, n_bins)``.  Binning first makes the fit immune
    to the huge imbalance between near and far floor pixels.
    """
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    nb = max(3, int(cfg.curve_bins))
    xs, ys, ws = [], [], []
    for i in range(nb):
        lo, hi = int(i * n / nb), int((i + 1) * n / nb)
        if hi - lo < cfg.curve_min_per_bin:
            continue
        xs.append(float(np.median(x[lo:hi])))
        ys.append(float(np.median(y[lo:hi])))
        ws.append(math.sqrt(hi - lo))
    if len(xs) < 3:
        return float("nan"), float("nan"), float("nan"), len(xs)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    ws = np.asarray(ws)
    A = np.vstack([xs * ws, ws]).T
    coef, *_ = np.linalg.lstsq(A, ys * ws, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = float(np.median(np.abs(ys - (a * xs + b))))
    return a, b, resid, len(xs)


def fit_scale_curve(
    depth: np.ndarray,
    horizon_deg: float,
    *,
    width: int,
    height: int,
    cfg: Optional[GroundCalibConfig] = None,
) -> Tuple[Optional[Dict], Dict]:
    """Fit the per-frame correction ``true = pred**a * exp(b)`` from the floor.

    The floor is the only surface whose *true* depth is known from geometry
    alone (``cam_height / c(v)``), so its pixels hand us free labels.  Because
    the head's scale error varies with distance, a single scale is not enough:
    the log-linear fit follows the floor's own range of distances and is then
    applied to the whole frame.  Falls back to a pure scale, then to
    "no correction at all", when the floor evidence is too thin.
    """
    cfg = cfg or GroundCalibConfig()
    info: Dict = {}
    z = np.asarray(depth, dtype=np.float64)
    stride = max(1, int(cfg.stride))
    zs = z[::stride, ::stride]
    hs = np.arange(0, z.shape[0], stride, dtype=np.float64)
    c = vertical_factor(horizon_deg, width=width, height=height, cfg=cfg)
    cs = c[::stride]
    valid = np.isfinite(zs) & (zs > cfg.z_min) & (zs < cfg.z_max) & (cs[:, None] > 1e-3)
    if valid.sum() < cfg.min_floor_samples:
        info["reason"] = "too_few_valid"
        return None, info
    a_pred = zs * cs[:, None]
    peak, mass, _ = _peak_of(a_pred[valid], cfg)
    if not math.isfinite(peak) or mass <= 0:
        info["reason"] = "no_floor_peak"
        return None, info
    floor = valid & (np.abs(a_pred - peak) < max(cfg.floor_tol_m,
                                                cfg.floor_tol_ref_m))
    info.update({"peak": float(peak), "peak_mass": float(mass),
                 "n_floor": int(floor.sum())})
    if floor.sum() < cfg.min_floor_samples:
        info["reason"] = "weak_floor"
        return None, info
    # exact true depth of each selected pixel: it lies on the floor by
    # construction, so geometry alone gives the label.
    z_floor = floor_depth_map(horizon_deg, width=width, height=height,
                              cfg=cfg)[::stride, ::stride]
    curves = None
    for it in range(max(1, int(cfg.refine_iters))):
        sel = floor & np.isfinite(z_floor)
        x = np.log(zs[sel])
        y = np.log(z_floor[sel])
        if len(x) < cfg.min_floor_samples:
            info["reason"] = "weak_floor_refined"
            return None, info
        if len(x) > cfg.max_curve_samples:
            keep = np.linspace(0, len(x) - 1, cfg.max_curve_samples).astype(int)
            x, y = x[keep], y[keep]
        a, b, resid, nb = _weighted_logfit(x, y, cfg)
        if not (math.isfinite(a) and math.isfinite(b)):
            info["reason"] = "fit_failed"
            return None, info
        # shrink toward a pure scale and clamp to a physically sane exponent
        a_shrunk = (1 - cfg.slope_shrink) * a + cfg.slope_shrink * 1.0
        a_used = float(min(max(a_shrunk, cfg.slope_min), cfg.slope_max))
        b_used = float(np.median(y - a_used * x))
        resid = float(np.median(np.abs(y - (a_used * x + b_used))))
        curves = {"a": a_used, "b": b_used, "raw_a": float(a),
                  "raw_b": float(b), "resid": resid, "fit_bins": int(nb),
                  "iter": it}
        if resid > cfg.max_curve_residual:
            info.update({"reason": "bad_curve_fit", **curves})
            return None, info
        if it + 1 >= max(1, int(cfg.refine_iters)):
            break
        corrected = apply_scale_curve(zs, curves)
        a_corr = corrected * cs[:, None]
        floor = valid & (np.abs(a_corr - cfg.cam_height) < cfg.floor_tol_ref_m)
        if floor.sum() < cfg.min_floor_samples:
            break
    if curves is None:
        info["reason"] = "fit_failed"
        return None, info
    # effective pure scale at the floor's own median distance (reported only)
    info.update(curves)
    info["k_equiv"] = float(math.exp(curves["b"]
                                     + (curves["a"] - 1.0)
                                     * float(np.median(np.log(zs[valid])))))
    return curves, info


def apply_scale_curve(depth: np.ndarray, curves: Dict) -> np.ndarray:
    z = np.asarray(depth, dtype=np.float64)
    out = np.exp(curves["a"] * np.log(np.clip(z, 1e-3, None)) + curves["b"])
    return out.astype(np.float32)
