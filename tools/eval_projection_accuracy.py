#!/usr/bin/env python3
"""Measure world-model localisation error against recorded ground truth.

For every frame of a recorded episode we take each visible object, find a
pixel that really belongs to it (modal instance colour inside its bbox in the
segmentation frame), read the *ground-truth* metric depth there, and unproject
that pixel with two camera models:

    old = the runtime model before 2026-09-17
          (yaw only, camera at y=0, vertical offset applied with flipped sign)
    new = the model in shared/geometry.py
          (yaw + cameraHorizon, camera at agent + 0.675, correct sign)

and compares both against the object's true world position, which AI2-THOR
encodes in the object id itself ("Sink|-00.70|+00.93|-00.65").

This isolates the geometry layer: the detector and the monocular depth head
are not involved, because we deliberately use the recorded GT depth.  A real
run also carries detection/depth noise, but the *difference* between the two
camera models is pure geometry and shows up at full size here.

Usage:
    python3 tools/eval_projection_accuracy.py --root /mnt/d/lightwm_data \
        --episodes 12 --json out.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CAMERA_Y = 0.675                     # AI2-THOR default camera height


# --------------------------------------------------------------------------
# the two camera models
# --------------------------------------------------------------------------
def _intrinsics(width, height, fov):
    fx = (width / 2.0) / math.tan(math.radians(fov) / 2.0)
    return fx, fx, width / 2.0, height / 2.0


def unproject_old(u, v, z, ax, ay, az, yaw, width, height, fov):
    """Byte-for-byte the pre-2026-09-17 runtime formula."""
    fx, fy, cx, cy = _intrinsics(width, height, fov)
    x_rel = (u - cx) * z / fx
    y_rel = (v - cy) * z / fy                       # <-- flipped sign
    rad = math.radians(yaw)
    fwd = (math.sin(rad), math.cos(rad))
    right = (math.cos(rad), -math.sin(rad))
    return np.array([
        ax + right[0] * x_rel + fwd[0] * z,
        ay + y_rel,                                  # <-- camera height ignored
        az + right[1] * x_rel + fwd[1] * z,
    ])


def unproject_new(u, v, depth, agent_xyz, yaw, horizon,
                  width, height, fov):
    """shared/geometry.py: camera at agent + (0, CAMERA_Y, 0), basis from
    (yaw, horizon), y_rel measured upwards along the camera's up axis."""
    fx, fy, cx, cy = _intrinsics(width, height, fov)
    cam = np.array([agent_xyz[0], agent_xyz[1] + CAMERA_Y, agent_xyz[2]],
                   dtype=float)
    rad = math.radians(yaw)
    fwd_h = np.array([math.sin(rad), 0.0, math.cos(rad)])
    right = np.array([math.cos(rad), 0.0, -math.sin(rad)])
    up = np.array([0.0, 1.0, 0.0])
    th = math.radians(-float(horizon))
    if abs(th) > 1e-9:
        c, s = math.cos(th), math.sin(th)
        fwd = fwd_h * c + up * s
        up_axis = up * c - fwd_h * s
    else:
        fwd, up_axis = fwd_h, up
    x_rel = (u - cx) * depth / fx
    y_rel = (cy - v) * depth / fy                    # <-- correct sign
    return cam + right * x_rel + up_axis * y_rel + fwd * depth


# --------------------------------------------------------------------------
def gt_position(object_id: str):
    """AI2-THOR objectId is '<Type>|<x>|<y>|<z>' with the true position."""
    parts = str(object_id).split("|")
    if len(parts) != 4:
        return None
    try:
        return np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    except ValueError:
        return None


def object_pixel(seg, bbox):
    """A pixel that really belongs to the object: the centroid of the most
    frequent non-background instance colour inside the bbox."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(seg.shape[1], x2), min(seg.shape[0], y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = seg[y1:y2, x1:x2].reshape(-1, 3)
    colours, counts = np.unique(crop, axis=0, return_counts=True)
    order = np.argsort(-counts)
    for idx in order[:3]:                     # skip black background
        colour = colours[idx]
        if colour.max() == 0:
            continue
        mask = np.all(seg[y1:y2, x1:x2] == colour, axis=-1)
        ys, xs = np.nonzero(mask)
        if len(xs) < 4:
            continue
        return float(x1 + xs.mean()), float(y1 + ys.mean())
    return None


NEW_MODEL = None       # set in main() when --runtime is given


def project_center_sample(gt, agent_pos, yaw, horizon, depth_m,
                          width, height, fov, max_depth, depth_tol, skipped):
    """Round-trip: project the object's true centre with the correct camera
    model, keep it only if the pixel really shows that object (recorded depth
    matches the true optical-axis depth), then let the caller unproject it
    with both models.

    Returns a dict with the comparison row plus the two depth-convention
    ratios, or None when the sample is unusable.
    """
    fx, fy, cx, cy = _intrinsics(width, height, fov)
    cam = np.array([agent_pos["x"], (agent_pos.get("y") or 0.0) + CAMERA_Y,
                    agent_pos["z"]], dtype=float)
    rad = math.radians(yaw)
    fwd_h = np.array([math.sin(rad), 0.0, math.cos(rad)])
    right = np.array([math.cos(rad), 0.0, -math.sin(rad)])
    up = np.array([0.0, 1.0, 0.0])
    th = math.radians(-float(horizon))
    if abs(th) > 1e-9:
        c, s = math.cos(th), math.sin(th)
        fwd = fwd_h * c + up * s
        up_axis = up * c - fwd_h * s
    else:
        fwd, up_axis = fwd_h, up

    d = gt - cam
    z_axis = float(np.dot(d, fwd))
    if z_axis <= 0.3 or z_axis > max_depth:
        skipped["behind_or_far"] += 1
        return None
    u = cx + fx * float(np.dot(d, right)) / z_axis
    v = cy - fy * float(np.dot(d, up_axis)) / z_axis
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < width and 0 <= vi < height):
        skipped["projected_off_frame"] += 1
        return None
    depth_png = float(depth_m[vi, ui])
    if not (0.1 < depth_png < max_depth):
        skipped["bad_depth"] += 1
        return None
    if abs(depth_png - z_axis) > depth_tol:
        skipped["pixel_not_on_object"] += 1
        return None

    p_old = unproject_old(u, v, depth_png, agent_pos["x"], 0.0,
                          agent_pos["z"], yaw, width, height, fov)
    p_new = (NEW_MODEL or unproject_new)(u, v, depth_png,
                          (agent_pos["x"], agent_pos.get("y") or 0.0,
                                        agent_pos["z"]), yaw, horizon,
                                       width, height, fov)
    return dict(
        row=dict(
            episode="", scene="", horizon=horizon, depth=depth_png,
            err3d_old=float(np.linalg.norm(p_old - gt)),
            err3d_new=float(np.linalg.norm(p_new - gt)),
            errh_old=float(np.hypot(p_old[0] - gt[0], p_old[2] - gt[2])),
            errh_new=float(np.hypot(p_new[0] - gt[0], p_new[2] - gt[2])),
            # how far the two models place the same pixel horizontally --
            # this is what actually differs for a run that never shows
            # positions to the model (hand/contents coupling only)
            delta_xz=float(np.hypot(p_old[0] - p_new[0], p_old[2] - p_new[2])),
            errz_old=float(abs(p_old[1] - gt[1])),
            errz_new=float(abs(p_new[1] - gt[1])),
        ),
        ratio_axis=depth_png / z_axis,
        ratio_eucl=depth_png / float(np.linalg.norm(d)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/lightwm_data")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-depth", type=float, default=6.0)
    ap.add_argument("--mode", choices=("pixel", "center"), default="pixel",
                    help="pixel: unproject the object's visible centroid; "
                         "center: project the true centre and round-trip it "
                         "(no object-size confound)")
    ap.add_argument("--depth-tol", type=float, default=0.15,
                    help="center mode: how far the recorded depth at the "
                         "projected pixel may differ from the true "
                         "optical-axis depth")
    ap.add_argument("--json", default="")
    ap.add_argument("--runtime", default="",
                    help="path to mllm_base_agent/agent/world_model.py; when "
                         "given, the 'new' column uses the actual shipped "
                         "WorldModel._unproject instead of the local copy")
    args = ap.parse_args()

    runtime_unproject = None
    if args.runtime:
        import importlib.util
        d = os.path.dirname(os.path.abspath(args.runtime))
        if d not in sys.path:
            sys.path.insert(0, d)
        spec = importlib.util.spec_from_file_location("rt_wm_audit",
                                                      args.runtime)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _wm = mod.WorldModel()

        def runtime_unproject(u, v, depth, agent_xyz, yaw, horizon,
                              W, H, fov):
            return np.array(_wm._unproject(
                u, v, depth, {"x": agent_xyz[0], "y": agent_xyz[1],
                              "z": agent_xyz[2]},
                {"y": yaw, "horizon": horizon}))

        print(f"'new' column = shipped runtime: {args.runtime}\n")
    new_model = runtime_unproject or unproject_new
    globals()["NEW_MODEL"] = runtime_unproject

    ep_root = Path(args.root) / "episodes"
    if not ep_root.exists():
        print(f"no episodes under {ep_root}")
        return 2

    eps = sorted(p for p in ep_root.iterdir() if (p / "episode.json").exists())
    # prefer the sweep episodes: they deliberately vary cameraHorizon
    eps = sorted(eps, key=lambda p: ("sweep" not in p.name, p.name))[:args.episodes]

    rows = []
    depth_samples = []
    skipped = collections.Counter()
    for ep in eps:
        meta = json.loads((ep / "episode.json").read_text())
        W = int(meta.get("width", 800))
        H = int(meta.get("height", 600))
        fov = float(meta.get("fov", 60.0))
        for frame in meta.get("frames", []):
            agent = frame.get("agent") or {}
            pos = agent.get("position") or {}
            yaw = float((agent.get("rotation") or {}).get("y") or 0.0)
            horizon = float(agent.get("cameraHorizon") or 0.0)
            if pos.get("x") is None:
                skipped["no_agent_pose"] += 1
                continue
            try:
                depth_img = np.asarray(Image.open(ep / frame["depth"]))
                seg = np.asarray(Image.open(ep / frame["seg"]).convert("RGB"))
            except Exception:
                skipped["frame_unreadable"] += 1
                continue
            if depth_img.ndim == 3:
                depth_m = depth_img[..., 0].astype(np.float64) / 1000.0
            else:
                depth_m = depth_img.astype(np.float64) / 1000.0

            for obj in frame.get("visible_objects") or []:
                gt = gt_position(obj.get("objectId"))
                if gt is None:
                    skipped["no_gt"] += 1
                    continue
                if args.mode == "center":
                    depth_samples.append(
                        project_center_sample(gt, pos, yaw, horizon, depth_m,
                                              W, H, fov, args.max_depth,
                                              args.depth_tol, skipped))
                    continue
                uv = object_pixel(seg, obj.get("bbox") or [])
                if uv is None:
                    skipped["no_pixel"] += 1
                    continue
                u, v = uv
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < depth_m.shape[1] and 0 <= vi < depth_m.shape[0]):
                    skipped["out_of_frame"] += 1
                    continue
                z = float(depth_m[vi, ui])
                if not (0.1 < z < args.max_depth):
                    skipped["bad_depth"] += 1
                    continue
                # the object centre is what we compare against; projects to a
                # pixel of its own, so re-use the true centre for the ray
                p_old = unproject_old(u, v, z, pos["x"], 0.0, pos["z"], yaw,
                                      W, H, fov)
                p_new = new_model(u, v, z, (pos["x"], pos.get("y") or 0.0,
                                            pos["z"]),
                                  yaw, horizon, W, H, fov)
                rows.append(dict(
                    episode=ep.name, scene=meta.get("scene"),
                    horizon=horizon, depth=z,
                    err3d_old=float(np.linalg.norm(p_old - gt)),
                    err3d_new=float(np.linalg.norm(p_new - gt)),
                    errh_old=float(np.hypot(p_old[0] - gt[0], p_old[2] - gt[2])),
                    errh_new=float(np.hypot(p_new[0] - gt[0], p_new[2] - gt[2])),
                    errz_old=float(abs(p_old[1] - gt[1])),
                    errz_new=float(abs(p_new[1] - gt[1])),
                ))

    if args.mode == "center":
        depth_samples = [s for s in depth_samples if s]
        rows = [s["row"] for s in depth_samples]
        if depth_samples:
            ratios_axis = sorted(s["ratio_axis"] for s in depth_samples)
            ratios_eucl = sorted(s["ratio_eucl"] for s in depth_samples)
            mid = len(depth_samples) // 2
            print("depth convention check (recorded PNG depth / hypothesis):")
            print(f"  optical-axis depth  median ratio = {ratios_axis[mid]:.3f}"
                  "   <- 1.0 means the PNG stores forward distance")
            print(f"  euclidean  distance  median ratio = {ratios_eucl[mid]:.3f}")
            print()

    if not rows:
        print("no usable samples"); print(skipped); return 1

    def med(key):
        return statistics.median(r[key] for r in rows)

    print(f"episodes: {len(eps)}   samples: {len(rows)}   skipped: {dict(skipped)}")
    print()
    print(f"{'metric':<28}{'old model':>12}{'new model':>12}{'improvement':>14}")
    print("-" * 66)
    for label, key in (("median 3D error (m)", "err3d"),
                       ("median horizontal err (m)", "errh"),
                       ("median vertical err (m)", "errz")):
        o, n = med(key + "_old"), med(key + "_new")
        print(f"{label:<28}{o:>12.3f}{n:>12.3f}{(o - n) / o * 100:>13.1f}%")

    dxz = sorted(r["delta_xz"] for r in rows)
    def pct(q):
        return dxz[min(len(dxz) - 1, int(q * len(dxz)))]
    print()
    print("horizontal disagreement between the two models (same pixel):")
    print(f"  median {pct(0.5):.3f} m   p90 {pct(0.9):.3f} m   p99 {pct(0.99):.3f} m")
    print()
    print("by camera pitch:")
    print(f"{'horizon':<12}{'n':>7}{'3D old':>10}{'3D new':>10}{'Δ':>9}")
    buckets = collections.defaultdict(list)
    for r in rows:
        h = float(r["horizon"])
        # NOTE: keep the sign.  AI2-THOR stores -30.0 when looking up and
        # +30.0000038 when looking down, so bucketing on abs() silently puts
        # mirrored conditions in different bins.
        b = f"{h:+.0f}"
        buckets[b].append(r)
    for b in sorted(buckets):
        rs = buckets[b]
        o = statistics.median(x["err3d_old"] for x in rs)
        n = statistics.median(x["err3d_new"] for x in rs)
        print(f"{b:<12}{len(rs):>6}{o:>10.3f}{n:>10.3f}{(o - n) / o * 100:>7.1f}%")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"samples": len(rows), "rows": rows[:5000]}, ensure_ascii=False))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
