"""Evaluate loop closure on real episodes with simulated drifting odometry.

Protocol:
  - simulate odometry from the action log with injected drift;
  - replay with GT detections+depth, two worlds: with loop closure and
    without (raw odometry);
  - metrics:
      a) closure detection vs true-trajectory revisits (TP/FP);
      b) revisit consistency: the same GT object seen in the first and last
         third of an episode should map to the SAME world location.  Without
         closure the drift pushes the two observations apart; with closure
         they stay consistent;
      c) anchor-to-scene_gt error (overall).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from shared.geometry import unproject  # noqa: E402
from shared.loop_closure import LoopClosure  # noqa: E402
from shared.spatial_memory import SpatialMemory  # noqa: E402

SKIP = {"Floor", "Wall", "Ceiling", "Window", "Door"}


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def load_depth(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("I"), dtype=np.float32) / 1000.0


def detections(fr):
    return [{"type": o.get("type") or o["name"].split("_")[0], "bbox": o["bbox"]}
            for o in (fr.get("visible", []) or [])
            if o.get("bbox") is not None and
            (o.get("type") or o["name"].split("_")[0]) not in SKIP]


def simulate_odometry(frames, rot_err=0.3, move_err=0.010, rng=None):
    x = z = 0.0
    yaw = 0.0
    mags = {"MoveAhead": 0.5, "MoveBack": 0.5, "MoveLeft": 0.5,
            "MoveRight": 0.5}
    out = []
    for fr in frames:
        act = fr.get("action", "")
        args = fr.get("action_args", {}) or {}
        ok = fr.get("success", True)
        if act in mags and ok:
            m = float(args.get("magnitude") or mags[act]) * \
                (1 + rng.normal(0, move_err))
            if act == "MoveAhead":
                x += math.sin(math.radians(yaw)) * m
                z += math.cos(math.radians(yaw)) * m
            elif act == "MoveBack":
                x -= math.sin(math.radians(yaw)) * m
                z -= math.cos(math.radians(yaw)) * m
            elif act == "MoveRight":
                x += math.cos(math.radians(yaw)) * m
                z -= math.sin(math.radians(yaw)) * m
            else:
                x -= math.cos(math.radians(yaw)) * m
                z += math.sin(math.radians(yaw)) * m
        elif act in ("RotateRight", "RotateLeft") and ok:
            sign = 1 if act == "RotateRight" else -1
            yaw = (yaw + sign * (90 + rng.normal(0, rot_err))) % 360
        out.append((x, z, yaw))
    return out


def gt_revisits(traj, gap=3, dist=1.0):
    events = []
    pts = [(t["position"]["x"], t["position"]["z"]) for t in traj]
    for j in range(gap, len(pts)):
        for i in range(j - gap, -1, -1):
            if math.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]) < dist:
                events.append((i, j))
                break
    return events


def anchor_errors(mem, scene_gt, picked):
    errs = []
    used = set()
    for a in mem.anchors.values():
        if a.obj_type in picked:
            continue
        cands = [(n, p) for n, p in scene_gt.items()
                 if n.split("_")[0] == a.obj_type and n not in used]
        best, bd = None, 0.8
        for n, p in cands:
            d = math.dist(a.pos, [p["x"], p["y"], p["z"]])
            if d < bd:
                best, bd = n, d
        if best:
            used.add(best)
            errs.append(bd)
    return errs


def box_depth(dep, b):
    y0, y1 = max(0, b[1]), min(dep.shape[0], b[3] + 1)
    x0, x1 = max(0, b[0]), min(dep.shape[1], b[2] + 1)
    vals = dep[y0:y1, x0:x1]
    vals = vals[vals > 0.15]
    return float(np.median(vals)) if vals.size else None


class Replay:
    """Replays one episode under a given odom->world mapping, logging GT
    instance observations for consistency checks."""

    def __init__(self, frames, odom, scene_gt, use_closure, device):
        self.frames = frames
        self.odom = odom
        self.scene_gt = scene_gt
        self.mem = LoopClosure(SpatialMemory(), device=device) if use_closure \
            else SpatialMemory()
        self.use_closure = use_closure
        self.obs = defaultdict(list)   # instance name -> [(x,y,z,step)]
        self.pose_before, self.pose_after = [], []

    def world_pose(self, i, op, oyaw):
        if self.use_closure:
            return self.mem.world_pose(op, oyaw)
        return op, oyaw

    def run(self):
        picked = set()
        for fr in self.frames:
            if fr["action"] == "PickupObject" and fr["success"]:
                oid = (fr.get("action_args") or {}).get("objectId", "")
                if oid:
                    picked.add(oid.split("_")[0])
        for i, fr in enumerate(self.frames):
            if not fr.get("agent") or not fr["agent"].get("position"):
                continue
            dep = load_depth(fr["depth"])
            ox, oz, oyaw = self.odom[i]
            op = {"x": ox, "z": oz, "y": 0.9}
            dets = detections(fr)
            if self.use_closure:
                off_before = (self.mem.off_t.copy(), self.mem.off_yaw)
                self.mem.observe(load_rgb(fr["rgb"]), dets, op, oyaw, dep, i)
                if i in self.mem.closure_steps:
                    wp_b = self.mem.world_pose_off(op, oyaw, *off_before)
                    wp_a = self.mem.world_pose(op, oyaw)[0]
                    self.pose_before.append((wp_b["x"], wp_b["z"]))
                    self.pose_after.append((wp_a["x"], wp_a["z"]))
            else:
                wpos, wyaw = self.world_pose(i, op, oyaw)
                self.mem.update(dets, wpos, wyaw, dep, i)
            wpos, wyaw = self.world_pose(i, op, oyaw)
            for o in fr.get("visible", []) or []:
                if o.get("bbox") is None or \
                        (o.get("type") or o["name"].split("_")[0]) in SKIP:
                    continue
                z = box_depth(dep, o["bbox"])
                if z is None:
                    continue
                uc = (o["bbox"][0] + o["bbox"][2]) / 2
                vc = (o["bbox"][1] + o["bbox"][3]) / 2
                p = unproject(uc, vc, z, wpos, wyaw)
                self.obs[o["name"]].append((float(p[0]), float(p[1]),
                                            float(p[2]), i))
        return picked

    def consistency(self):
        """Median displacement between first-third and last-third observations
        of the same instance (revisit consistency)."""
        pts = self.obs
        third = max(1, len(self.frames) // 3)
        disp = []
        for oid, arr in pts.items():
            first = [p for p in arr if p[3] < third]
            last = [p for p in arr if p[3] >= 2 * third]
            if len(first) < 2 or len(last) < 2:
                continue
            f = np.median([p[:3] for p in first], axis=0)
            l = np.median([p[:3] for p in last], axis=0)
            disp.append(float(np.linalg.norm(f - l)))
        return disp


def main(max_episodes: int = 60, device: str = "cuda", seed: int = 0):
    from shared.data_index import load_index
    index = load_index()
    scene_gt_all = index["scene_gt"]
    by_ep = defaultdict(list)
    for fr in index["frames"]:
        if fr.get("action") == "Teleport" or not fr.get("rgb"):
            continue
        by_ep[fr["episode"]].append(fr)
    rng = np.random.RandomState(seed)

    cons_with, cons_without = [], []
    err_with, err_without = [], []
    det_tp = det_fp = closures = 0
    pose_before, pose_after = [], []
    start_offsets = []
    eps = list(by_ep.items())[:max_episodes]
    for name, frames in eps:
        scene = frames[0]["scene"]
        gt = scene_gt_all.get(scene, {})
        if not gt or len(frames) < 30:
            continue
        if not any(fr["action"] in ("MoveAhead", "MoveLeft", "MoveRight")
                   for fr in frames):
            continue
        odom = simulate_odometry(frames, rng=rng)
        true_pts = [(fr["agent"]["position"]["x"], fr["agent"]["position"]["z"])
                    for fr in frames]
        rev = gt_revisits([{"position": {"x": a, "z": b}} for a, b in true_pts])

        rw = Replay(frames, odom, gt, use_closure=True, device=device)
        picked = rw.run()
        rb = Replay(frames, odom, gt, use_closure=False, device=device)
        rb.run()
        start = np.array([true_pts[0][0], true_pts[0][1]])
        for j, (b, a) in enumerate(zip(rw.pose_before, rw.pose_after)):
            s = rw.mem.closure_steps[j] if j < len(rw.mem.closure_steps) else 0
            true_here = np.array([true_pts[s][0], true_pts[s][1]]) - start
            pose_before.append(math.dist(b, true_here))
            pose_after.append(math.dist(a, true_here))

        closures += len(rw.mem.closure_steps)
        for s in rw.mem.closure_steps:
            if any(abs(s - j) <= 2 for (_, j) in rev):
                det_tp += 1
            else:
                det_fp += 1
        cons_with.extend(rw.consistency())
        cons_without.extend(rb.consistency())
        err_with.extend(anchor_errors(rw.mem.mem if rw.use_closure else rw.mem,
                                      gt, picked))
        err_without.extend(anchor_errors(rb.mem, gt, picked))

    def med(x):
        return float(np.median(x)) if x else float("nan")

    print(f"episodes={len(eps)} closures={closures} detection TP={det_tp} "
          f"FP={det_fp}")
    print("revisit consistency (same object, first vs last third):")
    print(f"  WITH closure: {med(cons_with):.3f} m (n={len(cons_with)})")
    print(f"  WITHOUT      : {med(cons_without):.3f} m (n={len(cons_without)})")
    print("anchor error vs scene_gt:")
    print(f"  WITH closure: {med(err_with):.3f} m (n={len(err_with)})")
    print(f"  WITHOUT      : {med(err_without):.3f} m (n={len(err_without)})")
    if pose_before:
        print(f"current-pose error at closure: before={med(pose_before):.3f} m "
              f"after={med(pose_after):.3f} m (n={len(pose_before)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-episodes", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.max_episodes, args.device, args.seed)
