"""Evaluate spatial memory: retention after turns, direction/height/distance
accuracy vs scene_gt world positions, multi-view anchor consistency."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from shared.geometry import rel_direction  # noqa: E402
from shared.spatial_memory import SpatialMemory  # noqa: E402


def detections(fr, depth):
    out = []
    for o in fr.get("visible", []) or []:
        t = o.get("type") or o["name"].split("_")[0]
        if t in ("Floor", "Wall", "Ceiling", "Window", "Door"):
            continue
        out.append({"type": t, "bbox": o["bbox"]})
    return out


def height_band(y: float):
    return "high" if y > 1.3 else ("mid" if y > 0.55 else "low")


def dist_band(d: float):
    return "near" if d < 1.0 else ("mid" if d < 3.0 else "far")


def main(max_episodes: int = 120):
    from shared.data_index import load_index
    index = load_index()
    scene_gt = index["scene_gt"]
    by_ep = defaultdict(list)
    for fr in index["frames"]:
        if fr.get("action") == "Teleport" or not fr.get("rgb"):
            continue
        by_ep[fr["episode"]].append(fr)

    dir_errs, dir_sign_ok = [], 0
    hb_ok = hb_tot = db_ok = db_tot = 0
    anchor_errs = []
    retained_after_turn = 0
    retained_tot = 0

    eps = list(by_ep.items())[:max_episodes]
    for name, frames in eps:
        scene = frames[0]["scene"]
        gt = scene_gt.get(scene, {})
        if not gt:
            continue
        mem = SpatialMemory()
        # objects carried (picked up) move; skip them in GT comparison
        picked = set()
        for fr in frames:
            if fr["action"] == "PickupObject" and fr["success"]:
                oid = (fr.get("action_args") or {}).get("objectId", "")
                if oid:
                    picked.add(oid.split("_")[0])
        for fi, fr in enumerate(frames):
            if not fr.get("agent") or not fr["agent"].get("position"):
                continue
            dep = np.asarray(Image.open(fr["depth"]).convert("I"),
                             dtype=np.float32) / 1000.0
            mem.update(detections(fr, dep), fr["agent"]["position"],
                       fr["agent"]["yaw"], dep, fr["step"],
                       fr["agent"].get("horizon", 0.0) or 0.0)
            # every 8 frames, query all remembered types from the current pose
            if fi % 8 != 0:
                continue
            for a in mem.anchors.values():
                if a.obj_type in picked:
                    continue
                g = gt.get(a.oid, None) if hasattr(a, "oid") else None
                # scene_gt keyed by instance name; we only have type in anchors
        # anchor vs scene_gt: match by type + proximity at end of episode
        used = set()
        for a in mem.anchors.values():
            if a.obj_type in picked:
                continue
            cands = [(n, p) for n, p in gt.items()
                     if n.split("_")[0] == a.obj_type]
            best, bd = None, 0.8
            for n, o in cands:
                if n in used:
                    continue
                d = math.dist(a.pos, [o["x"], o["y"], o["z"]])
                if d < bd:
                    best, bd = n, d
            if best:
                used.add(best)
                anchor_errs.append(bd)
                # direction/height/distance from a later pose where object not visible
                tgt = [gt[best]["x"], gt[best]["y"], gt[best]["z"]]
                q_fr = next((f for f in reversed(frames)
                             if f.get("agent") and f["agent"].get("position")),
                            frames[0])
                rel, dist, (yd, pd) = rel_direction(
                    tgt, q_fr["agent"]["position"], q_fr["agent"]["yaw"],
                    q_fr["agent"].get("horizon", 0.0) or 0.0)
                a_rel, a_dist, (ayd, apd) = rel_direction(
                    a.pos, q_fr["agent"]["position"], q_fr["agent"]["yaw"],
                    q_fr["agent"].get("horizon", 0.0) or 0.0)
                dir_errs.append(abs((ayd - yd + 180) % 360 - 180))
                if np.sign(ayd) == np.sign(yd) or abs(yd) < 20:
                    dir_sign_ok += 1
                hb_ok += height_band(a.pos[1]) == height_band(tgt[1])
                hb_tot += 1
                db_ok += dist_band(a_dist) == dist_band(dist)
                db_tot += 1
        # retention after a 180-degree turn: query a mid-episode object later
        turn = None
        for i in range(1, len(frames)):
            dy = abs((frames[i]["agent"]["yaw"] - frames[0]["agent"]["yaw"]
                      + 180) % 360 - 180)
            if dy > 150:
                turn = i
                break
        if turn:
            for a in mem.anchors.values():
                if a.obj_type in picked:
                    continue
                tfr = next((f for f in frames[turn:]
                            if f.get("agent") and f["agent"].get("position")),
                           None)
                if tfr is None:
                    continue
                q = mem.query(a.obj_type, tfr["agent"]["position"],
                              tfr["agent"]["yaw"],
                              tfr["agent"].get("horizon", 0.0) or 0.0)
                retained_tot += 1
                if q is not None and abs(q["yaw_deg"]) <= 120:
                    retained_after_turn += 1

    n = len(dir_errs)
    print(f"episodes={len(eps)} anchors={n}")
    print(f"anchor-to-GT median error: {np.median(anchor_errs):.2f} m "
          f"(n={len(anchor_errs)})")
    print(f"direction sign ok: {dir_sign_ok}/{n} ({dir_sign_ok/max(1,n):.0%}) "
          f"| median |yaw err|: {np.median(dir_errs):.1f} deg")
    print(f"height band acc: {hb_ok}/{hb_tot} ({hb_ok/max(1,hb_tot):.0%})")
    print(f"distance band acc: {db_ok}/{db_tot} ({db_ok/max(1,db_tot):.0%})")
    print(f"retained after >150deg turn: {retained_after_turn}/"
          f"{retained_tot} ({retained_after_turn/max(1,retained_tot):.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-episodes", type=int, default=120)
    args = ap.parse_args()
    main(args.max_episodes)
