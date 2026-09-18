#!/usr/bin/env python3
"""Replay the shipped world model over recorded episodes and score anchors.

This is the measuring stick for the WM's *own* output: object positions in
metres.  It runs the real pipeline (RF-DETR + depth head -> WorldModel), one
recorded episode at a time, and compares every anchor the WM keeps against the
true object position that AI2-THOR encodes in the object id.

Two things make it useful for tuning:

* the default rooms are the **classic-family non-evaluation** houses
  (FloorPlan1/2/4/5/7/10/19).  The depth head never trained on them, so they
  behave like the held-out evaluation domain without ever touching it;
* every knob that changes the answer is an argument, so an A/B is one command:
  ``--fov-convention``, ``--tri-weight``, ``--sampling``.

Scoring: each anchor is matched to the nearest not-yet-used ground-truth object
of the same type (2 m cap); the error is the 3D and horizontal distance between
the anchor and that object.  Both are reported per anchor, plus split by
"seen from >=2 poses" (which is where triangulation kicks in).

Usage:
    python3 tools/eval_wm_positions.py --episodes 12 --device cuda
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASSIC_NOEVAL = ["FloorPlan1", "FloorPlan2", "FloorPlan4", "FloorPlan5",
                  "FloorPlan7", "FloorPlan10", "FloorPlan19"]


def gt_position(object_id):
    parts = str(object_id).split("|")
    if len(parts) != 4:
        return None
    try:
        return np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    except ValueError:
        return None


def action_dict(raw):
    """Collector action -> the dict the world model consumes.

    The collector stores ``{"name": "MoveAhead", "args": {"moveMagnitude":
    0.25}, "kind": "move"}``; the world model wants ``action_name`` plus
    ``magnitude``/``degrees``/``object_type``.  ``Teleport`` (the coverage
    sweeps jump the agent around) has no world-model equivalent -- with
    ``--pose recorded`` the jump is taken from the recorded pose, and the
    teleport itself is passed through as an action name the world model
    ignores.
    """
    if isinstance(raw, str):
        name, _, rest = raw.partition("(")
        arg = rest.rstrip(")").strip()
        out = {"action_name": name.strip()}
        if arg:
            try:
                out["magnitude"] = out["degrees"] = float(arg)
            except ValueError:
                out["object_type"] = arg
        return out
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    args = raw.get("args") or {}
    out = {"action_name": name}
    if "degrees" in args:
        out["degrees"] = float(args["degrees"])
    if "moveMagnitude" in args:
        out["magnitude"] = float(args["moveMagnitude"])
    for key in ("objectId", "objectType"):
        if args.get(key):
            out["object_type"] = str(args[key]).split("|")[0]
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/lightwm_data")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--rooms", default="classic",
                    help="classic (default, non-eval classic homes), "
                         "all_non_eval, or a comma separated room list")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/depth_v2_20260918/dense_depth_v2_best.pt"))
    ap.add_argument("--detector", default=str(ROOT / "checkpoints/rfdetr_small_228094/checkpoint_best_total.pth"))
    ap.add_argument("--obj-thr", type=float, default=0.40)
    ap.add_argument("--fov-convention", default="vertical",
                    choices=("vertical", "horizontal"))
    ap.add_argument("--tri-weight", type=float, default=1.0)
    ap.add_argument("--sampling", default="center",
                    choices=("center", "boxmedian"),
                    help="depth probe for the anchor: the bbox centre pixel "
                         "(shipped) or the median over the inner box")
    ap.add_argument("--depth-source", default="head",
                    choices=("head", "da2"),
                    help="head = the project's trained depth head; da2 = the "
                         "public Depth-Anything-V2 metric indoor model")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--pose", default="deadreckon",
                    choices=("deadreckon", "recorded"),
                    help="deadreckon = exactly what ships (the pose is built "
                         "from the action log through frame-difference "
                         "verdicts); recorded = inject the recorded agent pose "
                         "each frame, which removes odometry drift so the "
                         "number isolates perception+geometry")
    ap.add_argument("--metric", default="both",
                    choices=("position", "surface", "both"),
                    help="position = 3D distance to the object's centre of "
                         "mass (what AI2-THOR encodes in the id); surface = "
                         "distance from the camera versus the ground-truth "
                         "depth at the object's bbox centre.  The centre of "
                         "mass sits *behind* the visible surface, so a head "
                         "that compresses depth can look better on 'position' "
                         "by accident -- always read the two together.")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    os.environ["LIGHTWM_FOV_CONVENTION"] = args.fov_convention
    os.environ["LIGHTWM_TRI_WEIGHT"] = str(args.tri_weight)
    os.environ["LIGHTWM_DEPTH_PROBE"] = args.sampling
    os.environ["LIGHTWM_DEPTH_SOURCE"] = args.depth_source
    sys.path.insert(0, "/home/sudidaren/SpatialWorld")

    import phase_b.runtime_factory as rf
    # keep the perception stack honest: this harness must use the shipped
    # runtime, not a special path
    runtime = rf.build_runtime({
        "perception_ckpt": args.ckpt,
        "detector_backend": "rfdetr_small_depth",
        "detector_path": args.detector,
        "obj_thr": args.obj_thr,
        "perception_runtime_root": str(ROOT),
        "device": args.device,
    })
    from mllm_base_agent.agent.world_model import WorldModel

    if args.rooms == "classic":
        rooms = set(CLASSIC_NOEVAL)
    elif args.rooms == "all_non_eval":
        split = json.loads((ROOT / "data/splits_noneval.json").read_text())
        s = split["scene_splits"]["ai2thor_floorplans"]
        rooms = set(s["train"]) | set(s["val"])
    else:
        rooms = set(args.rooms.split(","))

    eps = []
    for d in sorted((Path(args.root) / "episodes").iterdir()):
        if not (d / "episode.json").exists():
            continue
        room = d.name.split("__")[0].split("_")[0]
        if room not in rooms:
            continue
        eps.append(d)
        if len(eps) >= args.episodes:
            break
    if not eps:
        print("no episodes matched", sorted(rooms)[:5])
        return 2
    print(f"rooms={sorted(rooms)[:8]}  episodes={len(eps)}  device={args.device}")

    rows = []
    for d in eps:
        meta = json.loads((d / "episode.json").read_text())
        wm = WorldModel()
        wm.attach_perception(runtime)
        prev_rgb = None
        n_frame = 0
        #: step -> (frame index, camera xyz, {object type: pixel})
        seen_frames = {}
        for fr in meta.get("frames", [])[:args.max_frames]:
            try:
                rgb = np.asarray(Image.open(d / fr["rgb"]).convert("RGB"))
            except Exception:
                continue
            moved = None
            if prev_rgb is not None:
                mse = float(np.mean((rgb.astype(np.float32) -
                                     prev_rgb.astype(np.float32)) ** 2))
                moved = bool(mse > 1.0)
            if args.pose == "recorded":
                # the recorded pose must be in place *before* the observation,
                # because every unprojection of this frame uses it
                ag = fr.get("agent") or {}
                pos = ag.get("position") or {}
                if pos.get("x") is not None:
                    wm._pose = [float(pos["x"]), float(pos.get("y") or 0.0),
                                float(pos["z"]),
                                float((ag.get("rotation") or {}).get("y") or 0.0)]
                    wm._horizon = float(ag.get("cameraHorizon") or 0.0)
            try:
                wm.observe(None, action=action_dict(fr.get("action")),
                           moved=moved, action_ok=moved, frame=rgb)
            except Exception as exc:      # keep the sweep alive
                print(f"  skip frame ({type(exc).__name__}: {exc})")
                continue
            prev_rgb = rgb
            n_frame += 1
            ag = fr.get("agent") or {}
            pos = ag.get("position") or {}
            cam = (np.array([float(pos.get("x") or 0.0),
                             float(pos.get("y") or 0.0) + 0.675,
                             float(pos.get("z") or 0.0)])
                   if pos.get("x") is not None else None)
            px = {}
            for obj in fr.get("visible_objects") or []:
                box = obj.get("bbox") or []
                if len(box) == 4:
                    px[str(obj.get("objectId")).split("|")[0]] = (
                        int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
            seen_frames[n_frame] = (fr, cam, px)

        # score the final anchors against the true object positions
        # one entry per *object* (not per frame): the type must also be unique
        # in the room, otherwise "the nearest Cup" can be the wrong Cup
        by_id = {}
        for fr in meta.get("frames", []):
            for obj in fr.get("visible_objects") or []:
                g = gt_position(obj.get("objectId"))
                if g is not None:
                    by_id[str(obj.get("objectId"))] = g
        gts = [(oid.split("|")[0], g) for oid, g in by_id.items()]
        types_of = collections.defaultdict(set)
        for oid in by_id:
            types_of[oid.split("|")[0]].add(oid)
        used = set()
        for oid, slot in wm._slots.items():
            if slot.get("seen", 0) < 2:
                continue
            if len(types_of[slot["type"]]) != 1:
                continue
            best = None
            for i, (typ, g) in enumerate(gts):
                if i in used or typ != slot["type"]:
                    continue
                dist = float(np.linalg.norm(np.array(slot["pos"]) - g))
                if best is None or dist < best[0]:
                    best = (dist, i, g)
            if best is None or best[0] > 2.0:
                continue
            used.add(best[1])
            pos = np.array(slot["pos"])
            # surface metric: the anchor's range versus the ground-truth depth
            # at the object's bbox centre, in the frame the anchor last saw it
            err_surface = float("nan")
            if args.metric != "position":
                step = int(slot.get("last_seen") or 0)
                rec = seen_frames.get(step)
                if rec and rec[1] is not None and best[2] is not None:
                    fr2, cam2, px2 = rec
                    uv = px2.get(slot["type"])
                    if uv is not None:
                        try:
                            gt2 = np.asarray(Image.open(d / fr2["depth"])).astype(np.float64)
                            if gt2.ndim == 3:
                                gt2 = gt2[..., 0]
                            gt2 = gt2 / 1000.0
                            x, y = uv
                            if 0 <= x < gt2.shape[1] and 0 <= y < gt2.shape[0]:
                                err_surface = abs(
                                    float(np.linalg.norm(pos - cam2)) - float(gt2[y, x]))
                        except Exception:
                            pass
            rows.append(dict(
                episode=d.name, type=slot["type"], seen=slot.get("seen", 0),
                n_views=len(slot.get("obs") or []),
                tri_resid=slot.get("tri_resid", float("nan")),
                err_surface=err_surface,
                err3d=float(np.linalg.norm(pos - best[2])),
                errh=float(math.hypot(pos[0] - best[2][0], pos[2] - best[2][2])),
                errz=float(abs(pos[1] - best[2][1])),
                dist=float(np.linalg.norm(pos - np.array([0.0, 0.675, 0.0]))),
            ))
        print(f"  {d.name[:38]:38s} frames={n_frame:3d} anchors={len(wm._slots):3d}")

    if not rows:
        print("no anchors scored")
        return 1

    def med(k, rs=None):
        return statistics.median(r[k] for r in (rs if rs is not None else rows))

    print(f"\nanchors scored: {len(rows)}   "
          f"median 3D {med('err3d'):.3f} m   horizontal {med('errh'):.3f} m   "
          f"vertical {med('errz'):.3f} m")
    surf = [r for r in rows if np.isfinite(r.get("err_surface", float("nan")))]
    if surf:
        # the apples-to-apples "how far away is that surface" number, which is
        # what a depth head should be judged on
        print(f"surface metric (range vs GT depth at the bbox centre): "
              f"n={len(surf)}  median {med('err_surface', surf):.3f} m")
    multi = [r for r in rows if r["n_views"] >= 2]
    single = [r for r in rows if r["n_views"] < 2]
    for tag, rs in ((">=2 views (triangulated)", multi), ("1 view", single)):
        if rs:
            print(f"  {tag:24s} n={len(rs):4d}  3D {med('err3d', rs):.3f}  "
                  f"horizontal {med('errh', rs):.3f}")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    worst = sorted(((med("err3d", v), k, len(v)) for k, v in by.items()),
                   reverse=True)[:6]
    print("  最差类别:", ", ".join(f"{k} {e:.2f}m(n={n})" for e, k, n in worst))
    if args.json:
        Path(args.json).write_text(json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
