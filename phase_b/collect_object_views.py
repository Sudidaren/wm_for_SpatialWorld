"""Multi-angle object-centric data collector.

For every target object instance in a scene, teleports the agent to several
viewpoints around the object (different radii/angles), renders it under
multiple yaws (facing / +90deg) and horizons (0 / +25deg), and saves
RGB + depth + instance segmentation + bbox metadata in the SAME format as
the LightWM episodes, so the frame index picks it up directly.

Usage (WSLg display, envs/ai2thor venv):
  DISPLAY=:0 python phase_b/collect_object_views.py \
      --scenes FloorPlan1 FloorPlan2 FloorPlan10 ... --out /mnt/d/lightwm_data_objviews
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image

import ai2thor.controller

sys.path.insert(0, os.path.dirname(__file__))
from collect_coverage import collect_visible  # noqa: E402

SMALL_TARGETS = [
    "Watch", "CreditCard", "KeyChain", "Pencil", "AluminumFoil", "Egg",
    "StoveKnob", "Pen", "DishSponge", "Spoon", "SaltShaker", "SoapBar",
    "Potato", "PepperShaker", "ButterKnife", "Candle", "Spatula", "Fork",
    "CellPhone", "RemoteControl", "Knife", "PaperTowelRoll", "StoveBurner",
    "Apple", "Mug", "Ladle", "Tomato", "Plate", "Newspaper", "Plunger",
    "LightSwitch", "Bottle", "ToiletPaper", "Vase", "ShowerHead",
    "WateringCan", "Cup", "Lettuce", "TissueBox", "Kettle", "ScrubBrush",
    "Bowl", "Cloth", "BasketBall", "Pan", "SoapBottle", "ToiletPaperHanger",
    "Statue", "Book", "Dumbbell", "AlarmClock", "WineBottle", "SprayBottle",
    "Bread", "SinkBasin", "Faucet",
]
RADII = [1.0, 1.5, 2.0, 2.5]
Y_CANDS = [0.9, 1.1, 0.5]
MAX_RADIUS = 3.0
MIN_RADIUS = 0.8


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def scene_objects(scene: str, root: str) -> dict:
    p = os.path.join(root, "scene_gt", f"{scene}.json")
    if not os.path.exists(p):
        return {}
    g = json.load(open(p))
    out = {}
    for o in g.get("objects", []):
        nm = o.get("name", "")
        t = nm.split("_")[0] if "_" in nm else nm.split("|")[0]
        out[o["objectId"]] = {"objectId": o["objectId"], "name": nm,
                              "type": t,
                              "position": o.get("position")}
    return out


def pick_viewpoints(controller, op, n_angles: int, min_sep: float = 0.6
                    ) -> list:
    """Nearest reachable viewpoints with a minimum separation, so front
    viewpoints (where countertop objects are actually visible) are kept."""
    ev = controller.step(dict(action="GetReachablePositions"),
                         raise_for_failure=False)
    rp = ev.metadata.get("actionReturn", []) or []
    cands = []
    for p in rp:
        dx, dz = p["x"] - op["x"], p["z"] - op["z"]
        d = math.hypot(dx, dz)
        if d < MIN_RADIUS or d > MAX_RADIUS:
            continue
        cands.append((d, p))
    cands.sort()
    picked = []
    for d, p in cands:
        if all(math.hypot(p["x"] - q["x"], p["z"] - q["z"]) >= min_sep
               for _, q in picked):
            picked.append((d, p))
        if len(picked) >= n_angles:
            break
    return [p for _, p in picked]


def probe_visible(controller, obj_id: str, op, n_angles: int = 8) -> bool:
    """Check whether the object is visible from any reachable viewpoint."""
    vps = pick_viewpoints(controller, op, n_angles)
    for vp in vps[:4]:
        base = math.degrees(math.atan2(op["x"] - vp["x"], op["z"] - vp["z"]))
        for hor in (0, -25):
            ev = controller.step(dict(
                action="Teleport",
                position={"x": vp["x"], "y": vp["y"], "z": vp["z"]},
                rotation={"x": 0, "y": base, "z": 0},
                horizon=hor, standing=True), raise_for_failure=False)
            if not ev.metadata.get("lastActionSuccess"):
                continue
            masks = getattr(ev, "instance_masks", {}) or {}
            base = obj_id.split("|")[0]
            if any(k.split("|")[0] == base for k in masks):
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--objects", nargs="*", default=SMALL_TARGETS)
    ap.add_argument("--angles", type=int, default=8)
    ap.add_argument("--yaws", type=int, default=2)
    ap.add_argument("--horizons", type=int, default=2)
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--out", default="/mnt/d/lightwm_data_objviews")
    ap.add_argument("--gt-root", default="/mnt/d/lightwm_data")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    controller = ai2thor.controller.Controller(
        scene=args.scenes[0], width=800, height=600,
        renderDepthImage=True, renderInstanceSegmentation=True,
        visibilityDistance=20, gridSize=0.25)
    print("controller ready")
    os.makedirs(os.path.join(args.out, "episodes"), exist_ok=True)
    total = 0

    for si, scene in enumerate(args.scenes):
        if si > 0:
            controller.reset(scene=scene)
        objs = scene_objects(scene, args.gt_root)
        targets = [o for o in objs.values() if o["type"] in args.objects]
        # group by type, take up to N instances per type
        by_type = {}
        for o in targets:
            by_type.setdefault(o["type"], []).append(o)
        print(f"[{scene}] target instances: "
              + ", ".join(f"{t}:{len(v)}" for t, v in by_type.items()))
        for t, insts in by_type.items():
            for inst in insts[:args.instances]:
                if not probe_visible(controller, inst["objectId"],
                                     inst["position"],
                                     args.angles):
                    print(f"  {t}: not visible (inside container?), skipped")
                    continue
                ep_id = f"{scene}__objviews_{t}_{utcnow()}"
                ep_dir = os.path.join(args.out, "episodes", ep_id)
                frame_dir = os.path.join(ep_dir, "frames")
                os.makedirs(frame_dir, exist_ok=True)
                frames_meta = []
                step = 0
                op = inst["position"]
                viewpoints = pick_viewpoints(controller, op, args.angles)
                if not viewpoints:
                    print(f"  {t}: no reachable viewpoints, skipped")
                    continue
                for vp in viewpoints:
                    ax, ay, az = vp["x"], vp["y"], vp["z"]
                    # yaw facing the object (fwd=(sin,cos) convention)
                    base_yaw = math.degrees(math.atan2(
                        op["x"] - ax, op["z"] - az))
                    yaw_offsets = [0]   # always face the object
                    hor_list = [-25.0, 0.0, 25.0][:args.horizons]
                    for yoff in yaw_offsets:
                        for hor in hor_list:
                            ev = controller.step(dict(
                                action="Teleport",
                                position={"x": ax, "y": ay, "z": az},
                                rotation={"x": 0, "y": base_yaw + yoff, "z": 0},
                                horizon=hor, standing=True),
                                raise_for_failure=False)
                            if not ev.metadata.get("lastActionSuccess"):
                                continue
                            masks = getattr(ev, "instance_masks", {}) or {}
                            focus_visible = any(
                                k.split("|")[0] ==
                                inst["objectId"].split("|")[0] for k in masks)
                            if not focus_visible:
                                continue
                            stem = f"step_{step:05d}"
                            Image.fromarray(np.asarray(ev.frame)).save(
                                os.path.join(frame_dir, f"{stem}_rgb.png"))
                            depth = np.asarray(ev.depth_frame, dtype=np.float32)
                            Image.fromarray(
                                (depth * 1000).astype(np.uint16)).save(
                                os.path.join(frame_dir, f"{stem}_depth.png"))
                            Image.fromarray(np.asarray(
                                ev.instance_segmentation_frame)).save(
                                os.path.join(frame_dir, f"{stem}_seg.png"))
                            ag = ev.metadata["agent"]
                            frames_meta.append({
                                "step": step,
                                "rgb": f"frames/{stem}_rgb.png",
                                "depth": f"frames/{stem}_depth.png",
                                "seg": f"frames/{stem}_seg.png",
                                "action": {"name": "ObjView", "args": {}},
                                "action_success": True, "error_message": "",
                                "agent": {
                                    "position": ag["position"],
                                    "rotation": ag["rotation"],
                                    "cameraHorizon": ag.get(
                                        "cameraHorizon", hor),
                                },
                                "visible_objects": collect_visible(ev),
                                "focus_object": inst["name"],
                            })
                            step += 1
                with open(os.path.join(ep_dir, "episode.json"), "w") as f:
                    json.dump({
                        "episode_id": ep_id, "scene": scene,
                        "mode": "objviews", "focus_object_type": t,
                        "start_pose": None, "task": None,
                        "policy": "objviews", "width": 800, "height": 600,
                        "fov": 60.0, "ai2thor_version": "5.0.0",
                        "created_at": utcnow(),
                        "num_frames": len(frames_meta),
                        "frames": frames_meta,
                    }, f)
                with open(os.path.join(args.out, "manifest.jsonl"), "a") as f:
                    f.write(json.dumps({"episode_id": ep_id, "scene": scene,
                                        "mode": "objviews",
                                        "focus_object_type": t,
                                        "num_frames": len(frames_meta),
                                        "dir": ep_dir}) + "\n")
                total += len(frames_meta)
                print(f"  {t}: {len(frames_meta)} views -> {ep_id}")
        if args.smoke:
            break
    controller.stop()
    print(f"total views: {total}")


if __name__ == "__main__":
    main()
