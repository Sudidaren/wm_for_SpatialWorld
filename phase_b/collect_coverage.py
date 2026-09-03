"""Exhaustive coverage collector for AI2-THOR.

For each scene, enumerates every reachable grid cell (Teleport-tested) and
renders it under {yaws} x {horizons}: RGB + depth + instance segmentation +
metadata (pose, visible objects with 2D bbox + states + positions).

This gives the perception/depth/feasibility heads dense multi-view coverage
of every position the agent can be in, exactly as the user requested.

Output layout (same as collect_data.py so the frame index picks it up):
  <out>/episodes/<scene>__cov_<ts>/episode.json + frames/step_*.png
  <out>/scene_gt/<scene>.json

Usage (run with the envs/ai2thor venv + xvfb):
  xvfb-run -a envs/ai2thor/.venv/bin/python phase_b/collect_coverage.py \
    --scenes FloorPlan1 FloorPlan2 FloorPlan3 FloorPlan10 \
    --step 0.5 --yaws 4 --horizons 3 --out /mnt/d/lightwm_data_cov
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image

import ai2thor.controller


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def collect_visible(event) -> list:
    """Objects with a real instance mask -> type, bbox, meta position."""
    masks = getattr(event, "instance_masks", None) or {}
    meta_objs = {o["objectId"]: o for o in event.metadata["objects"]}
    out = []
    for oid, mask in masks.items():
        if mask is None or not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        mo = meta_objs.get(oid, {})
        if not mo:
            continue  # non-interactable asset objects (walls, doors, lights)
        name = mo.get("name", oid)
        # AI2-THOR exposes objectType; ProcTHOR exposes 'type' with the
        # canonical class ('Television', 'Wall', ...).  Prefer those over
        # deriving from instance names ('Television|6|2|1').
        otype = (mo.get("objectType") or mo.get("type")
                 or (name.split("_")[0] if "_" in name else name))
        out.append({
            "objectId": oid,
            "name": name,
            "type": otype,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "position": mo.get("position"),
            "states": {k: mo.get(k) for k in
                       ("isOpen", "isToggled", "isPickedUp", "isSliced",
                        "isBroken") if k in mo},
        })
    return out


def save_scene_gt(controller, scene: str, out_root: str) -> None:
    gt_dir = os.path.join(out_root, "scene_gt")
    os.makedirs(gt_dir, exist_ok=True)
    path = os.path.join(gt_dir, f"{scene}.json")
    if os.path.exists(path):
        return
    objs = []
    for o in controller.last_event.metadata["objects"]:
        objs.append({
            "objectId": o["objectId"], "name": o["name"],
            "position": o.get("position"),
            "flags": {k: o.get(k) for k in
                      ("openable", "pickupable", "toggleable", "sliceable",
                       "breakable", "canFillWithLiquid", "isReceptacle")},
        })
    with open(path, "w") as f:
        json.dump({"scene": scene, "objects": objs}, f, indent=1)
    print(f"  scene_gt saved: {path} ({len(objs)} objects)")


def floor_height(controller) -> float:
    """Auto-detect the walkable floor level from Floor objects."""
    ys = [o["position"]["y"] for o in controller.last_event.metadata["objects"]
          if o.get("objectType") == "Floor"]
    return min(ys) if ys else 0.0


def probe_position(controller, x, z, y_candidates) -> Optional[float]:
    """Return the first teleport y that works at (x,z), else None."""
    for y in y_candidates:
        ev = controller.step(dict(
            action="Teleport",
            position={"x": x, "y": y, "z": z},
            rotation={"x": 0, "y": 0, "z": 0},
            horizon=0, standing=True,
        ), raise_for_failure=False)
        if ev.metadata.get("lastActionSuccess"):
            return y
    return None


def reachable_positions(controller, step: float) -> List[tuple]:
    """Use AI2-THOR's GetReachablePositions (one call) instead of probing
    every candidate cell.  Returns [(x, z, y), ...] at the requested step."""
    ev = controller.step(dict(action="GetReachablePositions"),
                         raise_for_failure=False)
    if not ev.metadata.get("lastActionSuccess"):
        return []
    raw = ev.metadata.get("actionReturn") or []
    out = []
    for p in raw:
        x, y, z = p["x"], p["y"], p["z"]
        # keep only positions on the requested lattice (0.5m) to bound size
        if abs(round(x / step) * step - x) < 0.01 and \
           abs(round(z / step) * step - z) < 0.01:
            out.append((round(x, 3), round(z, 3), round(y, 3)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["FloorPlan1"])
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--yaws", type=int, default=4)
    ap.add_argument("--horizons", type=int, default=3)
    ap.add_argument("--grid", type=float, default=6.0,
                    help="half-extent of the candidate grid (meters)")
    ap.add_argument("--out", default="/mnt/d/lightwm_data_cov")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    yaw_list = [i * 360.0 / args.yaws for i in range(args.yaws)]
    if args.horizons == 1:
        hor_list = [0.0]
    elif args.horizons == 2:
        hor_list = [-30.0, 30.0]
    else:
        hor_list = [-30.0, 0.0, 30.0]

    controller = ai2thor.controller.Controller(
        scene=args.scenes[0],
        width=800, height=600,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        visibilityDistance=1.0,
        gridSize=0.25,
    )
    print("controller ready")

    os.makedirs(os.path.join(args.out, "episodes"), exist_ok=True)
    all_frames = 0
    for scene in args.scenes:
        if scene != args.scenes[0]:
            controller.reset(scene=scene)
        save_scene_gt(controller, scene, args.out)
        fy = floor_height(controller)
        y_cands = sorted({0.9, round(fy + 0.9, 2), round(fy + 0.5, 2),
                          round(fy + 1.2, 2)})
        print(f"  {scene}: floor y={fy:.2f}, teleport heights={y_cands}")
        ep_id = f"{scene}__cov_{utcnow()}"
        ep_dir = os.path.join(args.out, "episodes", ep_id)
        frame_dir = os.path.join(ep_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        # all reachable positions from the nav mesh, subsampled to args.step
        cells = reachable_positions(controller, args.step)
        print(f"  {scene}: {len(cells)} reachable cells (of "
              f"~{int(round(args.grid*2/args.step)+1)**2} candidates)")
        if args.max_cells:
            cells = cells[: args.max_cells]

        frames_meta = []
        step = 0
        t0 = time.time()
        for (x, z, y0) in cells:
            for yaw in yaw_list:
                for hor in hor_list:
                    ev = controller.step(dict(
                        action="Teleport",
                        position={"x": x, "y": y0, "z": z},
                        rotation={"x": 0, "y": yaw, "z": 0},
                        horizon=hor,
                        standing=True,
                    ), raise_for_failure=False)
                    if not ev.metadata.get("lastActionSuccess"):
                        continue
                    stem = f"step_{step:05d}"
                    Image.fromarray(np.asarray(ev.frame)).save(
                        os.path.join(frame_dir, f"{stem}_rgb.png"))
                    depth = np.asarray(ev.depth_frame, dtype=np.float32)
                    Image.fromarray((depth * 1000).astype(np.uint16)).save(
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
                        "action": {"name": "Coverage", "args": {}},
                        "action_success": True,
                        "error_message": "",
                        "agent": {
                            "position": ag["position"],
                            "rotation": ag["rotation"],
                            "cameraHorizon": ag.get("cameraHorizon", hor),
                        },
                        "visible_objects": collect_visible(ev),
                    })
                    step += 1
            if step % 500 == 0 and step:
                print(f"    {step} frames ({time.time()-t0:.0f}s)")
        ep_json = {
            "episode_id": ep_id, "scene": scene, "mode": "coverage",
            "start_pose": cells[0] if cells else None,
            "task": None, "policy": "coverage",
            "width": 800, "height": 600, "fov": 60.0,
            "ai2thor_version": "5.0.0",
            "created_at": utcnow(), "num_frames": len(frames_meta),
            "frames": frames_meta,
        }
        with open(os.path.join(ep_dir, "episode.json"), "w") as f:
            json.dump(ep_json, f)
        with open(os.path.join(args.out, "manifest.jsonl"), "a") as f:
            f.write(json.dumps({"episode_id": ep_id, "scene": scene,
                                "mode": "coverage",
                                "num_frames": len(frames_meta),
                                "dir": ep_dir}) + "\n")
        print(f"  {scene}: saved {len(frames_meta)} frames to {ep_dir} "
              f"({time.time()-t0:.0f}s)")
        all_frames += len(frames_meta)
        if args.smoke:
            break
    controller.stop()
    print(f"total frames: {all_frames}")


if __name__ == "__main__":
    main()
