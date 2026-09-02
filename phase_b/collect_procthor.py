"""Collect LightWM-format data from ProcTHOR-10K houses.

ProcTHOR uses the same ai2thor Controller as AI2-THOR, so the data format
(RGB / depth / instance seg / bbox / agent pose) is identical.  We reuse the
coverage strategy: every reachable position x 4 yaws x 3 horizons.

Usage (WSLg display, envs/ai2thor venv):
  DISPLAY=:0 python phase_b/collect_procthor.py --houses 20 --out /mnt/d/lightwm_data_procthor
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


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", type=int, default=20)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--yaws", type=int, default=2)
    ap.add_argument("--horizons", type=int, default=2)
    ap.add_argument("--position-step", type=int, default=3,
                    help="subsample reachable positions (every Nth)")
    ap.add_argument("--out", default="/mnt/d/lightwm_data_procthor")
    args = ap.parse_args()

    import prior
    ds = prior.load_dataset("procthor-10k", offline=True)
    houses = getattr(ds, args.split)
    os.makedirs(os.path.join(args.out, "episodes"), exist_ok=True)
    total = 0

    for hi in range(args.start, min(args.start + args.houses, len(houses))):
        house = houses[hi]
        controller = ai2thor.controller.Controller(
            scene=house, width=800, height=600,
            renderDepthImage=True, renderInstanceSegmentation=True,
            visibilityDistance=20, gridSize=0.25)
        ev = controller.step(dict(action="GetReachablePositions"),
                             raise_for_failure=False)
        rp = ev.metadata.get("actionReturn", []) or []
        if len(rp) < 20:
            controller.stop()
            print(f"house {hi}: only {len(rp)} reachable positions, skip")
            continue
        ep_id = f"procthor_{args.split}_{hi:04d}"
        ep_dir = os.path.join(args.out, "episodes", ep_id)
        frame_dir = os.path.join(ep_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        frames_meta = []
        step = 0
        yaw_list = [i * 360.0 / args.yaws for i in range(args.yaws)]
        hor_list = [-25.0, 0.0, 25.0][:args.horizons]
        for p in rp[::args.position_step]:
            for yaw in yaw_list:
                for hor in hor_list:
                    ev = controller.step(dict(
                        action="Teleport",
                        position={"x": p["x"], "y": p["y"], "z": p["z"]},
                        rotation={"x": 0, "y": yaw, "z": 0},
                        horizon=hor, standing=True), raise_for_failure=False)
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
                        "action_success": True, "error_message": "",
                        "agent": {
                            "position": ag["position"],
                            "rotation": ag["rotation"],
                            "cameraHorizon": ag.get("cameraHorizon", hor),
                        },
                        "visible_objects": collect_visible(ev),
                    })
                    step += 1
        controller.stop()
        with open(os.path.join(ep_dir, "episode.json"), "w") as f:
            json.dump({
                "episode_id": ep_id, "scene": f"procthor-{args.split}-{hi}",
                "mode": "procthor_coverage", "task": None,
                "policy": "procthor_coverage", "width": 800, "height": 600,
                "fov": 60.0, "ai2thor_version": "5.0.0",
                "created_at": utcnow(), "num_frames": len(frames_meta),
                "frames": frames_meta,
            }, f)
        total += len(frames_meta)
        print(f"house {hi}: {len(frames_meta)} frames -> {ep_id}",
              flush=True)
    print(f"total frames: {total}")


if __name__ == "__main__":
    main()
