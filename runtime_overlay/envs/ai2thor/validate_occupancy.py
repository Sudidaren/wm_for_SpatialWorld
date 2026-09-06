#!/usr/bin/env python3
"""Validate the world model's self-built occupancy grid (no GPT-5 / no oracle
map at runtime). Drives ai2thor03073's golden path while the WorldModel builds
its map from seg+depth, then compares against GetReachablePositions.

Run:
  envs/ai2thor/.venv/bin/python -u envs/ai2thor/validate_occupancy.py
"""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from actions.parser import parse_action_string
from mllm_base_agent.agent.world_model import WorldModel
from mllm_base_agent.environments.ai2thor.wrapper import AI2ThorEnvWrapper

TASK_ID = "ai2thor03073"
W, H = 512, 384


def _rss_gb() -> float:
    """Current process RSS in GiB, read from /proc (no psutil dependency)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0


def main() -> None:
    task = json.load(
        open(f"/home/sudidaren/SpatialWorld/data/ai2thor/tasks/{TASK_ID}/task.json")
    )
    init = json.load(
        open(f"/home/sudidaren/SpatialWorld/data/ai2thor/tasks/{TASK_ID}/init.json")
    )["actions"]
    golden = task["golden_actions"]["actions"]

    env = AI2ThorEnvWrapper(
        scene=task["scene"],
        width=W,
        height=H,
        render_depth_image=True,
        render_instance_segmentation=True,
        output_dir="/tmp/occ_validate",
    )
    try:
        print(f"[mem] env created: RSS={_rss_gb():.2f}GB", flush=True)
        env.configure_task(
            target_object_types=["Potato"],
            success_predicate=lambda o: False,
            target_description="x",
        )
        obs = env.reset(task["instruction"], scene=task["scene"])
        for s in init:
            if s.strip().upper() == "DONE":
                break
            obs, err = env.step_with_action_dict(parse_action_string(s))

        ckpt_dir = "/tmp/occ_validate"
        ckpt_path = os.path.join(ckpt_dir, "world_model_ckpt.json")
        wm = WorldModel(
            targets=["Potato", "Plate", "Microwave"],
            fov=60,
            width=W,
            height=H,
            pose_from_action_log=True,
            pose_initial="sim",
            checkpoint_dir=ckpt_dir,
        )
        if os.path.exists(ckpt_path) and os.environ.get("WORLD_MODEL_RESUME") == "1":
            try:
                wm = WorldModel.load(
                    ckpt_path,
                    targets=["Potato", "Plate", "Microwave"],
                    fov=60,
                    width=W,
                    height=H,
                    pose_from_action_log=True,
                    pose_initial="sim",
                    checkpoint_dir=ckpt_dir,
                )
                print(f"↻ Resumed world model from {ckpt_path} (step {wm._step})")
            except Exception as exc:
                print(f"⚠️ Checkpoint load failed, starting fresh: {exc}")

        for i, a in enumerate(golden, 1):
            ad = parse_action_string(a)
            obs, err = env.step_with_action_dict(ad)
            wm.observe(
                obs,
                {
                    "action_name": ad.get("action_name"),
                    "object_type": ad.get("object_type"),
                },
                err,
                env,
            )
            if i % 10 == 0:
                print(f"[mem] step {i}: RSS={_rss_gb():.2f}GB", flush=True)

        # oracle walkability (validation only)
        ev = env.controller.step(action="GetReachablePositions")
        oracle = {
            (round(p["x"] / 0.25), round(p["z"] / 0.25))
            for p in (ev.metadata.get("actionReturn") or [])
            if "x" in p and "z" in p
        }
    finally:
        env.close()

    static = wm.static_cells()
    print(f"visited cells        : {len(wm._visited)}", flush=True)
    print(f"occupancy cells      : {len(wm._occupancy)}", flush=True)
    print(f"static (>=2) cells   : {len(static)}", flush=True)
    walk_est = wm.reachable_cells()

    # region of interest = visited bbox expanded by 4 cells (1m)
    xs = [c[0] for c in wm._visited]
    zs = [c[1] for c in wm._visited]
    x0, x1 = min(xs) - 4, max(xs) + 4
    z0, z1 = min(zs) - 4, max(zs) + 4
    region = {(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)}

    not_walkable = region - oracle
    obs_region = region & (set(wm._occupancy) | wm._visited | oracle)

    region_static = static & region  # only judge static cells inside the region
    tp = region_static & not_walkable
    fp = region_static & oracle
    fn = not_walkable - region_static
    print(f"oracle walkable      : {len(oracle)}")
    print(f"walkable estimate    : {len(walk_est)}")
    print(f"region cells         : {len(region)} (visited bbox ±4)")
    print(f"static in region     : {len(region_static)}")
    print()
    print("== obstacle detection ==")
    print(f"  precision: {len(tp)}/{len(region_static)} = {100*len(tp)/max(len(region_static),1):.1f}%")
    print(f"  recall   : {len(tp)}/{len(fn)+len(tp)} = {100*len(tp)/max(len(fn)+len(tp),1):.1f}%")
    print(f"  false positives (walkable marked static): {len(fp)}")
    if fp:
        print("   FP cells:", sorted(fp)[:20])
    if fn:
        print("   missed obstacles:", sorted(fn)[:20])
    print()
    print("== walkable estimate ==")
    wp = walk_est & oracle
    print(f"  precision: {len(wp)}/{len(walk_est)} = {100*len(wp)/max(len(walk_est),1):.1f}%")
    cover = walk_est & oracle & obs_region
    print(f"  coverage over visited region: {len(cover)}/{len(oracle & obs_region)} = {100*len(cover)/max(len(oracle & obs_region),1):.1f}%")

    # ASCII map overlay: # = detected obstacle (TP), ! = false positive
    # (oracle says walkable), O = missed obstacle (oracle says obstacle but
    # not detected), . = walkable estimate, ? = unknown, + visited, S = start.
    label = {}
    for c in region:
        if c in static:
            label[c] = "#" if c in not_walkable else "!"
        elif c in not_walkable:
            label[c] = "O"
        elif c in walk_est:
            label[c] = "."
        else:
            label[c] = "?"
    start_cell = list(wm._visited)[0] if wm._visited else None
    for c in wm._visited:
        label[c] = "S" if c == start_cell else "+"
    print("\n== ASCII map (top row = max z) ==")
    print("   # detected obstacle | ! false positive | O missed obstacle")
    print("   . walkable-est      | ? unknown        | + visited / S start")
    for z in range(z1, z0 - 1, -1):
        row = "".join(label.get((x, z), "?") for x in range(x0, x1 + 1))
        print(row)


if __name__ == "__main__":
    main()
