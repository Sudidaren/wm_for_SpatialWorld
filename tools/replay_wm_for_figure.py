#!/usr/bin/env python3
"""Replay a recorded episode through the *current* world model, for figures.

Why this exists: the screenshots on disk come with the prompt text the runtime
printed *at the time*, and that rendering has changed since (the old probe
quoted the simulator's own error strings, injected a long 建议 block, and
printed 手持：det).  A figure must show what the code does *now*, so this
script takes the recorded frames plus the recorded actions and runs them
through the shipped WorldModel + MemoryProbe, emitting:

  * ``step_NN_prompt.txt``  -- the block injected in front of the image
  * ``step_NN_det.png``     -- the frame with the WM's own detections drawn
  * ``step_NN_meta.json``   -- the WM metadata for that step (anchors, sigma)
  * ``map.png``             -- top-down view of the anchors + dead-reckoned path

Usage:
    python3 tools/replay_wm_for_figure.py --task ai2thor03017 \
        --run wingman_wm_qwen3vl30b_438_v1 --out /mnt/d/wm_figure
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SW = Path("/home/sudidaren/SpatialWorld")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SW))

RUNS = Path("/home/sudidaren/spatialworld_eval/runs")


def episode_dir(run: str, task: str) -> Path:
    hits = sorted(RUNS.glob(f"{run}/ai2thor/{task}/worker_*/{task}/log.json"))
    if not hits:
        raise SystemExit(f"no log.json for {run}/{task}")
    # the newest attempt is the one whose images list matches the episode
    return hits[-1].parent


def load_episode(task: str, run: str):
    d = episode_dir(run, task)
    log = json.loads((d / "log.json").read_text())
    images = log.get("images") or []
    actions = []
    for msg in log.get("messages") or []:
        if msg.get("role") != "assistant":
            continue
        c = msg.get("content")
        text = c if isinstance(c, str) else " ".join(
            x.get("text", "") for x in c if isinstance(x, dict))
        for name, arg in re.findall(
                r"<ACTIONS?>\s*\[?\s*([A-Za-z]+)\s*\(?\s*([A-Za-z]*)", text):
            actions.append({"action_name": name, "object_type": arg})
            break
        else:
            actions.append({"action_name": "DONE"})
    return d, log, images, actions


def draw_detections(frame_path: str, dets, out_path: Path, title: str = ""):
    im = Image.open(frame_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        dr.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
        label = f"{det['type']} {det['score']:.2f}"
        tw = dr.textlength(label)
        dr.rectangle([x1, max(0, y1 - 16), x1 + tw + 6, y1], fill=(255, 80, 80))
        dr.text((x1 + 3, max(0, y1 - 15)), label, fill=(255, 255, 255))
    im.save(out_path)
    return out_path


def draw_map(slots, pose, out_path: Path, scale_px: float = 60.0):
    """Top-down view in the world model's own (dead-reckoned) frame."""
    W = H = 460
    im = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    cx, cy = W / 2, H / 2

    def to_px(x, z):
        return cx + x * scale_px, cy - z * scale_px

    x, _y, z, yaw = pose
    fx, fy = to_px(x, z)
    dr.line([cx, cy, fx, fy], fill=(120, 120, 120), width=3)
    dr.ellipse([fx - 5, fy - 5, fx + 5, fy + 5], fill=(60, 60, 60))
    head = 0.35 * scale_px
    import math
    dr.line([fx, fy, fx + head * math.sin(math.radians(yaw)),
             fy - head * math.cos(math.radians(yaw))], fill=(60, 60, 60), width=4)
    for oid, slot in sorted(slots.items()):
        px, py = to_px(slot["pos"][0], slot["pos"][2])
        r = 4 + 26 * min(1.0, float(slot.get("sigma") or 0.3))
        dr.ellipse([px - r, py - r, px + r, py + r],
                   outline=(70, 110, 200), width=2)
        dr.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(70, 110, 200))
        dr.text((px + 6, py - 14), str(slot.get("type", "?")), fill=(20, 20, 20))
    # 1 m scale bar
    y0 = H - 24
    dr.line([20, y0, 20 + scale_px, y0], fill=(0, 0, 0), width=3)
    dr.text((24, y0 - 14), "1 m", fill=(0, 0, 0))
    im.save(out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ai2thor03017")
    ap.add_argument("--run", default="wingman_wm_qwen3vl30b_438_v1")
    ap.add_argument("--out", default="/mnt/d/wm_figure")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/small_objects_20260910/dense_depth_best.pt"))
    ap.add_argument("--detector", default=str(ROOT / "checkpoints/rfdetr_small_228094/checkpoint_best_total.pth"))
    ap.add_argument("--target-hint", action="store_true",
                    help="also render the optional WM_TARGET_HINT block")
    args = ap.parse_args()

    out = Path(args.out) / args.task
    out.mkdir(parents=True, exist_ok=True)

    d, log, images, actions = load_episode(args.task, args.run)
    print(f"episode : {d}")
    print(f"指令    : {log['metadata']['task_description']}")
    print(f"步数    : {log['metadata']['total_steps']}  结果 {log['metadata']['task_result']}")
    print(f"帧      : {len(images)}  动作: {len(actions)}")

    from phase_b.runtime_factory import build_runtime
    from mllm_base_agent.agent.world_model import WorldModel
    from mllm_base_agent.agent.memory_probe import MemoryProbe

    runtime = build_runtime({
        "perception_ckpt": args.ckpt, "detector_backend": "rfdetr_small_depth",
        "detector_path": args.detector, "obj_thr": 0.40,
        "perception_runtime_root": str(ROOT), "device": "cpu",
    })
    wm = WorldModel()
    wm.attach_perception(runtime)
    probe = MemoryProbe(task_description=log["metadata"]["task_description"])
    if args.target_hint:
        probe.set_target_hint({"enabled": True, "limit": 6, "names_limit": 5,
                               "max_dist": 3.0, "max_sigma": 0.5})

    prev = None
    rows = []
    for i, (frame_path, action) in enumerate(zip(images, actions), start=1):
        frame = np.asarray(Image.open(frame_path).convert("RGB"))
        rgb = frame
        res = runtime(rgb)
        draw_detections(frame_path, res["detections"],
                        out / f"step_{i:02d}_det.png")
        if prev is None:
            moved, ok = None, None
        else:
            ok = bool(np.mean((frame.astype(float) - prev.astype(float)) ** 2) > 1.0)
            moved = ok
        raw_meta = wm.observe(None, action=action, moved=moved, action_ok=ok,
                              frame=rgb)
        block = probe.update(
            wm_metadata=raw_meta,
            action_name=action.get("action_name"),
            object_type=action.get("object_type"),
            blocked=(moved is False and action.get("action_name", "").startswith("Move")),
            action_ok=(raw_meta.get("action_outcome") or {}).get("ok"),
        )
        (out / f"step_{i:02d}_prompt.txt").write_text(block, encoding="utf-8")
        meta = {k: v for k, v in raw_meta.items() if k != "objects"}
        meta["objects"] = [
            {k: o.get(k) for k in ("objectType", "visible", "distance", "sigma",
                                   "last_seen_step", "seen_count", "position")}
            for o in raw_meta.get("objects") or []
        ]
        (out / f"step_{i:02d}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append((i, action.get("action_name"), action.get("object_type"),
                     len(res["detections"]), block))
        prev = frame
        print(f"step {i:2d} {action.get('action_name'):18s} dets={len(res['detections']):2d} "
              f"| {block.splitlines()[-1][:90] if block else ''}")

    draw_map(wm._slots, wm._pose or [0, 0, 0, 0], out / "map.png")
    (out / "all_prompts.txt").write_text(
        "\n\n".join(f"=== step {i}  {a}({o}) ===\n{b}"
                    for i, a, o, _n, b in rows), encoding="utf-8")
    print(f"\n输出目录: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
