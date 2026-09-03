"""Collect LightWM-format coverage data from VirtualHome (Unity backend).

VirtualHome exposes no per-frame depth in usable units, so each frame keeps
RGB + class segmentation; visible objects (type/bbox/position) are derived
from instance segmentation + the environment graph.  Depth supervision is
not applicable to this source.

Usage (backend must already run, e.g.):
  DISPLAY=:0 <unity_simulator>/linux_exec.v2.2.4.x86_64 \
      -windowed -screen-width 960 -screen-height 540 -http-port 8080

  VIRTUALHOME_VENV_PYTHON phase_b/collect_virtualhome.py \
      --out /mnt/d/lightwm_data_virtualhome --scenes 0 1 2 3 5 6 15 17 20 40 \
      --anchors-per-scene 2
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import glob
import json
import math
import os
import sys
import time

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping

import numpy as np
np.fromstring = np.frombuffer  # VirtualHome decode compat shim

from PIL import Image

sys.path.insert(0, "/home/sudidaren/SpatialWorld/envs/virtualhome/.venv/"
                 "lib/python3.12/site-packages")
from virtualhome.simulation.unity_simulator.comm_unity import (  # noqa: E402
    UnityCommunication,
)

PITCH_CAMS = [
    (-60, "wasd_fp_u60"),
    (-30, "wasd_fp_u30"),
    (0, "wasd_fp"),
    (30, "wasd_fp_d30"),
    (60, "wasd_fp_d60"),
]
EXCLUDE_CATEGORIES = {
    "Rooms", "Floor", "Floors", "Walls", "Ceiling", "Ceilings",
    "Doors", "Windows", "Characters", "Lights",
}
MIN_VISIBLE_PIXELS = 25


def post_reset(comm, scene: int) -> bool:
    return comm.post_command(
        {"id": "reset", "action": "reset", "intParams": [int(scene)]}
    )["success"]


def setup_cameras(comm) -> list[int]:
    _, raw = comm.character_cameras()
    existing = raw if isinstance(raw, list) else []
    for pitch, cname in PITCH_CAMS:
        if cname not in existing:
            comm.add_character_camera(
                position=[0, 1.8, 0.15], rotation=[pitch, 0, 0],
                field_view=60, name=cname)
    _, total = comm.camera_count()
    return list(range(total - len(PITCH_CAMS), total))


def quat_yaw_deg(q) -> float:
    try:
        _, y, _, w = [float(v) for v in (q or [0, 0, 0, 1])]
    except Exception:
        return 0.0
    return math.degrees(2.0 * math.atan2(y, w)) % 360.0


def graph_char(graph) -> dict | None:
    for n in (graph or {}).get("nodes", []) or []:
        if n.get("category") == "Characters" and \
                n.get("class_name") == "character":
            return n
    return None


def color_to_id_map(raw: dict) -> dict:
    out = {}
    for key, val in (raw or {}).items():
        try:
            rgb = tuple(int(round(float(c) * 255)) for c in val[:3])
            out[rgb] = int(key)
        except Exception:
            continue
    return out


def visible_from_seg(segi: np.ndarray, color2id, id2node) -> list[dict]:
    """bbox/type/position per visible, non-structural object in seg_inst."""
    frame = np.asarray(segi)[:, :, :3][:, :, ::-1].astype(np.uint8)
    colors, counts = np.unique(frame.reshape(-1, 3), axis=0,
                               return_counts=True)
    out = []
    for color, count in zip(colors, counts):
        rgb = (int(color[0]), int(color[1]), int(color[2]))
        oid = color2id.get(rgb)
        if oid is None or int(count) < MIN_VISIBLE_PIXELS:
            continue
        node = id2node.get(oid)
        if not node or node.get("category") in EXCLUDE_CATEGORIES:
            continue
        mask = (frame[:, :, 0] == rgb[0]) & (frame[:, :, 1] == rgb[1]) \
            & (frame[:, :, 2] == rgb[2])
        ys, xs = np.nonzero(mask)
        tr = (node.get("obj_transform") or {})
        out.append({
            "name": node.get("class_name", ""),
            "type": node.get("class_name", ""),
            "category": node.get("category", ""),
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max()), int(ys.max())],
            "position": tr.get("position"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/d/lightwm_data_virtualhome")
    ap.add_argument("--port", default="8080")
    ap.add_argument("--scenes", nargs="*", type=int,
                    default=[0, 1, 2, 3, 5, 6, 15, 17, 20, 40])
    ap.add_argument("--anchors-per-scene", type=int, default=2)
    ap.add_argument("--tasks-root",
                    default="/home/sudidaren/SpatialWorld/data/"
                            "virtualhome/tasks")
    ap.add_argument("--yaws", type=int, default=12)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    args = ap.parse_args()

    # anchor positions from task init.json (per scene)
    anchors: dict[int, list[list[float]]] = {}
    for p in sorted(glob.glob(os.path.join(args.tasks_root, "*", "init.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        s = int(d.get("scene", -1))
        pos = d.get("character_position")
        if s >= 0 and isinstance(pos, list) and len(pos) >= 3:
            anchors.setdefault(s, []).append([float(v) for v in pos[:3]])

    comm = UnityCommunication(port=args.port)
    ep_root = os.path.join(args.out, "episodes")
    gt_root = os.path.join(args.out, "scene_gt")
    os.makedirs(ep_root, exist_ok=True)
    os.makedirs(gt_root, exist_ok=True)
    resource = ["Chars/Female1", "Chars/Male1"]
    total_frames = 0

    for scene in args.scenes:
        positions = anchors.get(scene, [])[: args.anchors_per_scene]
        if not positions:
            print(f"scene {scene}: no anchors, skip")
            continue
        for ai, pos in enumerate(positions):
            post_reset(comm, scene)
            time.sleep(1.0)
            # capture scene_gt from the first anchor of each scene
            ep_id = f"virtualhome_{scene:02d}__a{ai:02d}"
            gt_path = os.path.join(gt_root, f"virtualhome-{scene}.json")
            if ai == 0 and not os.path.exists(gt_path):
                okg, g0 = comm.environment_graph()
                objs = []
                for n in (g0.get("nodes", []) or []):
                    if n.get("category") in EXCLUDE_CATEGORIES:
                        continue
                    tr = n.get("obj_transform") or {}
                    objs.append({
                        "id": n.get("id"),
                        "name": n.get("class_name", ""),
                        "category": n.get("category", ""),
                        "position": tr.get("position"),
                    })
                with open(gt_path, "w") as f:
                    json.dump({"scene": f"virtualhome-{scene}",
                               "objects": objs}, f)
            ok_add = comm.add_character(resource[ai % 2])
            time.sleep(1.0)
            if ok_add:
                try:
                    comm.move_character(0, pos)
                    time.sleep(0.5)
                except Exception:
                    pass
            ok, graph = comm.environment_graph()
            if not ok:
                print(f"scene {scene} anchor {ai}: graph fail, skip")
                continue
            id2node = {n.get("id"): n for n in (graph.get("nodes", []) or [])}
            ch = graph_char(graph)
            if ch is None:
                print(f"scene {scene} anchor {ai}: no character node, skip")
                continue
            cam_ids = setup_cameras(comm)
            okc, colors = comm.instance_colors()
            color2id = color_to_id_map(colors)
            ep_dir = os.path.join(ep_root, ep_id)
            frame_dir = os.path.join(ep_dir, "frames")
            if os.path.isdir(frame_dir) and \
                    os.path.isfile(os.path.join(ep_dir, "episode.json")):
                print(f"{ep_id}: exists, skip")
                continue
            os.makedirs(frame_dir, exist_ok=True)
            frames_meta = []
            step = 0
            for ys in range(args.yaws):
                ok, graph = comm.environment_graph()
                ch = graph_char(graph)
                char_pos = ((ch or {}).get("obj_transform") or {}).get(
                    "position")
                yaw = quat_yaw_deg(((ch or {}).get("obj_transform") or {})
                                   .get("rotation"))
                try:
                    okn, imgs_n = comm.camera_image(
                        cam_ids, mode="normal",
                        image_width=args.width, image_height=args.height)
                    oks, imgs_s = comm.camera_image(
                        cam_ids, mode="seg_class",
                        image_width=args.width, image_height=args.height)
                    oki, imgs_i = comm.camera_image(
                        cam_ids, mode="seg_inst",
                        image_width=args.width, image_height=args.height)
                except Exception as exc:
                    print(f"{ep_id} yaw {ys}: render error {exc}")
                    break
                for k, (pitch, _cname) in enumerate(PITCH_CAMS):
                    stem = f"step_{step:05d}"
                    rgb = np.asarray(imgs_n[k])
                    seg = np.asarray(imgs_s[k])[:, :, ::-1]
                    Image.fromarray(rgb).save(
                        os.path.join(frame_dir, f"{stem}_rgb.png"))
                    Image.fromarray(seg).save(
                        os.path.join(frame_dir, f"{stem}_seg.png"))
                    vis = visible_from_seg(imgs_i[k], color2id, id2node)
                    frames_meta.append({
                        "step": step,
                        "rgb": f"frames/{stem}_rgb.png",
                        "depth": None,
                        "seg": f"frames/{stem}_seg.png",
                        "action": {"name": "VHCoverage",
                                   "args": {"yaw_step": ys, "pitch": pitch}},
                        "action_success": True,
                        "error_message": "",
                        "agent": {
                            "position": char_pos,
                            "rotation": {"x": pitch, "y": yaw, "z": 0.0},
                            "cameraHorizon": pitch,
                        },
                        "visible_objects": vis,
                    })
                    step += 1
                if ys < args.yaws - 1:
                    comm.render_script(["<char0> [TurnLeft]"],
                                       skip_animation=True)
                time.sleep(0.05)
            with open(os.path.join(ep_dir, "episode.json"), "w") as f:
                json.dump({
                    "episode_id": ep_id,
                    "scene": f"virtualhome-{scene}",
                    "mode": "virtualhome_coverage",
                    "task": None,
                    "policy": "virtualhome_coverage",
                    "width": args.width, "height": args.height,
                    "fov": 60.0,
                    "created_at": time.strftime("%Y%m%d_%H%M%S"),
                    "num_frames": len(frames_meta),
                    "frames": frames_meta,
                }, f)
            total_frames += len(frames_meta)
            print(f"{ep_id}: {len(frames_meta)} frames -> done", flush=True)
    print(f"total frames: {total_frames}", flush=True)


if __name__ == "__main__":
    main()
