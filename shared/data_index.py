"""Build and cache a flat index over all LightWM episodes/frames.

The index is a list of dicts (one per frame) with:
  episode, scene, mode, step, rgb, depth, seg, action, success,
  error_class, agent (position/yaw), visible (name, type, bbox),
  plus global vocabularies (object types, actions, error classes).

Used by phase B (perception/depth/feasibility training), phase B evaluation
(spatial memory) and phase C (gate candidate extraction).
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional

EPISODES_GLOBS = [
    "/mnt/d/lightwm_data/episodes/*/episode.json",
    "/mnt/d/lightwm_data_cov/episodes/*/episode.json",
]
SCENE_GT_GLOBS = [
    "/mnt/d/lightwm_data/scene_gt/*.json",
    "/mnt/d/lightwm_data_cov/scene_gt/*.json",
]
DEFAULT_INDEX = "/home/sudidaren/lightwm_phases/data/frame_index.pkl"

_ERROR_RULES = [
    ("hand_occupied", ["hand already has an object", "hand is already holding"]),
    ("not_holding", ["not holding anything", "isn't holding", "agent isn't holding"]),
    ("look_limit", ["can't look down beyond", "can't look up beyond", "look up beyond", "look down beyond"]),
    ("not_in_view", ["not in view", "isn't visible", "not visible"]),
    ("blocked", ["blocking agent", "blocked by"]),
]


def classify_error(err: Optional[str]) -> str:
    e = (err or "").lower()
    for cls, pats in _ERROR_RULES:
        if any(p in e for p in pats):
            return cls
    return "other" if e else "none"


def base_type(name: str) -> str:
    return name.split("_")[0] if "_" in name else name


def _load_episode(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def build_index(out_path: str = DEFAULT_INDEX) -> Dict[str, Any]:
    episodes = sorted({p for g in EPISODES_GLOBS for p in glob.glob(g)})
    frames: List[Dict] = []
    type_counts: Dict[str, int] = {}
    action_counts: Dict[str, int] = {}
    err_counts: Dict[str, int] = {}
    seen_episodes = set()
    for ep_path in episodes:
        try:
            d = _load_episode(ep_path)
        except Exception as e:
            print(f"skip {ep_path}: {e}")
            continue
        ep_name = os.path.basename(os.path.dirname(ep_path))
        if ep_name in seen_episodes:
            continue
        seen_episodes.add(ep_name)
        scene = d.get("scene", "")
        mode = d.get("mode", "")
        for f in d.get("frames", []):
            if not f.get("rgb"):
                continue
            vis = []
            for o in f.get("visible_objects", []):
                t = base_type(o.get("name", ""))
                type_counts[t] = type_counts.get(t, 0) + 1
                vis.append({
                    "name": o.get("name", ""),
                    "type": t,
                    "bbox": o.get("bbox"),
                })
            act = f.get("action") or {}
            aname = act.get("name", "")
            action_counts[aname] = action_counts.get(aname, 0) + 1
            err = classify_error(f.get("error_message"))
            err_counts[err] = err_counts.get(err, 0) + 1
            ag = f.get("agent") or {}
            frames.append({
                "episode": ep_name,
                "scene": scene,
                "mode": mode,
                "step": f.get("step"),
                "rgb": os.path.join(os.path.dirname(ep_path), f.get("rgb", "")),
                "depth": os.path.join(os.path.dirname(ep_path), f.get("depth", "")),
                "seg": os.path.join(os.path.dirname(ep_path), f.get("seg", "")),
                "action": aname,
                "action_args": act.get("args", {}) or {},
                "success": bool(f.get("action_success", True)),
                "error_class": err,
                "agent": {
                    "position": ag.get("position"),
                    "yaw": float(ag.get("rotation", {}).get("y", 0.0) or 0.0),
                    "horizon": float(ag.get("rotation", {}).get("x", 0.0) or 0.0),
                },
                "visible": vis,
            })
        if len(seen_episodes) % 200 == 0:
            print(f"indexed {len(seen_episodes)}/{len(episodes)} episodes, "
                  f"{len(frames)} frames")

    # scene ground truth (world positions per object name) for eval
    scene_gt: Dict[str, Dict[str, Dict[str, float]]] = {}
    for gp in sorted({p for g in SCENE_GT_GLOBS for p in glob.glob(g)}):
        g = json.load(open(gp))
        scene_gt[g["scene"]] = {
            o["name"]: o["position"] for o in g.get("objects", [])
        }

    types = sorted(type_counts)
    actions = sorted(action_counts)
    errors = sorted(err_counts)
    index = {
        "frames": frames,
        "object_types": types,
        "actions": actions,
        "error_classes": errors,
        "scene_gt": scene_gt,
        "stats": {
            "frames": len(frames),
            "episodes": len(seen_episodes),
            "type_counts": type_counts,
            "action_counts": action_counts,
            "error_counts": err_counts,
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(index, f, protocol=4)
    print(f"wrote {out_path}: {len(frames)} frames, {len(types)} types, "
          f"{len(actions)} actions, {len(errors)} error classes")
    return index


def load_index(path: str = DEFAULT_INDEX) -> Dict[str, Any]:
    if not os.path.exists(path):
        return build_index(path)
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INDEX
    build_index(out)
