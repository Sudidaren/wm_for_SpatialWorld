"""Deterministic train/val/test splits at SCENE and TASK level.

Leakage rules:
  * split by scene id (FloorPlan/cov/objviews share the same FloorPlan key);
  * rare-object scenes are stratified so every split keeps small-object
    coverage;
  * SpatialWorld tasks split at task level (never the same task in both
    train and eval), independent of the A/B/C gate group.

Output: data/splits.json
"""

from __future__ import annotations

import json
import os
import pickle
import random
import re
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from data_index import load_index  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "splits.json")

RARE_TYPES = {
    "CreditCard", "Watch", "KeyChain", "CellPhone", "Pencil",
    "AluminumFoil", "Egg", "RemoteControl", "TennisRacket", "BasketBall",
    "Candle", "Statue", "Vase", "AlarmClock",
}

SPATIALWORLD_TASKS = [
    "egg_pot", "keys_box", "open_fridge", "open_microwave", "pickup_phone",
    "potato_plate_microwave", "put_egg_in_pan", "slice_apple",
    "turn_on_light",
]
# default 8:2 -> test holds out the multi-step core + search tasks
SPATIALWORLD_TEST_DEFAULT = ["potato_plate_microwave", "keys_box"]


def split_ratio(items: List[str], rng: random.Random,
                tr: float = 0.8, va: float = 0.1) -> Dict[str, List[str]]:
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_tr = int(round(n * tr))
    n_va = int(round(n * va))
    return {
        "train": sorted(items[:n_tr]),
        "val": sorted(items[n_tr:n_tr + n_va]),
        "test": sorted(items[n_tr + n_va:]),
    }


def main() -> None:
    seed = int(os.environ.get("SPLIT_SEED", "42"))
    rng = random.Random(seed)
    index = load_index()
    frames = index["frames"]
    scene_frames: Dict[str, int] = {}
    scene_types: Dict[str, set] = {}
    for fr in frames:
        s = fr["scene"]
        scene_frames[s] = scene_frames.get(s, 0) + 1
        scene_types.setdefault(s, set()).update(
            o["type"] for o in fr.get("visible", []) if o.get("type"))

    floorplans = sorted(s for s in scene_frames if s.startswith("FloorPlan"))
    procthor = sorted(s for s in scene_frames if s.startswith("procthor"))
    vhome = sorted(s for s in scene_frames if s.startswith("virtualhome"))

    # stratify FloorPlans by whether they ever show a rare small object
    rare = [s for s in floorplans if scene_types[s] & RARE_TYPES]
    nrare = [s for s in floorplans if not (scene_types[s] & RARE_TYPES)]
    rare_s = split_ratio(rare, rng)
    nrare_s = split_ratio(nrare, rng)
    fp_split = {k: sorted(rare_s[k] + nrare_s[k]) for k in ("train", "val", "test")}

    proc_split = split_ratio(procthor, rng)
    vh_split = split_ratio(vhome, rng)

    fd_index = pickle.load(open(os.path.join(ROOT, "data", "fd_index.pkl"),
                                "rb"))
    fd_frames = fd_index["frames"]
    fd_scene_frames: Dict[str, int] = {}
    for fr in fd_frames:
        fd_scene_frames[fr.get("scene", "?")] = \
            fd_scene_frames.get(fr.get("scene", "?"), 0) + 1
    fd_scenes = sorted(fd_scene_frames)
    fd_split = split_ratio(fd_scenes, rng)

    splits = {
        "seed": seed,
        "scene_splits": {
            "ai2thor_floorplans": fp_split,
            "procthor": proc_split,
            "virtualhome": vh_split,
            "fd_scenes": fd_split,
        },
        "task_splits": {
            "spatialworld": {
                "train": [t for t in SPATIALWORLD_TASKS
                          if t not in SPATIALWORLD_TEST_DEFAULT],
                "val": [],
                "test": list(SPATIALWORLD_TEST_DEFAULT),
            },
        },
        "meta": {
            "scene_frames": scene_frames,
            "rare_scenes": sorted(rare),
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(splits, f, indent=1, ensure_ascii=False)

    # summary
    def summ(name: str, sp: Dict[str, List[str]]) -> None:
        rows = []
        for k in ("train", "val", "test"):
            scenes = sp[k]
            fr = sum(scene_frames.get(s, 0) for s in scenes)
            rows.append(f"{k}:{len(scenes)}场景/{fr}帧")
        print(f"{name}: " + " | ".join(rows))

    summ("AI2-THOR FloorPlans", fp_split)
    summ("ProcTHOR", proc_split)
    summ("VirtualHome", vh_split)
    def summ_fd(name: str, sp: Dict[str, List[str]]) -> None:
        rows = []
        for k in ("train", "val", "test"):
            scenes = sp[k]
            fr = sum(fd_scene_frames.get(s, 0) for s in scenes)
            rows.append(f"{k}:{len(scenes)}场景/{fr}帧")
        print(f"{name}: " + " | ".join(rows))

    summ_fd("FD scenes", fd_split)
    print("SpatialWorld tasks train:",
          splits["task_splits"]["spatialworld"]["train"])
    print("SpatialWorld tasks test :",
          splits["task_splits"]["spatialworld"]["test"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
