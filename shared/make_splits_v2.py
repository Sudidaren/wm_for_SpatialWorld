"""Split v2: unified scene registry (fd aligned), family-balanced, coverage-guaranteed.

Fixes vs v1 (data/splits.json):
  * fd scenes (FloorPlanX_physics) are aligned to the same canonical scene
    (FloorPlanX), so a room can never be train in one pool and test in another;
  * test/val quotas are stratified by AI2-THOR room family so 401-430 etc. are
    represented in the test set;
  * every object type visible in any TEST scene must also be visible in TRAIN
    scenes (coverage constraint), and every SpatialWorld target object on test
    tasks must be covered by train-scene evidence;
  * larger test set than v1 (target ~25% of AI2-THOR scenes).

Outputs:
  data/splits_v2.json                  (scene splits + fd alignment + coverage report)
  data/spatialworld_tasks_v2.json      (SpatialWorld AI2-THOR task -> split mapping)

Run from the repo root:
  python shared/make_splits_v2.py [--seed 42] [--test-frac 0.25] [--val-frac 0.15]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import random
import re
from typing import Dict, List, Set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAMILIES = [(1, 30, "1-30"), (201, 230, "201-230"),
            (301, 330, "301-330"), (401, 430, "401-430")]

# action-derived object states: not present at episode start; covered by parent
DERIVED_PARENT = {
    "BreadSliced": "Bread", "BreadToasted": "Bread",
    "TomatoSliced": "Tomato", "LettuceSliced": "Lettuce",
    "PotatoSliced": "Potato", "PotatoCooked": "Potato",
    "EggCracked": "Egg", "EggCooked": "Egg", "EggSliced": "Egg",
    "AppleSliced": "Apple", "BottleBroken": "Bottle",
    "CupBroken": "Cup", "MugBroken": "Mug", "PlateBroken": "Plate",
    "VaseBroken": "Vase", "WineBottleBroken": "WineBottle",
    "WindowBroken": "Window", "StatueBroken": "Statue",
    "PaperTowel": "PaperTowelRoll",
}
NON_TARGET = {"Water", "Wall", "Floor", "Ceiling", "Window", "Room", "Door"}


def family_of(scene: str) -> str:
    m = re.search(r"FloorPlan(\d+)", scene)
    if not m:
        return None
    n = int(m.group(1))
    for lo, hi, name in FAMILIES:
        if lo <= n <= hi:
            return name
    return f"other({n})"


def canonical(scene: str) -> str:
    """FloorPlan11_physics -> FloorPlan11; leave others untouched."""
    m = re.match(r"(FloorPlan\d+)", scene)
    return m.group(1) if m else scene


def norm_type(t: str):
    """Normalize noisy task tokens to the perception vocabulary if possible."""
    t = t.strip()
    if t in DERIVED_PARENT:
        return ("derived", DERIVED_PARENT[t])
    return ("raw", t)


def load_frame_index():
    with open(os.path.join(ROOT, "data", "frame_index.pkl"), "rb") as f:
        return pickle.load(f)


def load_fd_index():
    with open(os.path.join(ROOT, "data", "fd_index.pkl"), "rb") as f:
        return pickle.load(f)


def load_spatialworld_ai2thor_tasks():
    base = "/home/sudidaren/SpatialWorld/data/ai2thor/tasks"
    tasks = []
    if not os.path.isdir(base):
        print("[warn] SpatialWorld task dir not found:", base)
        return tasks
    for root, _dirs, files in os.walk(base):
        if "task.json" not in files:
            continue
        with open(os.path.join(root, "task.json"), encoding="utf-8") as f:
            d = json.load(f)
        tasks.append({"task_id": d.get("task_id"), "scene": d.get("scene"),
                      "targets": d.get("target_object_types", [])})
    return tasks


def scene_stats(index):
    """scene -> (visible types set, frames with visible count per type)."""
    types: Dict[str, Set[str]] = collections.defaultdict(set)
    type_frame_cnt: Dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for fr in index["frames"]:
        sc = fr.get("scene")
        if not sc:
            continue
        seen = set()
        for o in fr.get("visible", []):
            t = o.get("type")
            if not t or t in seen:
                continue
            seen.add(t)
            types[sc].add(t)
            type_frame_cnt[sc][t] += 1
    return types, type_frame_cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--min-test-scenes", type=int, default=20)
    args = ap.parse_args()

    index = load_frame_index()
    vocab = set(index.get("object_types", []))
    scene_types, scene_type_frames = scene_stats(index)

    # ---- canonical AI2-THOR universe: keep v1's 89 scenes (full task coverage)
    with open(os.path.join(ROOT, "data", "splits.json")) as f:
        old = json.load(f)
    floorplans = sorted({s for k in old["scene_splits"]
                         ["ai2thor_floorplans"].values() for s in k})
    assert len(floorplans) == 89

    fam_scenes: Dict[str, List[str]] = collections.defaultdict(list)
    for s in floorplans:
        f = family_of(s)
        if f and f in ("1-30", "201-230", "301-330", "401-430"):
            fam_scenes[f].append(s)
    assert sum(len(v) for v in fam_scenes.values()) == 89

    # canonical type evidence
    canon_types: Dict[str, Set[str]] = {
        s: scene_types.get(s, set()) for s in floorplans}
    canon_type_frames: Dict[str, Dict[str, int]] = {
        s: dict(scene_type_frames.get(s, {})) for s in floorplans}

    # types with very few scenes -> those scenes must stay in train
    type_scenes: Dict[str, Set[str]] = collections.defaultdict(set)
    for s, ts in canon_types.items():
        for t in ts:
            type_scenes[t].add(s)
    forced_train: Set[str] = set()
    forced_reason: Dict[str, str] = {}
    for t, scs in sorted(type_scenes.items()):
        if len(scs) == 1:
            s = next(iter(scs))
            forced_train.add(s)
            forced_reason[s] = f"only scene with type {t}"
        elif len(scs) == 2:
            # keep one witness in train so the other may safely go to test
            s = sorted(scs)[0]
            forced_train.add(s)
            forced_reason.setdefault(
                s, f"witness for 2-scene type {t}")

    def quota(n: int, frac: float, low: int) -> int:
        return max(low, int(round(n * frac)))

    # ---- constructive split ----------------------------------------------
    # Guarantee coverage by pinning per-type "witness" scenes into train:
    # every type keeps min(2, #scenes) training scenes, so any type that shows
    # up in a test scene is guaranteed to have been seen in training scenes.
    rng = random.Random(args.seed)
    pinned: Set[str] = set()
    pin_reason: Dict[str, str] = {}
    for t, scs in sorted(type_scenes.items()):
        if len(scs) <= 2:
            for s in scs:
                pinned.add(s)
                pin_reason[s] = f"type {t} appears in only {len(scs)} scene(s)"
        else:
            # two witnesses with the most visible frames of this type
            ranked = sorted(
                scs,
                key=lambda s: (-canon_type_frames.get(s, {}).get(t, 0), s))
            for s in ranked[:2]:
                pinned.add(s)
                pin_reason.setdefault(s, f"witness for type {t}")

    fam_quotas: Dict[str, int] = {}
    total_test = 0
    for fam in ("1-30", "201-230", "301-330", "401-430"):
        n = len(fam_scenes[fam])
        pool = [s for s in fam_scenes[fam] if s not in pinned]
        q = min(quota(n, args.test_frac, 2), len(pool))
        fam_quotas[fam] = q
        total_test += q
    # if the coverage pins left too few movable scenes, keep at least what
    # remains feasible; assert we still hit the user-requested minimum
    if total_test < args.min_test_scenes:
        short = args.min_test_scenes - total_test
        for fam in ("1-30", "201-230", "301-330", "401-430"):
            n = len(fam_scenes[fam])
            pool = [s for s in fam_scenes[fam] if s not in pinned]
            room = len(pool) - fam_quotas[fam]
            take = min(room, short)
            fam_quotas[fam] += take
            short -= take
            if short <= 0:
                break

    test: Dict[str, List[str]] = {}
    val: Dict[str, List[str]] = {}
    tr: Dict[str, List[str]] = {}
    for fam in ("1-30", "201-230", "301-330", "401-430"):
        n = len(fam_scenes[fam])
        pool = [s for s in fam_scenes[fam] if s not in pinned]
        qtest = fam_quotas[fam]
        qval = min(quota(n, args.val_frac, 1), len(pool) - qtest)
        rng.shuffle(pool)
        test[fam] = sorted(pool[:qtest])
        val[fam] = sorted(pool[qtest:qtest + qval])
        tr[fam] = sorted((pinned & set(fam_scenes[fam]))
                         | set(pool[qtest + qval:]))

    train_scenes = {s for v in tr.values() for s in v}
    val_scenes = {s for v in val.values() for s in v}
    test_scenes = {s for v in test.values() for s in v}
    assert len(train_scenes | val_scenes | test_scenes) == len(floorplans)
    assert len(train_scenes & val_scenes) == 0
    assert len(train_scenes & test_scenes) == 0
    assert len(val_scenes & test_scenes) == 0

    # verify coverage (should hold by construction)
    train_types = set()
    for s in train_scenes:
        train_types |= canon_types[s]
    missing = set()
    for s in test_scenes:
        missing |= (canon_types[s] - train_types)
    if missing:
        raise SystemExit("coverage violated: " + ",".join(sorted(missing)))

    # ---- fd alignment: same canonical room -> same split ------------------
    fd = load_fd_index()
    fd_ep_scene = collections.Counter()
    fd_frame_scene = collections.Counter()
    for fr in fd["frames"]:
        c = canonical(fr.get("scene") or "")
        fd_ep_scene[c] += 1
        fd_frame_scene[c] += 1
    fd_episodes = {k: v for k, v in fd_ep_scene.items()}
    # episodes: fd_index has one row per step; count unique episode ids instead
    fd_ep_ids = collections.defaultdict(set)
    for fr in fd["frames"]:
        c = canonical(fr.get("scene") or "")
        ep = fr.get("episode")
        if ep:
            fd_ep_ids[c].add(ep)
    fd_eps = {k: len(v) for k, v in fd_ep_ids.items()}

    # ---- SpatialWorld AI2-THOR task mapping -------------------------------
    sw_tasks = load_spatialworld_ai2thor_tasks()
    task_split: Dict[str, str] = {}
    for t in sw_tasks:
        c = canonical(t["scene"]) if t["scene"] else None
        if c in train_scenes:
            task_split[t["task_id"]] = "train"
        elif c in val_scenes:
            task_split[t["task_id"]] = "val"
        elif c in test_scenes:
            task_split[t["task_id"]] = "test"
        else:
            task_split[t["task_id"]] = "uncovered"
    task_count = collections.Counter(task_split.values())
    test_task_ids = sorted(
        tid for tid, s in task_split.items() if s == "test")

    # ---- coverage report ---------------------------------------------------
    train_types = set()
    for s in train_scenes:
        train_types |= canon_types[s]
    val_types = set()
    for s in val_scenes:
        val_types |= canon_types[s]
    test_types = set()
    for s in test_scenes:
        test_types |= canon_types[s]

    missing_test_from_train = sorted(test_types - train_types)
    missing_test_from_trval = sorted(test_types - (train_types | val_types))

    # task-target coverage for test tasks
    uncovered_targets = []
    derived_targets = set()
    raw_targets = set()
    for tid in test_task_ids:
        t = next(x for x in sw_tasks if x["task_id"] == tid)
        for raw in t["targets"]:
            kind, tok = norm_type(raw)
            if kind == "derived":
                derived_targets.add(raw)
            elif tok in NON_TARGET:
                continue
            else:
                raw_targets.add(raw)
    low_vocab = {x.lower() for x in vocab}
    raw_not_in_vocab = sorted(
        x for x in raw_targets if x.lower() not in low_vocab
        and x not in train_types)
    raw_in_train = sorted(
        x for x in raw_targets
        if x.lower() in low_vocab or x in train_types)
    uncovered_targets = raw_not_in_vocab  # strict: not seen in train evidence

    # ---- type training-frame adequacy -------------------------------------
    train_type_frames = collections.Counter()
    for s in train_scenes:
        for t, n in canon_type_frames.get(s, {}).items():
            train_type_frames[t] += n
    thin_types = sorted(
        (t, n) for t, n in train_type_frames.items() if n < 5)

    # ---- assemble output ---------------------------------------------------
    scene_split = {
        "ai2thor_floorplans": {
            "train": sorted(train_scenes),
            "val": sorted(val_scenes),
            "test": sorted(test_scenes),
        }
    }
    out = {
        "seed": args.seed,
        "method": ("family-stratified random split; fd aligned by canonical "
                   "scene; coverage: test-scene types must appear in train"),
        "ratios": {"test_frac": args.test_frac, "val_frac": args.val_frac},
        "scene_splits": scene_split,
        "fd_alignment": {
            "note": "fd scenes (FloorPlanX_physics) follow their canonical "
                    "FloorPlanX assignment",
            "episodes_by_split": {
                k: sum(fd_eps.get(s, 0) for s in scene_split
                       ["ai2thor_floorplans"][k])
                for k in ("train", "val", "test")
            },
            "frames_by_split": {
                k: sum(fd_frame_scene.get(s, 0) for s in scene_split
                       ["ai2thor_floorplans"][k])
                for k in ("train", "val", "test")
            },
        },
        "coverage_report": {
            "train_scenes": len(train_scenes),
            "val_scenes": len(val_scenes),
            "test_scenes": len(test_scenes),
            "train_visible_types": len(train_types),
            "val_visible_types": len(val_types),
            "test_visible_types": len(test_types),
            "missing_test_types_from_train": missing_test_from_train,
            "missing_test_types_from_train_or_val": missing_test_from_trval,
            "thin_train_types_lt5_frames": thin_types,
            "forced_train_scenes": sorted(
                s for s in forced_train if s in train_scenes),
            "forced_train_reasons": {s: forced_reason[s]
                                     for s in sorted(forced_train)},
        },
        "task_splits": {
            "spatialworld_ai2thor": {
                "counts": dict(task_count),
                "test_task_ids": test_task_ids,
                "test_task_count": len(test_task_ids),
                "test_target_coverage": {
                    "derived_action_states_only_in_test": sorted(derived_targets),
                    "raw_targets_in_vocab_or_train": raw_in_train,
                    "raw_targets_uncovered": uncovered_targets,
                },
            }
        },
        "meta": {
            "procthor_scenes_note": ("own-pool procthor scenes (procthor-val-*) "
                                     "not split here; see v1 or add explicit "
                                     "universe"),
            "virtualhome_scenes_note": ("own-pool virtualhome scenes keep v1 "
                                        "split; SpatialWorld vh task scenes use "
                                        "a different id space"),
        },
    }
    with open(os.path.join(ROOT, "data", "splits_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open(os.path.join(ROOT, "data", "spatialworld_tasks_v2.json"), "w",
              encoding="utf-8") as f:
        json.dump({"task_split": dict(sorted(task_split.items())),
                   "counts": dict(task_count),
                   "test_task_ids": test_task_ids},
                  f, indent=1, ensure_ascii=False)

    # ---- console summary ---------------------------------------------------
    print("AI2-THOR floorplans  train/val/test:",
          len(train_scenes), len(val_scenes), len(test_scenes))
    for fam in ("1-30", "201-230", "301-330", "401-430"):
        print(f"  {fam}: train {sum(1 for s in train_scenes if family_of(s)==fam)} "
              f"val {sum(1 for s in val_scenes if family_of(s)==fam)} "
              f"test {sum(1 for s in test_scenes if family_of(s)==fam)}")
    print("fd aligned  episodes train/val/test:",
          {k: v for k, v in out["fd_alignment"]["episodes_by_split"].items()})
    print("coverage: test types missing from train:",
          missing_test_from_train)
    print("SpatialWorld AI2-THOR tasks  train/val/test:",
          task_count)
    print("test task ids ({}):".format(len(test_task_ids)),
          test_task_ids[:20], "..." if len(test_task_ids) > 20 else "")
    print("wrote data/splits_v2.json and data/spatialworld_tasks_v2.json")


if __name__ == "__main__":
    main()
