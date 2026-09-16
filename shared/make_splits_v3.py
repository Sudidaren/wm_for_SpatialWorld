"""Split v3: size-bucket coverage on top of the v2 scene split.

Spec (see 数据补采与切分计划_2026-09-06.md):
  * test rooms keep zero training frames (per canonical AI2-THOR room);
  * pool policy: all 100 procTHOR val houses -> train (SpatialWorld
    procTHOR eval runs on the procTHOR-10k *train* partition, so the val
    partition houses we collected can never leak into eval rooms);
    virtualhome scenes 1-5 (+legacy 15/17/20/40) -> train, scenes 0/6 -> test;
  * coverage is checked at (category x size-bucket) granularity, buckets
    defined on the 224px model scale (bbox * 224/800):
      A <=16px, B 17-48px, C 49-144px, D >144px;
  * the SpatialWorld AI2-THOR 60-task test set (v2 rooms) is preserved; the
    binding guarantee is per test-task target: every (target category x size
    bucket) present in a test room must have >=10 training-evidence frames
    (AI2-THOR train rooms + procTHOR + VH train).  Shortfalls are reported
    with concrete witness-collection suggestions instead of shrinking the
    eval set (--fix-by-room-moves opts into the greedy fallback).

Outputs (repo data/):
  splits_v3.json               scene-level split incl. explicit pool policy
  spatialworld_tasks_v3.json   task -> split for AI2-THOR + VirtualHome
  coverage_report_v3.json      per test room / task (category x bucket) proof
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import re
from typing import Dict, List, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE = 224.0 / 800.0          # perception model input over capture width
BUCKETS = ["A<=16", "B17-48", "C49-144", "D>144"]
MIN_TEST_SCENES = 20
MIN_EVIDENCE_FRAMES = 10


def bucket_of(bbox) -> str:
    if not bbox or len(bbox) < 4:
        return ""
    sz = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * SCALE
    if sz <= 16:
        return "A<=16"
    if sz <= 48:
        return "B17-48"
    if sz <= 144:
        return "C49-144"
    return "D>144"


def canonical(scene: str) -> str:
    m = re.match(r"(FloorPlan\d+)", str(scene or ""))
    return m.group(1) if m else ""


def load_frame_index():
    with open(os.path.join(ROOT, "data", "frame_index.pkl"), "rb") as f:
        return pickle.load(f)


def load_fd_index():
    with open(os.path.join(ROOT, "data", "fd_index.pkl"), "rb") as f:
        return pickle.load(f)


def scene_tb_counts(index):
    """canonical ai2thor scene -> Counter[(type,bucket)] unique frames."""
    sc_tb: Dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for fr in index["frames"]:
        sc = canonical(fr.get("scene", ""))
        if not sc:
            continue
        seen = set()
        for o in fr.get("visible", []):
            t = o.get("type")
            b = bucket_of(o.get("bbox"))
            if t and b:
                seen.add((t, b))
        for tb in seen:
            sc_tb[sc][tb] += 1
    return sc_tb


def extra_pool_evidence(index) -> collections.Counter:
    """(type,bucket) frame counts from procTHOR + VirtualHome train scenes."""
    vh_test = {"virtualhome-0", "virtualhome-6"}
    extra: collections.Counter = collections.Counter()
    for fr in index["frames"]:
        s = str(fr.get("scene", ""))
        if s.startswith("procthor") or (
                s.startswith("virtualhome") and s not in vh_test):
            seen = set()
            for o in fr.get("visible", []):
                t = o.get("type")
                b = bucket_of(o.get("bbox"))
                if t and b:
                    seen.add((t, b))
            for tb in seen:
                extra[tb] += 1
    return extra


def load_tasks(base: str) -> List[dict]:
    tasks = []
    if not os.path.isdir(base):
        return tasks
    for root, _dirs, files in os.walk(base):
        if "task.json" not in files:
            continue
        with open(os.path.join(root, "task.json"), encoding="utf-8") as f:
            d = json.load(f)
        tasks.append({
            "task_id": d.get("task_id"),
            "scene": d.get("scene"),
            "targets": d.get("target_object_types") or [],
        })
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-test-scenes", type=int, default=MIN_TEST_SCENES)
    ap.add_argument("--min-evidence-frames", type=int,
                    default=MIN_EVIDENCE_FRAMES)
    ap.add_argument("--fix-by-room-moves", action="store_true",
                    help="move violating test rooms to train (shrinks the "
                         "60-task eval set); default keeps v2 rooms")
    args = ap.parse_args()

    index = load_frame_index()
    with open(os.path.join(ROOT, "data", "splits_v2.json")) as f:
        v2 = json.load(f)
    base = v2["scene_splits"]["ai2thor_floorplans"]
    train = set(base["train"])
    val = set(base["val"])
    test = set(base["test"])
    assert len(train | val | test) == 89
    assert not (train & test) and not (val & test) and not (train & val)

    sc_tb = scene_tb_counts(index)
    extra = extra_pool_evidence(index)

    def evidence(tb: Tuple[str, str]) -> int:
        return extra[tb] + sum(sc_tb[s][tb] for s in train)

    ai2_tasks = load_tasks(
        "/home/sudidaren/SpatialWorld/data/ai2thor/tasks")
    room_targets: Dict[str, Set[str]] = collections.defaultdict(set)
    for t in ai2_tasks:
        sc = canonical(t.get("scene", ""))
        if sc:
            room_targets[sc].update(t.get("targets") or [])

    def target_violations() -> Dict[str, Dict[Tuple[str, str], int]]:
        """Test-room gaps on SpatialWorld task targets (drive room moves)."""
        v = collections.defaultdict(dict)
        for s in sorted(test):
            for t in sorted(room_targets.get(s, set())):
                for b in BUCKETS:
                    tb = (t, b)
                    if sc_tb[s].get(tb, 0) > 0 and evidence(tb) < \
                            args.min_evidence_frames:
                        v[s][tb] = evidence(tb)
        return v

    moves: List[str] = []
    v = target_violations()
    if args.fix_by_room_moves:
        while v and len(test) > args.min_test_scenes:
            ranked = sorted(
                v.items(),
                key=lambda kv: (-len(kv[1]),
                                -sum(sc_tb[kv[0]][tb] for tb in kv[1]),
                                kv[0]))
            s = ranked[0][0]          # most violated rooms first
            test.discard(s)
            train.add(s)
            moves.append(f"test->train:{s}")
            v = target_violations()
            if len(moves) > 80:
                raise SystemExit("greedy move loop did not converge")
        while v and len(test) > args.min_test_scenes and val:
            cnt = collections.Counter()
            for dd in v.values():
                for tb in dd:
                    cnt[tb] += 1
            tb, _ = cnt.most_common(1)[0]
            cand = max(val, key=lambda s: sc_tb[s].get(tb, 0))
            if sc_tb[cand].get(tb, 0) == 0:
                break
            val.discard(cand)
            train.add(cand)
            moves.append(f"val->train:{cand}")
            v = target_violations()

    train_scenes = sorted(train)
    val_scenes = sorted(val)
    test_scenes = sorted(test)
    assert len(set(train_scenes) | set(val_scenes) | set(test_scenes)) == 89
    assert not (set(train_scenes) & set(test_scenes))
    assert len(test_scenes) >= args.min_test_scenes

    # ---- explicit pool policies ------------------------------------------
    frames = index["frames"]
    pool_frames = collections.Counter()
    pool_eps = collections.Counter()
    seen_eps = set()
    for fr in frames:
        s = str(fr.get("scene", ""))
        ep = fr.get("episode", "")
        if s.startswith("procthor"):
            pool_frames["procthor:train"] += 1
            if ep not in seen_eps:
                seen_eps.add(ep)
                pool_eps["procthor:train"] += 1
        elif s.startswith("virtualhome"):
            split = "test" if s in ("virtualhome-0", "virtualhome-6") \
                else "train"
            pool_frames[f"virtualhome:{split}"] += 1
            if ep not in seen_eps:
                seen_eps.add(ep)
                pool_eps[f"virtualhome:{split}"] += 1
        else:
            sc = canonical(s)
            split = ("test" if sc in test_scenes
                     else "val" if sc in val_scenes else "train")
            pool_frames[f"ai2thor:{split}"] += 1
            if ep not in seen_eps:
                seen_eps.add(ep)
                pool_eps[f"ai2thor:{split}"] += 1

    # ---- coverage report --------------------------------------------------
    residual = {}
    for s in test_scenes:
        row = {}
        for tb, c in sorted(sc_tb[s].items()):
            ev = evidence(tb)
            if ev < args.min_evidence_frames:
                row[".".join(tb)] = {"test_frames": c, "train_evidence": ev}
        if row:
            residual[s] = row

    # ai2thor per-task proof
    ai2_tasks = load_tasks(
        "/home/sudidaren/SpatialWorld/data/ai2thor/tasks")
    ai2_task_split = {}
    for t in ai2_tasks:
        sc = canonical(t.get("scene", ""))
        if sc in test_scenes:
            ai2_task_split[t["task_id"]] = "test"
        elif sc in val_scenes:
            ai2_task_split[t["task_id"]] = "val"
        else:
            ai2_task_split[t["task_id"]] = "train"
    ai2_test_ids = sorted(
        tid for tid, sp in ai2_task_split.items() if sp == "test")

    task_proof = {}
    for tid in ai2_test_ids:
        t = next(x for x in ai2_tasks if x["task_id"] == tid)
        sc = canonical(t["scene"])
        rows = []
        for raw in t["targets"]:
            for b in BUCKETS:
                ev = evidence((raw, b))
                test_fr = sc_tb[sc].get((raw, b), 0)
                rows.append({
                    "type": raw, "bucket": b,
                    "test_frames": test_fr,
                    "train_evidence_frames": ev,
                    "ok": ev >= args.min_evidence_frames,
                })
        task_proof[tid] = {"scene": t["scene"], "rows": rows}
    target_failures = sum(
        1 for rows in task_proof.values()
        for r in rows["rows"] if r["test_frames"] > 0 and not r["ok"])

    final_tv = target_violations()
    target_gaps = []
    for s in sorted(final_tv):
        for tb, ev_now in sorted(final_tv[s].items()):
            t, b = tb
            cand = []
            for sc in sorted(train):
                n = sum(sc_tb[sc].get((t, bb), 0) for bb in BUCKETS)
                if n:
                    cand.append({"scene": sc, "type_frames": n})
            cand.sort(key=lambda x: -x["type_frames"])
            target_gaps.append({
                "room": s, "type": t, "bucket": b,
                "test_frames": sc_tb[s].get(tb, 0),
                "train_evidence_frames": ev_now,
                "need_more_frames": max(
                    0, args.min_evidence_frames - ev_now),
                "witness_candidate_train_scenes": cand[:8],
            })

    vh_tasks = load_tasks(
        "/home/sudidaren/SpatialWorld/data/virtualhome/tasks")
    vh_task_split = {}
    for t in vh_tasks:
        sc = str(t.get("scene", ""))
        if sc in ("0", "6"):
            vh_task_split[t["task_id"]] = "test"
        elif sc in ("1", "2", "3", "4", "5"):
            vh_task_split[t["task_id"]] = "train"
        else:
            vh_task_split[t["task_id"]] = "legacy_excluded"
    vh_test_ids = sorted(
        tid for tid, sp in vh_task_split.items() if sp == "test")

    procthor_tasks = load_tasks(
        "/home/sudidaren/SpatialWorld/data/procthor/tasks")

    # ---- write outputs ----------------------------------------------------
    scene_split = {
        "ai2thor_floorplans": {
            "train": train_scenes, "val": val_scenes, "test": test_scenes,
        },
        "procthor_val": {"train": sorted(
            f"procthor-val-{i}" for i in range(100))},
        "virtualhome": {
            "train": ["virtualhome-1", "virtualhome-2", "virtualhome-3",
                      "virtualhome-4", "virtualhome-5",
                      "virtualhome-15", "virtualhome-17",
                      "virtualhome-20", "virtualhome-40"],
            "test": ["virtualhome-0", "virtualhome-6"],
        },
    }
    out = {
        "version": "v3",
        "base": ("v2 scene split (kept: preserves the 60-task AI2-THOR test "
                 "set); optional greedy room moves disabled by default"),
        "fix_by_room_moves": args.fix_by_room_moves,
        "moves_from_v2": moves,
        "bucket_definition": "bbox longer side scaled by 224/800",
        "evidence_floor_frames": args.min_evidence_frames,
        "scene_splits": scene_split,
        "pool_frames": dict(pool_frames),
        "pool_episodes": dict(pool_eps),
        "coverage": {
            "residual_underfloor": residual,
            "residual_room_count": len(residual),
            "target_coverage_failures": target_failures,
            "target_gaps": target_gaps,
            "target_gap_count": len(target_gaps),
            "note": ("residual_underfloor rows are INCIDENTAL non-target "
                     "(type,bucket) combinations with sparse train evidence. "
                     "The binding guarantee is per SpatialWorld test-task "
                     "target; target_gaps list the shortfalls with concrete "
                     "witness train scenes to collect more frames from."),
        },
        "task_splits": {
            "spatialworld_ai2thor": {
                "counts": dict(collections.Counter(
                    ai2_task_split.values())),
                "test_task_ids": ai2_test_ids,
                "test_task_count": len(ai2_test_ids),
                "per_task_coverage": task_proof,
            },
            "virtualhome": {
                "counts": dict(collections.Counter(vh_task_split.values())),
                "test_task_ids": vh_test_ids,
                "test_task_count": len(vh_test_ids),
                "note": ("vh targets are lowercase asset names; vocab "
                         "mapping/extensions to ~130 classes pending"),
            },
            "procthor": {
                "task_count": len(procthor_tasks),
                "note": ("procthor SpatialWorld tasks run on procTHOR-10k "
                         "train-partition houses (scene not exposed); our "
                         "100 collected houses are val-partition -> no "
                         "overlap with eval rooms by construction"),
            },
        },
        "meta": {
            "index_stats": index["stats"],
            "virtualhome_scenes_note": ("scenes 0/6 are the 30 SpatialWorld "
                                        "vh test tasks; 1-5 train; 15/17/20/40 "
                                        "legacy train evidence from another "
                                        "simulator build"),
        },
    }
    with open(os.path.join(ROOT, "data", "splits_v3.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open(os.path.join(ROOT, "data", "spatialworld_tasks_v3.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "ai2thor": {"task_split": ai2_task_split,
                        "counts": dict(collections.Counter(
                            ai2_task_split.values())),
                        "test_task_ids": ai2_test_ids},
            "virtualhome": {"task_split": vh_task_split,
                            "counts": dict(collections.Counter(
                                vh_task_split.values())),
                            "test_task_ids": vh_test_ids},
        }, f, indent=1, ensure_ascii=False)
    with open(os.path.join(ROOT, "data", "coverage_report_v3.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "bucket_definition": out["bucket_definition"],
            "evidence_floor_frames": args.min_evidence_frames,
            "residual": residual,
            "target_gaps": target_gaps,
            "per_task_coverage": task_proof,
        }, f, indent=1, ensure_ascii=False)

    print("AI2-THOR train/val/test:",
          len(train_scenes), len(val_scenes), len(test_scenes))
    print("moves_from_v2:", moves)
    print("pool frames:", dict(pool_frames))
    print("per-task target coverage failures:", target_failures)
    print("incidental underfloor rooms:", len(residual),
          "combos:", sum(len(x) for x in residual.values()))
    for s, row in sorted(residual.items()):
        print("  ", s, row)
    print("target gaps (test task targets needing witness frames):",
          len(target_gaps))
    for g in target_gaps:
        print("  ", g["room"], g["type"], g["bucket"],
              "ev", g["train_evidence_frames"],
              "need", g["need_more_frames"],
              "| candidates",
              [c["scene"] for c in g["witness_candidate_train_scenes"][:4]])
    print("ai2thor test tasks:", len(ai2_test_ids),
          "| vh test tasks:", len(vh_test_ids),
          "| procthor tasks:", len(procthor_tasks))
    print("wrote data/splits_v3.json, spatialworld_tasks_v3.json, "
          "coverage_report_v3.json")


if __name__ == "__main__":
    main()
