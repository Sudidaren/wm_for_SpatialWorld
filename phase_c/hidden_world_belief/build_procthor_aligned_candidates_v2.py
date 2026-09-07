"""Build ProcTHOR hidden-location data with a SpatialWorld-train candidate schema.

V1 only considered object types that happened to contain children in ProcTHOR,
which made its candidate sets much easier than SpatialWorld and mapped most
SpatialWorld furniture to ``<unk>``.  V2 derives the admissible candidate type
schema from SpatialWorld TRAIN rows and adds ProcTHOR TRAIN parent types.  All
objects of those types are candidates even when they are currently empty.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from phase_c.hidden_world_belief.build_procthor_official_benchmark import (
    CONTAINER,
    DEFAULT_TARGETS,
    OPENABLE,
    ROOM_MAP,
    object_type,
    official_dir,
    read_jsonl,
    write_jsonl,
)


def root_room_id(item: dict):
    parts = item.get("id", "").split("|")
    return f"room|{parts[1]}" if len(parts) > 2 else None


def flatten_house(house: dict):
    """Return (object, inherited room, immediate parent) for every tree node."""
    flattened = []

    def visit(item: dict, room: str, parent: dict | None) -> None:
        flattened.append((item, room, parent))
        for child in item.get("children", []):
            visit(child, room, item)

    for item in house.get("objects", []):
        visit(item, root_room_id(item), None)
    return flattened


def training_parent_types(source: Path) -> set[str]:
    kinds = set()
    with gzip.open(source / "train.jsonl.gz", "rt") as handle:
        for line in handle:
            for item, _, _ in flatten_house(json.loads(line)):
                if item.get("children"):
                    kinds.add(object_type(item))
    return kinds


def schema_candidate_types(path: Path) -> set[str]:
    return {candidate["receptacle_type"]
            for row in read_jsonl(path)
            for candidate in row["candidates"]}


def candidate_row(item: dict) -> dict:
    kind = object_type(item)
    return {
        "receptacle_id": item["id"], "receptacle_type": kind,
        "position": item.get("position", {"x": 0, "y": 0, "z": 0}),
        "openable": kind in OPENABLE,
        "relation": "in" if kind in CONTAINER else "on",
    }


def extract_split(source: Path, split: str, targets: set[str], quota: int,
                  candidate_types: set[str], seed: int):
    rng = random.Random(seed)
    reservoirs = defaultdict(list)
    seen_queries, seen_houses = Counter(), Counter()
    candidate_widths = []
    with gzip.open(source / f"{split}.jsonl.gz", "rt") as handle:
        for house_index, line in enumerate(handle):
            house = json.loads(line)
            scene = f"procthor-{split}-{house_index}"
            rooms = {room["id"]: ROOM_MAP.get(room.get("roomType"))
                     for room in house.get("rooms", [])}
            flat = flatten_house(house)
            by_room = defaultdict(list)
            for item, rid, _ in flat:
                if rooms.get(rid) and object_type(item) in candidate_types:
                    by_room[rid].append(item)

            house_options = defaultdict(list)
            for child, rid, parent in flat:
                if parent is None or rooms.get(rid) is None:
                    continue
                target = object_type(child)
                if target not in targets or object_type(parent) not in candidate_types:
                    continue
                candidates = [candidate_row(item) for item in by_room[rid]
                              if item["id"] != child["id"]]
                if len(candidates) < 2:
                    continue
                parent_ids = {item["receptacle_id"] for item in candidates}
                if parent["id"] not in parent_ids:
                    raise AssertionError("true parent missing from candidate set")
                parent_kind = object_type(parent)
                row = {
                    "scene": scene, "split": split,
                    "source": "procthor10k_official_aligned_candidates_v2",
                    "official_split": split, "house_index": house_index,
                    "room_type": rooms[rid], "target_id": child["id"],
                    "target_type": target,
                    "target_position": child.get(
                        "position", {"x": 0, "y": 0, "z": 0}),
                    "parent_id": parent["id"], "parent_type": parent_kind,
                    "relation": "in" if parent_kind in CONTAINER else "on",
                    "candidates": [
                        {**item, "label": int(item["receptacle_id"] == parent["id"])}
                        for item in candidates
                    ],
                }
                house_options[target].append(row)
                candidate_widths.append(len(candidates))

            for target, options in house_options.items():
                seen_houses[target] += 1
                seen_queries[target] += len(options)
                row = rng.choice(options)
                bucket = reservoirs[target]
                if len(bucket) < quota:
                    bucket.append(row)
                else:
                    replacement = rng.randrange(seen_houses[target])
                    if replacement < quota:
                        bucket[replacement] = row

    missing = {target: len(reservoirs[target]) for target in targets
               if len(reservoirs[target]) < quota}
    if missing:
        raise ValueError(f"{split} does not meet quota {quota}: {missing}")
    rows = [row for target in sorted(targets)
            for row in sorted(reservoirs[target], key=lambda item: item["scene"])]
    selected_widths = [len(row["candidates"]) for row in rows]
    stats = {
        "available_queries": dict(sorted(seen_queries.items())),
        "available_houses": dict(sorted(seen_houses.items())),
        "selected_candidate_count": {
            "min": min(selected_widths), "max": max(selected_widths),
            "mean": sum(selected_widths) / len(selected_widths),
        },
    }
    return rows, stats


def build(args) -> dict:
    targets = set(args.targets)
    schema_types = schema_candidate_types(args.candidate_schema_train)
    proc_parent_types = training_parent_types(args.source)
    candidate_types = schema_types | proc_parent_types
    quotas = {"train": args.train_per_type, "val": args.val_per_type,
              "test": args.test_per_type}
    outputs, split_stats = {}, {}
    for offset, split in enumerate(("train", "val", "test")):
        outputs[split], split_stats[split] = extract_split(
            args.source, split, targets, quotas[split], candidate_types,
            args.seed + offset)

    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        write_jsonl(args.out / f"{split}.jsonl", rows)
    keys = [(row["scene"], row["target_id"], row["parent_id"])
            for row in sum(outputs.values(), [])]
    summary = {
        "schema_version": 2, "dataset": "procthor-10k",
        "official_revision": args.source.name,
        "official_split_preserved": True,
        "candidate_schema_source": str(args.candidate_schema_train),
        "candidate_schema_uses_spatialworld_train_only": True,
        "seed": args.seed, "target_types": sorted(targets),
        "quotas": quotas,
        "records": {split: len(rows) for split, rows in outputs.items()},
        "candidate_types": sorted(candidate_types),
        "candidate_type_sources": {
            "spatialworld_train": sorted(schema_types),
            "procthor_train_parents": sorted(proc_parent_types),
            "spatialworld_types_absent_from_selected_procthor_train": sorted(
                schema_types - {item["receptacle_type"]
                                for row in outputs["train"]
                                for item in row["candidates"]}),
        },
        "split_stats": split_stats,
        "integrity": {
            "query_unique": len(keys) == len(set(keys)),
            "exact_quotas": all(
                sum(row["target_type"] == target for row in outputs[split])
                == quotas[split]
                for split in outputs for target in targets),
            "one_label_per_query": all(
                sum(int(item["label"]) for item in row["candidates"]) == 1
                for row in sum(outputs.values(), [])),
            "one_target_query_per_house": all(
                len({(row["scene"], row["target_type"]) for row in rows})
                == len(rows) for rows in outputs.values()),
            "official_scene_prefixes": all(
                row["scene"].startswith(f"procthor-{split}-")
                for split, rows in outputs.items() for row in rows),
        },
    }
    if not all(summary["integrity"].values()):
        raise AssertionError(summary["integrity"])
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=official_dir())
    parser.add_argument("--candidate-schema-train", type=Path, default=root /
                        "data/object_location_benchmark_v2/train.jsonl")
    parser.add_argument("--out", type=Path, default=root /
                        "data/procthor_hidden_location_aligned_v2")
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--train-per-type", type=int, default=50)
    parser.add_argument("--val-per-type", type=int, default=5)
    parser.add_argument("--test-per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
