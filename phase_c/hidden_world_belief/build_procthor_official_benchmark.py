"""Build a balanced hidden-location benchmark from official ProcTHOR splits.

Each record is a natural child-object/parent-receptacle relation extracted
from a ProcTHOR house.  Official train, validation, and test houses are never
mixed, and at most one query per target type is sampled from a house.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_TARGETS = (
    "Apple", "Egg", "Potato", "KeyChain", "CreditCard", "Book",
    "CellPhone", "Mug", "Bowl", "Plate", "Pan", "Pot", "Bottle", "Box",
    "Kettle", "Ladle", "Newspaper", "PaperTowelRoll", "TeddyBear",
    "WineBottle",
)
ROOM_MAP = {
    "Kitchen": "kitchen", "LivingRoom": "living_room",
    "Bedroom": "bedroom", "Bathroom": "bathroom",
}
OPENABLE = {"Box", "Cabinet", "Drawer", "Fridge", "Microwave", "Safe"}
CONTAINER = OPENABLE | {
    "Bowl", "Cup", "GarbageCan", "Mug", "Pan", "Plate", "Pot", "Sink",
    "SinkBasin",
}


def object_type(item: dict) -> str:
    return item.get("id", "").split("|", 1)[0]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def query_key(row: dict) -> tuple[str, str, str]:
    return str(row["scene"]), str(row["target_id"]), str(row["parent_id"])


def room_id(item: dict):
    parts = item.get("id", "").split("|")
    return f"room|{parts[1]}" if len(parts) > 2 else None


def official_dir() -> Path:
    root = Path("/home/junchi.yao/.prior/datasets/allenai/procthor-10k")
    matches = [path for path in root.iterdir()
               if path.is_dir() and (path / "train.jsonl.gz").exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one cached ProcTHOR revision: {matches}")
    return matches[0]


def derive_receptacle_types(source: Path) -> set[str]:
    """Derive candidate vocabulary from train houses only."""
    kinds = set()
    with gzip.open(source / "train.jsonl.gz", "rt") as handle:
        for line in handle:
            for parent in json.loads(line).get("objects", []):
                if parent.get("children"):
                    kinds.add(object_type(parent))
    return kinds


def candidate_row(item: dict) -> dict:
    kind = object_type(item)
    return {
        "receptacle_id": item["id"], "receptacle_type": kind,
        "position": item.get("position", {"x": 0, "y": 0, "z": 0}),
        "openable": kind in OPENABLE,
        "relation": "in" if kind in CONTAINER else "on",
    }


def extract_split(source: Path, split: str, targets: set[str], quota: int,
                  receptacle_types: set[str], seed: int):
    """Reservoir-sample one natural query per target and house."""
    rng = random.Random(seed)
    reservoirs = defaultdict(list)
    seen_queries = Counter()
    seen_houses = Counter()
    path = source / f"{split}.jsonl.gz"
    with gzip.open(path, "rt") as handle:
        for house_index, line in enumerate(handle):
            house = json.loads(line)
            scene = f"procthor-{split}-{house_index}"
            room_types = {room["id"]: ROOM_MAP.get(room.get("roomType"))
                          for room in house.get("rooms", [])}
            by_room = defaultdict(list)
            for item in house.get("objects", []):
                rid = room_id(item)
                if (object_type(item) in receptacle_types
                        and room_types.get(rid)):
                    by_room[rid].append(item)

            house_options = defaultdict(list)
            for rid, objects in by_room.items():
                candidates = [candidate_row(item) for item in objects]
                if len(candidates) < 2:
                    continue
                candidate_ids = {item["receptacle_id"] for item in candidates}
                for parent in objects:
                    if parent["id"] not in candidate_ids:
                        continue
                    parent_kind = object_type(parent)
                    for child in parent.get("children", []):
                        target = object_type(child)
                        if target not in targets:
                            continue
                        relation = "in" if parent_kind in CONTAINER else "on"
                        row = {
                            "scene": scene, "split": split,
                            "source": "procthor10k_official_natural",
                            "official_split": split, "house_index": house_index,
                            "room_type": room_types[rid],
                            "target_id": child["id"], "target_type": target,
                            "target_position": child.get(
                                "position", {"x": 0, "y": 0, "z": 0}),
                            "parent_id": parent["id"],
                            "parent_type": parent_kind, "relation": relation,
                            "candidates": [
                                {**candidate,
                                 "label": int(candidate["receptacle_id"]
                                              == parent["id"])}
                                for candidate in candidates
                            ],
                        }
                        house_options[target].append(row)

            # One query of a target type per house makes every selected record
            # an independent scene observation.
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
            for row in sorted(reservoirs[target], key=lambda x: x["scene"])]
    return rows, seen_queries, seen_houses


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")


def build(args) -> dict:
    targets = set(args.targets)
    receptacles = derive_receptacle_types(args.source)
    quotas = {"train": args.train_per_type, "val": args.val_per_type,
              "test": args.test_per_type}
    outputs, availability = {}, {}
    for index, split in enumerate(("train", "val", "test")):
        rows, queries, houses = extract_split(
            args.source, split, targets, quotas[split], receptacles,
            args.seed + index)
        outputs[split] = rows
        availability[split] = {
            target: {"queries": queries[target], "houses": houses[target]}
            for target in sorted(targets)
        }

    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        write_jsonl(args.out / f"{split}.jsonl", rows)
    all_rows = sum(outputs.values(), [])
    keys = [(row["scene"], row["target_id"], row["parent_id"])
            for row in all_rows]
    scenes = {split: {row["scene"] for row in rows}
              for split, rows in outputs.items()}
    summary = {
        "schema_version": 1,
        "dataset": "procthor-10k",
        "official_revision": args.source.name,
        "official_split_preserved": True,
        "seed": args.seed,
        "independent_query_key": ["scene", "target_id", "parent_id"],
        "target_types": sorted(targets), "quotas": quotas,
        "records": {split: len(rows) for split, rows in outputs.items()},
        "availability_before_sampling": availability,
        "integrity": {
            "query_unique": len(keys) == len(set(keys)),
            "scene_disjoint": not (
                scenes["train"] & scenes["val"] or
                scenes["train"] & scenes["test"] or
                scenes["val"] & scenes["test"]),
            "one_query_per_target_per_house": all(
                len({(row["scene"], row["target_type"]) for row in rows})
                == len(rows) for rows in outputs.values()),
            "exact_quotas": all(
                sum(row["target_type"] == target for row in outputs[split])
                == quotas[split]
                for split in outputs for target in targets),
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
    parser.add_argument("--out", type=Path,
                        default=root / "data/procthor_hidden_location_v1")
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--train-per-type", type=int, default=50)
    parser.add_argument("--val-per-type", type=int, default=5)
    parser.add_argument("--test-per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
