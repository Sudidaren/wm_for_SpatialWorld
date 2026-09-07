"""Build a balanced, scene-disjoint hidden-location benchmark v2.

The main split contains an identical number of independent
``(scene, target_id, parent_id)`` queries per target type. Camera views are
never treated as independent queries. Non-core types are written only to
supplementary/long-tail/zero-shot files and never enter the main average.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


DEFAULT_CORE = (
    "Apple", "Egg", "Potato", "KeyChain", "CreditCard", "Book",
    "CellPhone", "Mug", "Bowl", "Plate", "Pan", "Pot",
)
FAMILIES = ((1, 30), (201, 230), (301, 330), (401, 430))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")


def query_key(row: dict) -> tuple[str, str, str]:
    return str(row["scene"]), str(row["target_id"]), str(row["parent_id"])


def scene_number(scene: str) -> int:
    if not scene.startswith("FloorPlan"):
        raise ValueError(f"not a native AI2-THOR scene: {scene}")
    return int(scene[len("FloorPlan"):])


def dedupe(rows: list[dict]) -> list[dict]:
    unique = {}
    for row in rows:
        if sum(int(candidate.get("label", 0))
               for candidate in row.get("candidates", [])) != 1:
            raise ValueError(f"query has non-unique label: {query_key(row)}")
        unique.setdefault(query_key(row), row)
    return list(unique.values())


def solve_scene_split(rows: list[dict], core: list[str], *, seed: int,
                      train_min: int, val_min: int, test_min: int,
                      family_val_scenes: int, family_test_scenes: int) -> dict:
    scenes = sorted({row["scene"] for row in rows}, key=scene_number)
    index = {scene: i for i, scene in enumerate(scenes)}
    n = len(scenes)
    counts = {target: np.zeros(n) for target in core}
    presence = {target: np.zeros(n) for target in core}
    for row in rows:
        target = row["target_type"]
        if target in counts:
            i = index[row["scene"]]
            counts[target][i] += 1
            presence[target][i] = 1

    # Variables are test_scene[0:n], val_scene[n:2n]; train is the remainder.
    matrix, lower, upper = [], [], []
    for i in range(n):
        constraint = np.zeros(2 * n)
        constraint[i] = constraint[n + i] = 1
        matrix.append(constraint); lower.append(-np.inf); upper.append(1)
    for target in core:
        total = float(counts[target].sum())
        if total < train_min + val_min + test_min:
            raise ValueError(
                f"{target} has {int(total)} queries, fewer than required "
                f"{train_min + val_min + test_min}")
        matrix.append(np.r_[counts[target], np.zeros(n)])
        lower.append(test_min); upper.append(np.inf)
        matrix.append(np.r_[np.zeros(n), counts[target]])
        lower.append(val_min); upper.append(np.inf)
        matrix.append(np.r_[counts[target], counts[target]])
        lower.append(-np.inf); upper.append(total - train_min)
        matrix.append(np.r_[presence[target], np.zeros(n)])
        lower.append(3); upper.append(np.inf)
    for lo, hi in FAMILIES:
        mask = np.array([lo <= scene_number(scene) <= hi
                         for scene in scenes], dtype=float)
        matrix.append(np.r_[mask, np.zeros(n)])
        lower.append(family_test_scenes); upper.append(family_test_scenes)
        matrix.append(np.r_[np.zeros(n), mask])
        lower.append(family_val_scenes); upper.append(family_val_scenes)

    # Seeded costs choose one deterministic feasible split without consulting
    # labels beyond the declared coverage constraints or any model metric.
    rng = random.Random(seed)
    objective = np.array([rng.random() for _ in range(2 * n)])
    result = milp(
        objective, integrality=np.ones(2 * n), bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.asarray(matrix), np.asarray(lower),
                                     np.asarray(upper)),
        options={"time_limit": 120},
    )
    if not result.success:
        raise RuntimeError(f"no feasible scene split: {result.message}")
    selected = np.rint(result.x).astype(int)
    test = [scene for i, scene in enumerate(scenes) if selected[i]]
    val = [scene for i, scene in enumerate(scenes) if selected[n + i]]
    held_out = set(test) | set(val)
    train = [scene for scene in scenes if scene not in held_out]
    return {"train": train, "val": val, "test": test}


def diverse_sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Prefer one query per scene before selecting extra instances."""
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    scenes = sorted(by_scene)
    rng.shuffle(scenes)
    chosen, remaining = [], []
    for scene in scenes:
        options = sorted(by_scene[scene], key=query_key)
        rng.shuffle(options)
        chosen.append(options[0])
        remaining.extend(options[1:])
    if len(chosen) > count:
        chosen = chosen[:count]
    else:
        rng.shuffle(remaining)
        chosen.extend(remaining[:count - len(chosen)])
    if len(chosen) != count:
        raise ValueError(f"only {len(chosen)} queries available; need {count}")
    return chosen


def build(args) -> dict:
    rng = random.Random(args.seed)
    native = dedupe(sum((read_jsonl(args.native / f"{split}.jsonl")
                         for split in ("train", "val", "test")), []))
    core = list(dict.fromkeys(args.core))
    split_scenes = solve_scene_split(
        native, core, seed=args.seed, train_min=args.native_train_min,
        val_min=args.val_per_type, test_min=args.test_per_type,
        family_val_scenes=args.family_val_scenes,
        family_test_scenes=args.family_test_scenes)
    scene_to_split = {scene: split for split, scenes in split_scenes.items()
                      for scene in scenes}
    native_by = defaultdict(list)
    for row in native:
        native_by[(scene_to_split[row["scene"]], row["target_type"])].append(row)
    # The historical ``procthor10k_task`` file also contains a small native
    # AI2-THOR seed set.  It must not be allowed to re-enter train after the
    # native scenes have been reassigned to val/test.
    procthor = [row for row in dedupe(read_jsonl(args.procthor / "train.jsonl"))
                if row["scene"].startswith("procthor-")]
    proc_by = defaultdict(list)
    for row in procthor:
        proc_by[row["target_type"]].append(row)

    outputs = {"train": [], "val": [], "test": []}
    per_type = {}
    for target in core:
        test = diverse_sample(native_by[("test", target)],
                              args.test_per_type, rng)
        val = diverse_sample(native_by[("val", target)],
                             args.val_per_type, rng)
        native_train = sorted(native_by[("train", target)], key=query_key)
        rng.shuffle(native_train)
        train = native_train[:args.train_per_type]
        if len(train) < args.train_per_type:
            supplement = sorted(proc_by[target], key=query_key)
            rng.shuffle(supplement)
            train.extend(supplement[:args.train_per_type - len(train)])
        if len(train) != args.train_per_type:
            raise ValueError(f"insufficient train support for {target}")
        outputs["train"].extend({**row, "split": "train"} for row in train)
        outputs["val"].extend({**row, "split": "val"} for row in val)
        outputs["test"].extend({**row, "split": "test"} for row in test)
        per_type[target] = {
            "train": len(train), "val": len(val), "test": len(test),
            "test_scenes": len({row["scene"] for row in test}),
            "native_train": sum(not row["scene"].startswith("procthor-")
                                for row in train),
            "procthor_train": sum(row["scene"].startswith("procthor-")
                                  for row in train),
        }
    for split in outputs:
        outputs[split].sort(key=lambda row: (row["target_type"], query_key(row)))

    core_set = set(core)
    train_support = Counter(row["target_type"] for row in native
                            if scene_to_split[row["scene"]] == "train")
    noncore_test = [row for row in native
                    if scene_to_split[row["scene"]] == "test"
                    and row["target_type"] not in core_set]
    zero = [row for row in noncore_test if train_support[row["target_type"]] == 0]
    long_tail = [row for row in noncore_test
                 if 0 < train_support[row["target_type"]] < args.native_train_min]
    supplementary = [row for row in noncore_test
                     if train_support[row["target_type"]] >= args.native_train_min]

    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        write_jsonl(args.out / f"{split}.jsonl", rows)
    write_jsonl(args.out / "long_tail_test.jsonl", long_tail)
    write_jsonl(args.out / "zero_shot_test.jsonl", zero)
    write_jsonl(args.out / "supplementary_test.jsonl", supplementary)
    summary = {
        "schema_version": 2,
        "independent_query_key": ["scene", "target_id", "parent_id"],
        "seed": args.seed, "core_types": core,
        "quotas": {"train_per_type": args.train_per_type,
                   "val_per_type": args.val_per_type,
                   "test_per_type": args.test_per_type,
                   "minimum_native_train_support": args.native_train_min},
        "records": {name: len(rows) for name, rows in outputs.items()},
        "scenes": {name: scenes for name, scenes in split_scenes.items()},
        "per_type": per_type,
        "excluded_from_main_average": {
            "long_tail_records": len(long_tail),
            "long_tail_types": sorted({row["target_type"] for row in long_tail}),
            "zero_shot_records": len(zero),
            "zero_shot_types": sorted({row["target_type"] for row in zero}),
            "supplementary_records": len(supplementary),
            "supplementary_types": sorted({row["target_type"]
                                           for row in supplementary}),
        },
        "integrity": {
            "scene_disjoint": not (
                set(split_scenes["train"]) & set(split_scenes["val"]) or
                set(split_scenes["train"]) & set(split_scenes["test"]) or
                set(split_scenes["val"]) & set(split_scenes["test"])),
            "query_disjoint": len(set(query_key(row) for row in sum(
                outputs.values(), []))) == sum(len(rows) for rows in outputs.values()),
            "keychain_in_test": any(row["target_type"] == "KeyChain"
                                    for row in outputs["test"]),
            "creditcard_in_test": any(row["target_type"] == "CreditCard"
                                      for row in outputs["test"]),
            "all_test_types_have_ten_queries": all(
                item["test"] == args.test_per_type for item in per_type.values()),
            "all_test_types_have_three_scenes": all(
                item["test_scenes"] >= 3 for item in per_type.values()),
        },
        "warning": "Models trained on the old split must be retrained; some v2 test scenes were old training scenes.",
    }
    if not all(summary["integrity"].values()):
        raise AssertionError(summary["integrity"])
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path,
                        default=root / "data/object_location_native_extended")
    parser.add_argument("--procthor", type=Path,
                        default=root / "data/object_location_procthor10k_task")
    parser.add_argument("--out", type=Path,
                        default=root / "data/object_location_benchmark_v2")
    parser.add_argument("--core", nargs="+", default=list(DEFAULT_CORE))
    parser.add_argument("--train-per-type", type=int, default=50)
    parser.add_argument("--val-per-type", type=int, default=5)
    parser.add_argument("--test-per-type", type=int, default=10)
    parser.add_argument("--native-train-min", type=int, default=10)
    parser.add_argument("--family-val-scenes", type=int, default=5)
    parser.add_argument("--family-test-scenes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
