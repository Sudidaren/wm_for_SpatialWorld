import json
from pathlib import Path

from phase_c.hidden_world_belief.build_procthor_official_benchmark import (
    DEFAULT_TARGETS,
    read_jsonl,
)


def test_aligned_v2_is_balanced_and_uses_train_only_candidate_schema():
    root = Path(__file__).resolve().parent
    data = root / "data/procthor_hidden_location_aligned_v2"
    summary = json.loads((data / "summary.json").read_text())
    assert summary["candidate_schema_uses_spatialworld_train_only"] is True
    assert all(summary["integrity"].values())
    quotas = {"train": 50, "val": 5, "test": 10}
    for split, quota in quotas.items():
        rows = read_jsonl(data / f"{split}.jsonl")
        assert len(rows) == len(DEFAULT_TARGETS) * quota
        for target in DEFAULT_TARGETS:
            selected = [row for row in rows if row["target_type"] == target]
            assert len(selected) == quota
            assert len({row["scene"] for row in selected}) == quota
            assert all(sum(int(item["label"]) for item in row["candidates"]) == 1
                       for row in selected)


def test_aligned_v2_expands_candidate_types_over_v1():
    root = Path(__file__).resolve().parent / "data"
    old = read_jsonl(root / "procthor_hidden_location_v1/test.jsonl")
    new = read_jsonl(root / "procthor_hidden_location_aligned_v2/test.jsonl")
    old_types = {item["receptacle_type"] for row in old
                 for item in row["candidates"]}
    new_types = {item["receptacle_type"] for row in new
                 for item in row["candidates"]}
    assert old_types < new_types
    assert sum(len(row["candidates"]) for row in new) > sum(
        len(row["candidates"]) for row in old)
