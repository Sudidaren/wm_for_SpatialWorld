from pathlib import Path

from phase_c.hidden_world_belief.build_procthor_official_benchmark import (
    DEFAULT_TARGETS,
    query_key,
    read_jsonl,
)


def test_official_procthor_benchmark_is_balanced_and_disjoint():
    data = Path(__file__).resolve().parent / "data/procthor_hidden_location_v1"
    splits = {split: read_jsonl(data / f"{split}.jsonl")
              for split in ("train", "val", "test")}
    quotas = {"train": 50, "val": 5, "test": 10}
    for split, rows in splits.items():
        assert len(rows) == len(DEFAULT_TARGETS) * quotas[split]
        assert len({query_key(row) for row in rows}) == len(rows)
        assert all(row["scene"].startswith(f"procthor-{split}-")
                   for row in rows)
        for target in DEFAULT_TARGETS:
            selected = [row for row in rows if row["target_type"] == target]
            assert len(selected) == quotas[split]
            assert len({row["scene"] for row in selected}) == quotas[split]
            assert all(sum(candidate["label"] for candidate in row["candidates"])
                       == 1 for row in selected)

    scenes = {split: {row["scene"] for row in rows}
              for split, rows in splits.items()}
    assert not scenes["train"] & scenes["val"]
    assert not scenes["train"] & scenes["test"]
    assert not scenes["val"] & scenes["test"]
