from phase_c.hidden_world_belief.build_balanced_benchmark_v2 import (
    DEFAULT_CORE,
    query_key,
    read_jsonl,
)
from phase_c.hidden_world_belief.evaluate_balanced_benchmark_v2 import summarize


def test_generated_main_benchmark_is_balanced_and_disjoint():
    root = __import__("pathlib").Path(__file__).resolve().parent
    data = root / "data/object_location_benchmark_v2"
    splits = {name: read_jsonl(data / f"{name}.jsonl")
              for name in ("train", "val", "test")}
    expected = {"train": 50, "val": 5, "test": 10}
    for name, rows in splits.items():
        for target in DEFAULT_CORE:
            selected = [row for row in rows if row["target_type"] == target]
            assert len(selected) == expected[name]
        assert len({query_key(row) for row in rows}) == len(rows)
    scenes = {name: {row["scene"] for row in rows
                     if not row["scene"].startswith("procthor-")}
              for name, rows in splits.items()}
    assert not (scenes["train"] & scenes["val"])
    assert not (scenes["train"] & scenes["test"])
    assert not (scenes["val"] & scenes["test"])
    for target in DEFAULT_CORE:
        target_scenes = {row["scene"] for row in splits["test"]
                         if row["target_type"] == target}
        assert len(target_scenes) >= 3
    assert {"KeyChain", "CreditCard"}.issubset(
        {row["target_type"] for row in splits["test"]})


def test_rank_summary_uses_search_rank_as_step_count():
    result = summarize([1, 2, 4, 8])
    assert result["hit_at_1"] == 0.25
    assert result["hit_at_3"] == 0.5
    assert result["hit_at_5"] == 0.75
    assert result["mean_search_steps"] == 3.75
