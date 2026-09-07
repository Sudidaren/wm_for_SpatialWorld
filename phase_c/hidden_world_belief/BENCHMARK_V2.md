# Hidden-location benchmark v2

This benchmark replaces the heterogeneous pilot splits with one fixed,
scene-disjoint protocol for the main hidden-location result.

## Main benchmark

- Core target types: `Apple`, `Egg`, `Potato`, `KeyChain`, `CreditCard`,
  `Book`, `CellPhone`, `Mug`, `Bowl`, `Plate`, `Pan`, and `Pot`.
- Every type has exactly 50 train, 5 validation, and 10 test queries.
- Every type's 10 test queries come from 10 distinct AI2-THOR scenes (the
  declared minimum is 3).
- Native scene allocation is 60 train / 20 validation / 40 test, with each
  room family contributing 15 / 5 / 10 scenes respectively.
- The independent unit is exactly `(scene, target_id, parent_id)`. Repeated
  camera views do not increase the query count.
- Validation and test contain only natural AI2-THOR placements. Native train
  examples are supplemented to 50 per type with natural ProcTHOR-10K
  placements; rows are never copied merely to meet a quota.

The main files are in `data/object_location_benchmark_v2/`. The split seed,
scene lists, source counts, per-type coverage, and integrity checks are frozen
in `summary.json`.

## Separate evaluation tracks

Types that do not satisfy the main support rule are excluded from the main
average:

- `long_tail_test.jsonl`: 1--9 native training queries for the target type.
- `zero_shot_test.jsonl`: no native training query for the target type.
- `supplementary_test.jsonl`: non-core types with at least 10 native training
  queries. This is reported separately and does not change the 12-type main
  score.

## Reporting protocol

Report Hit@1, Hit@3, MRR, and average search steps for the 120-query main test.
Because each core type has exactly 10 queries, the row average and target-type
macro average have equal target weighting. Confidence intervals should be
bootstrapped over independent query keys, not observation frames.

Select models and thresholds using train/validation only. Evaluate the main
test once after freezing the choice, then report the long-tail, zero-shot, and
supplementary tracks in separate table rows.

## Reproducibility

Build and validate with:

```bash
python -m phase_c.hidden_world_belief.build_balanced_benchmark_v2
PYTHONPATH=. python -m pytest -q \
  phase_c/hidden_world_belief/test_balanced_benchmark_v2.py
```

Existing checkpoints trained on the old scene split are not valid clean-split
results for v2: several v2 test scenes previously belonged to training. Models
must be retrained from scratch on the v2 train file before reporting v2 test
accuracy.
