# Phase C v2 final report (2026-09-07)

## Decision

Deploy the retrained static `bilinear_type_v5` prior in LightWM. Do not deploy
the learned within-type instance residual or the visible-context observation
head. The instance residual improved Hit@1 but missed the strict no-regression
gate; the observation head selected the exact `alpha=0` fallback.

The frozen decision is recorded in
`checkpoints/phase_c_final_manifest.json`. Model selection did not read
either test split. After freezing, each test split was evaluated exactly once.

The GitHub release contains only the deployed static checkpoint. Rejected
research checkpoints and their dedicated datasets remain local and are not
part of the published repository state.

## Data correction

`procthor_hidden_location_aligned_v2` preserves the official ProcTHOR train,
validation and test split. Candidate extraction recursively traverses object
trees, includes empty candidate-capable receptacles as negatives, excludes the
target itself, and fails closed on schema violations. The candidate vocabulary
is the union of ProcTHOR-train parent types and the SpatialWorld-train candidate
schema; SpatialWorld validation/test never define the vocabulary.

The aligned data contain 1,000/100/200 train/validation/test queries (20 target
types, fixed 50/5/10 quota per type) and 37 candidate types. Mean ProcTHOR test
candidate count increased from 5.325 in v1 to 7.805 in v2. Six SpatialWorld
candidate types are absent naturally from ProcTHOR and are supplied only by
SpatialWorld train: `Cabinet`, `Drawer`, `Footstool`, `Shelf`, `SinkBasin`, and
`StoveBurner`.

## Training sequence and validation

1. ProcTHOR aligned-v2 pretraining reached validation type Hit@1 68.0%, type
   Hit@3 98.0%, instance Hit@1 66.0%, and instance Hit@3 98.0%.
2. SpatialWorld train-only mixing selected the source-balanced checkpoint:
   validation type Hit@1 66.7%, type Hit@3 98.3%, instance Hit@1 35.0%, and
   instance Hit@3 85.0%.
3. The hierarchical instance residual improved SpatialWorld validation
   instance Hit@1 to 53.3% and Hit@3 to 86.7%, while type metrics were exactly
   invariant. ProcTHOR validation Hit@1/Hit@3 remained 62.0%/98.0%.
4. The visible-context head trained on 1,448 active context rows for six epochs
   of 2,048 stratified updates. None of 60 epoch/alpha combinations passed the
   task-context plus unseen-context gate, so its deployed alpha is zero.

## Frozen one-time test results

| Domain | Variant | Hit@1 | Hit@3 | MRR | Mean candidate checks |
|---|---:|---:|---:|---:|---:|
| SpatialWorld v2 (120) | deployed static v5 | 32.5% | 65.8% | 0.5179 | 4.65 |
| SpatialWorld v2 (120) | + instance residual | 37.5% | 64.2% | 0.5467 | 4.50 |
| ProcTHOR aligned v2 (200) | deployed static v5 | 60.5% | 97.0% | 0.7810 | 1.60 |
| ProcTHOR aligned v2 (200) | + instance residual | 60.5% | 97.0% | 0.7808 | 1.61 |

The residual saves 0.15 full-search candidate checks per SpatialWorld query
(3.2%) but loses 1.67 percentage points of Hit@3. It changes no ProcTHOR Hit@K
and costs 0.01 mean checks. Under the declared no-regression rule it is kept as
an experimental checkpoint, not a runtime default.

Compared on the same SpatialWorld benchmark v2, the old ProcTHOR-v1 checkpoint
had mean rank 8.058, instance Hit@1 26.7%, and instance Hit@3 63.3%. The deployed
v2 static prior reaches 4.65, 32.5%, and 65.8%, respectively: 3.408 fewer mean
candidate checks (42.3%), +5.83 points Hit@1, and +2.50 points Hit@3.

For a search policy capped at three candidates, expected checks equal
`3 - Hit@1 - Hit@2`. On SpatialWorld the residual changes this proxy from 2.175
to 2.100 checks (3.45% fewer), but its lower Top-3 success rate still causes
rejection. ProcTHOR remains 1.495 checks in both variants.

## Runtime status

`HiddenLocationAdvisor` now supports the deployed v5 architecture, and the
Phase-D B-group configuration generator points to the new checkpoint. The
advisor's persistent belief still performs the safe useful update: a confirmed
empty search candidate is removed and the remaining posterior is reranked.
Learned visible-context likelihood stays in shadow mode with exact projection.

The newer end-to-end GPT A/B run requires `OPENAI_API_KEY` in the submission
environment. No key is stored in configs, checkpoints, reports, or repository
files.
