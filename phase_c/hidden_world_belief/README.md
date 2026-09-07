# Hidden World Belief labels

This directory is isolated from the existing Phase-C VoI gate. It builds the
two supervised datasets needed by the hidden-world advisor:

- `build_object_location_labels.py`: hidden object -> current-scene receptacle.
- `build_walkability_labels.py`: partial observation target -> complete reachable map.

Both builders use `data/splits.json`; ground-truth labels are for training and
evaluation only and must never be passed to the online GPT-5 agent.

`search_state/` keeps the validated `object_ranker.pt` as the static prior and
updates suggestions with online evidence. A receptacle confirmed empty is
removed; direct visibility overrides the prior. Route cost may reorder action
utility but never changes the reported location probability. This improves
instance search without adding distribution-shifted placement snapshots.

## Final released model (2026-09-07)

The repository publishes one Phase-C hidden-location model only:

`checkpoints/spatialworld_v2_finetune_mixed_balanced.pt`

It is a static `bilinear_type_v5` prior trained with ProcTHOR official-train
data and SpatialWorld train-only data. The runtime never reads target ground
truth. A searched container confirmed empty is removed and the remaining
candidates are reranked. Learned observation likelihood remains disabled
(`alpha=0`).

| Domain | Split | Hit@1 | Hit@3 | MRR | Mean rank |
|---|---|---:|---:|---:|---:|
| SpatialWorld | validation | 35.0% | 85.0% | 60.69% | 2.63 |
| SpatialWorld | test | 32.5% | 65.8% | 51.79% | 4.65 |
| ProcTHOR | validation | 62.0% | 98.0% | 79.33% | 1.49 |
| ProcTHOR | test | 60.5% | 97.0% | 78.10% | 1.60 |

The released data preserve scene-disjoint splits and independent
`(scene, target_id, parent_id)` queries. Test was read only after the checkpoint
was frozen. Details are in `PHASE_C_V2_FINAL_REPORT_2026-09-07.md`.

Experimental candidates that failed the SpatialWorld no-regression gate are
kept locally and are deliberately excluded from the GitHub release.
