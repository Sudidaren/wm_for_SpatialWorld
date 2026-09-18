# Depth head A/B on the held-out evaluation rooms

Protocol: `tools/depth_error_analysis.py --rooms test` over the 81 evaluation
rooms in `/mnt/d/lightwm_data`.  Those rooms are never trained on; they are the
clean report set (`data/eval_rooms.json`).  Metric: `|pred - gt|` at the depth
of the detector box centre, where `gt` is the ground-truth depth PNG.

Heads compared

| tag | checkpoint | input resolution |
|---|---|---|
| `old` | `checkpoints/small_objects_20260910/dense_depth_best.pt` | 336 |
| `v2` | `checkpoints/depth_v2_20260918/dense_depth_v2_best.pt` | 448 |

`old_head.*` / `new_head.*` are a 400-frame sweep (dominated by the classic
AI2-THOR homes).  `strat_*.txt` is the family-stratified re-run (up to 300
frames per family, `--per-family-frames 300`), which is the number to quote.

Headline (400-frame sweep, 5004 objects):

| head | median \|err\| | median bias | median pred/gt | Spearman |
|---|---|---|---|---|
| old 336 | 0.270 m | -0.232 m | 0.862 | 0.903 |
| v2 448 | 0.739 m | -0.738 m | 0.619 | 0.838 |
| v2 448, after one global log-linear calibration | 0.253 m | -0.049 m | 0.975 | — |

Reading: the v2 head is not mis-shaped, it is *mis-scaled* on this domain
(fitted `log(gt) = 0.964 * log(pred) + 0.463`, i.e. a 0.62x constant factor).
It was trained without a single classic-home room, and classic homes carry
~58% of the objects visible in the evaluation rooms.  The `old` head looks
better only because its training pool contained 81 of the 89 rooms it was
first built from - that is leakage, not accuracy.

Regenerate:

```bash
PY=python3   # any env with torch + numpy + pillow
for tag_ckpt_res in "old:checkpoints/small_objects_20260910/dense_depth_best.pt:336" \
                    "v2:checkpoints/depth_v2_20260918/dense_depth_v2_best.pt:448"; do
  tag=${tag_ckpt_res%%:*}; rest=${tag_ckpt_res#*:}
  ck=${rest%:*}; res=${rest##*:}
  $PY tools/depth_error_analysis.py --rooms test --frames 2000 \
      --per-family-frames 300 --ckpt $ck --resolution $res \
      --json results/depth_head_ab_20260918/strat_${tag}.json
done
```
