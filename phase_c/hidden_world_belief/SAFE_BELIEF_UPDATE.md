# Safety-constrained sequential hidden-location belief

For target `q`, room `r`, and receptacle candidate `c`, the frozen ranker
provides the initial logit `s0(c)`.  New observations contribute learned
log-likelihood ratios `g_theta(o_t, c)`:

```text
shadow_logit_t(c) = s0(c) + alpha * sum_j g_theta(o_j, c)
shadow_belief_t(c) = softmax(shadow_logit_t(c))
```

The runtime keeps this learned posterior in shadow mode.  The default `exact`
safety projection accepts it only when its complete deterministic ordering is
identical to the frozen-prior ordering.  Otherwise deployment falls back to
`s0`.  Thus `alpha=0` and the projected learned path are ranking-equivalent to
the validated static baseline.

After a receptacle has been opened, sufficiently observed, and confirmed not
to contain the target, the safe updater applies:

```text
b_{t+1}(c_searched) = 0
b_{t+1}(c) = b_t(c) / (1 - b_t(c_searched)), c != c_searched
```

Under a reliable negative observation, removing a false candidate cannot
worsen the rank of the true remaining candidate. Candidate tracks persist
across temporary perception misses, and observation IDs are deduplicated so a
single frame is not counted repeatedly as independent evidence.

## Reproduce

```bash
PYTHONPATH=runtime_overlay:. pytest -q \
  phase_c/hidden_world_belief/test_safe_belief.py \
  runtime_overlay/mllm_base_agent/agent/test_hidden_location_advisor.py

sbatch phase_c/hidden_world_belief/train_visible_object_context.slurm

/apps/local/anaconda3/bin/python -m \
  phase_c.hidden_world_belief.eval_safe_belief_update \
  --checkpoint phase_c/hidden_world_belief/checkpoints/visible_object_context_shadow_v2.pt \
  --data phase_c/hidden_world_belief/data/observation_task_queries_native_visible_context_v1 \
  --device cpu
```

`visible_object_context_shadow_v2.pt` is explicitly marked
`shadow_only_not_deployable`. The validated deployment configuration keeps
`learned_alpha: 0.0` and `safety_projection: exact` until the context gate
passes on both task-context and unseen-context validation.
