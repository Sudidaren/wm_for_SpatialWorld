# Evaluation tables (auto-built from existing runs)

> `-` = not measured yet. Coverage column shows how much of the
> planned batch is already decided (nulls excluded).

## Table 5.4.1 — Model x environment

| Method | Env | Success | Failure | Null | Total | TSR | Null rate | Completion | SE_success | SE_all | Avg steps (succ) | Avg steps (fail) | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-VL-30B-A3B (BF16, vLLM) | AI2-THOR | 20 | 291 | 0 | 311 | 6.4% | 0.0% | 6.4% | 74.9% | 33.3% | 3.6 | 19.0 | 311/311 |
| Qwen3-VL-30B-A3B (BF16, vLLM) | ProcTHOR | 0 | 127 | 0 | 127 | 0.0% | 0.0% | 0.0% | - | 67.1% | - | 17.6 | 127/127 |
| Gemini 3.1 Pro (frozen v1) | AI2-THOR | 29 | 70 | 6 | 311 | 29.3% | 5.7% | 9.3% | 55.3% | 24.8% | 10.1 | 29.2 | 99/311 (partial) |
| Gemini 3.1 Pro (frozen v1) | ProcTHOR | 0 | 87 | 1 | 127 | 0.0% | 1.1% | 0.0% | - | 18.6% | - | 43.2 | 87/127 (partial) |

## Table 5.4.3 — Failure taxonomy (decided non-success only)

| Method | Env | DONE-but-wrong | Max steps | Consecutive action failures | API/env null | Other |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-VL-30B-A3B (BF16, vLLM) | AI2-THOR | 91 | 172 | 0 | 0 | 28 |
| Qwen3-VL-30B-A3B (BF16, vLLM) | ProcTHOR | 13 | 21 | 93 | 0 | 0 |
| Gemini 3.1 Pro (frozen v1) | AI2-THOR | 15 | 49 | 0 | 6 | 6 |
| Gemini 3.1 Pro (frozen v1) | ProcTHOR | 26 | 52 | 7 | 1 | 2 |

## World-model A/B probe (not part of the frozen batch)

| Task | Arm | Success | Steps | Prompt tokens |
|---|---|---|---:|---:|
| ai2thor03022 | base | True | 4 | 16494 |
| ai2thor03022 | hint | True | 4 | 16757 |
| ai2thor04065 | base | False | 10 | 61564 |
| ai2thor04065 | hint | False | 16 | 140845 |
| ai2thor04039 | base | True | 4 | 16815 |
| ai2thor04039 | hint | False | 11 | 76439 |
| ai2thor03007 | base | False | 7 | 39364 |
| ai2thor03007 | hint | True | 3 | 11676 |

## Still empty (needs data)

- GPT-5 x {AI2-THOR, ProcTHOR, VirtualHome, Overall}
- VirtualHome x {Qwen, Gemini}
- GPT-5 + WingmanWM x {AI2-THOR, ProcTHOR, Overall}
- DreamerV3 / DIAMOND baselines
- Table 5.4.2 category breakdown (TSR)
- Table 5.4.4 ablations (LLM only / +perception+memory / +gate)
- Table 5.4.5 cost and runtime
