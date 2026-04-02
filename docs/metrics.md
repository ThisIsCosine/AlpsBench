# Metrics

The public metric surface is implemented in `src/benchmark/metrics.py` and
used by `scripts/evaluate.py`.

## Public Scoring Availability

- `examples`: local scoring available
- `dev`: local scoring available
- `validation`: local scoring available
- `test`: no public local scoring

For `test`, AlpsBench writes predictions and run metadata but does not score
publicly because no public references are released.

## Task Metrics

- `task1`: set-based memory-item precision, recall, F1, and exact-match rate
- `task2`: set-based updated-memory precision, recall, F1, and exact-match rate
- `task3_d100`, `task3_d300`, `task3_d500`, `task3_d700`, `task3_d1000`:
  public local proxy is selected-memory accuracy using `selected_memory_id`
  match; benchmark-side paper reporting treats Task 3 as a retrieval task and
  reports `precision` and `recall`
- `task4_ability1..5`: grounding proxy based on `used_memory_fact` match against the gold selected memory

## Metric Types

The public docs distinguish three layers:

- `exact` or deterministic metrics:
  Task 1 and Task 2 public local scoring, plus the public Task 3 memory-id
  match proxy
- `judge-driven` benchmark-side metrics:
  Task 3 memory-usage judging and Task 4 ability-specific response judging
- `heuristic public proxies`:
  the current released Task 4 local scorer, which is intentionally simpler than
  the benchmark-side judge

For Task 4 specifically, the benchmark-side judge is ability-specific rather
than one shared scalar prompt. Public documentation describes the dimensions it
measures, while the released local scorer only exposes a grounding proxy.

## Task 4 Reporting Policy

Task 4 is not reported as one merged public benchmark score.

Officially:

- each ability is its own scored track
- each ability is ranked independently
- each ability is compared independently
- there is no public `Task4_overall` aggregation across abilities

This means the effective per-ability metric schema is:

- `track_id`: one of `task4_ability1` .. `task4_ability5`
- `ability_score`: the judge-driven score surface for that ability
- `dimension_scores`: the ability-specific subdimensions when exposed
- `judge_reasoning`: supporting explanation text when available

The exact field names can differ by ability, but the public evaluation policy is
uniform: treat each ability as a separate benchmark track rather than forcing a
shared cross-ability total.

## Release Vs Run Artifacts

`benchmark_data/` is the final public benchmark dataset release.

- committed dataset artifacts:
  `benchmark_data/examples/`, `dev/`, `validation/`, `test/`, and
  `benchmark_data/artifacts/`
- generated run artifacts:
  `runs/public/.../predictions.jsonl`, `scores.jsonl`, `summary.json`, and
  `run_manifest.json`
- submission artifact for hidden `test` evaluation:
  generated `predictions.jsonl`

## Prediction Contract

The released public scorer expects task-specific prediction rows, not full
`reference_output.jsonl` rows.

- `task1`: `benchmark_id`, `memory_items`
- `task2`: `benchmark_id`, `memory_items`
- `task3_d100`, `task3_d300`, `task3_d500`, `task3_d700`, `task3_d1000`:
  `benchmark_id`, `answer`, `reason`, `selected_memory_id`
- `task4`: `benchmark_id`, `answer`, `used_memory_fact`

`reference_output.jsonl` is public gold for local-reference splits, not the
required submission format.

## Run Outputs

For `examples`, `dev`, and `validation`, `scripts/evaluate.py` writes:

- `predictions.jsonl`
- `scores.jsonl`
- `summary.json`
- `run_manifest.json`

For `test`, it writes:

- `predictions.jsonl`
- `summary.json`
- `run_manifest.json`

without public scoring.
