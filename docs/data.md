# Data Design

## Dataset Version

The current public dataset release identifier is `v1`.

- `benchmark_data/` in this repository is the released public layout for `v1`
- `benchmark_data/artifacts/` contains the release manifests for that layout
- future releases should bump this identifier when committed splits, schemas,
  or row sets change

## Public Split Layout

The public benchmark data is split by use case:

- `benchmark_data/examples/`
- `benchmark_data/dev/`
- `benchmark_data/validation/`
- `benchmark_data/test/`
- `benchmark_data/artifacts/`

Purpose of each layer:

- `examples/`: tiny runnable samples for onboarding and schema inspection
- `dev/`: public development split with public references
- `validation/`: public holdout split for self-checking before hidden-test submission
- `test/`: official public inputs only
- `artifacts/`: release manifests, split summaries, and metadata

## Track Names

The released public tracks are:

- `task1`
- `task2`
- `task3_d100`
- `task3_d300`
- `task3_d500`
- `task3_d700`
- `task3_d1000`
- `task4_ability1`
- `task4_ability2`
- `task4_ability3`
- `task4_ability4`
- `task4_ability5`

On disk, the Task 3 release files are grouped under `task3/` with one
subdirectory per distractor level: `d100`, `d300`, `d500`, `d700`, and
`d1000`.

Public CLI mapping:

- `--task task3` defaults to `task3_d100`
- `--task task3 --distractors 300|500|700|1000` resolves to the corresponding
  `task3_d*` track
- `--task task4` requires `--ability ability1..ability5`

## File Contracts

### `model_input.jsonl`

Committed benchmark input rows.

- present in `examples/`, `dev/`, `validation/`, and `test`
- contains only model-visible task inputs plus benchmark metadata

### `reference_output.jsonl`

Committed benchmark gold rows.

- present only in `examples/`, `dev/`, and `validation`
- absent from the public `test/` tree
- used for local scoring and oracle checks on public-reference splits

### `predictions.jsonl`

User-produced run artifact.

- not part of the committed dataset layout
- written by `scripts/evaluate.py` under `runs/public/...`
- each row must match the corresponding sample `reference_output.jsonl` schema exactly
- this is the artifact users submit for hidden evaluation on `test`

## Split Policy

The released split policy is deterministic and track-local.

- each public track is repartitioned independently
- `dev` receives `1/5` of the rows
- `validation` receives `1/5` of the rows
- `test` receives the remaining `3/5`
- `examples` remains a tiny curated public sample layer

The exact committed counts live in `benchmark_data/artifacts/build_summary.json`.
The deterministic repartition metadata lives in
`benchmark_data/artifacts/split_manifest.json`.

## Artifacts

`benchmark_data/artifacts/` is part of the released dataset package.

Key files:

- `build_summary.json`: committed row counts for each split and track
- `public_layout_manifest.json`: expected committed public directories
- `example_smoke_report.json`: summary of the shipped examples split
- `raw_export_index.json`: note about the canonical public layout
- `split_manifest.json`: deterministic split-policy metadata

## Hidden Gold

Public `test` references are private.

- public `benchmark_data/test/` contains only `model_input.jsonl`
- private `test` references live under `hidden/private_gold/`
- public code must not import from `hidden/`

## Maintainer Utilities

Two scripts operate on the released data layout itself:

- `scripts/build_data.py`: refresh release manifests and required directories
- `scripts/split_public_data.py`: repartition public tracks into `dev`, `validation`, and `test`

Normal benchmark users do not need either command to run evaluation.
