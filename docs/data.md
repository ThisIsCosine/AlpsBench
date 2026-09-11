# Data Design

## Dataset Version

The current public dataset release identifier is `v4`.

- `benchmark_data/` in this repository is the released public layout for `v4`
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
- `task4_ability1`
- `task4_ability2`
- `task4_ability3`
- `task4_ability4`
- `task4_ability5` (reserved and empty in v4)

On disk, the Task 3 release files are grouped under `task3/` with one
subdirectory for the released distractor level: `d100`.

Public CLI mapping:

- `--task task3` defaults to `task3_d100`
- `--task task3 --distractors 100` selects the only Task 3 track in v4
- `--task task4` requires `--ability ability1..ability5`

The source archive contains an empty `ability5.json`. Empty files are committed
for that reserved track so the ability namespace remains stable, but it cannot
be scored in v4. The archive does not contain Task 3 d300/d500/d700/d1000, so
older files for those tracks are deliberately excluded rather than mixed into
the v4 release.

### Task 3 candidate normalization

The v4 Task 3 source uses session-local IDs such as `m1` and samples candidates
with replacement, which produces duplicate IDs and duplicate memory content in
the same row. During import, AlpsBench:

1. deduplicates candidates by `(label, value)`;
2. fills each row back to one target plus 100 distinct candidates from the v4
   Task 1 memory pool using a deterministic session-ID hash;
3. assigns row-local IDs `c000` through `c100`; and
4. preserves the original ID as `source_memory_id`.

This makes `selected_memory_id` unambiguous without changing the public
prediction shape. The source anomaly counts and transformation policy are
recorded in `source_bundle_manifest.json`.

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
- `source_bundle_manifest.json`: source-file checksums, row counts, and the
  non-sensitive annotation audit summary

## Hidden Gold

Public `test` references are private.

- public `benchmark_data/test/` contains only `model_input.jsonl`
- private `test` references live under `hidden/private_gold/`
- public code must not import from `hidden/`

## Maintainer Utilities

Two scripts operate on the released data layout itself:

- `scripts/build_data.py`: refresh release manifests and required directories
- `scripts/split_public_data.py`: repartition public tracks into `dev`, `validation`, and `test`

To import the canonical v4 archive reproducibly:

```bash
python scripts/build_data.py --source /path/to/Alps_data_final_v4.zip --overwrite
python scripts/validate_data.py
```

The raw annotation audit file is read for provenance checks but is not copied
into the public layout because it contains information used to construct hidden
test references.

Normal benchmark users do not need either command to run evaluation.
