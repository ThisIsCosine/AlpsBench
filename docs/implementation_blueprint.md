# Implementation Blueprint

## Purpose

This file is the working spec for the repository refactor.

It has two jobs:

- define the target public repository shape
- act as a checklist for the implementation thread

The goal is not abstract "organization."
The goal is a clean benchmark repository with a small public surface, a strict
public/private boundary, and a directory layout that is obvious at a glance.

This file is the implementation source of truth unless
`REPOSITORY_REFACTOR_ROADMAP.md` is explicitly updated first.

## Status Legend

- `[x]` completed in the repo
- `[ ]` not implemented yet
- `[-]` intentionally deferred to a later checkpoint

## Current Checkpoint Plan

- [x] C0. Create a dedicated `meta_script/` workspace for helper scripts
- [x] C1. Turn the blueprint into a checklist-driven implementation document
- [x] C2. Land the new public package skeleton under `src/benchmark/`
- [x] C3. Land the new public command entrypoints under `scripts/`
- [x] C4. Create the new public data layout: `examples/`, `dev/`, `validation/`, `test/`, `artifacts/`
- [x] C5. Add minimal public docs for usage, data, metrics, and repository structure
- [x] C6. Add minimal tests and CI for the new public scaffold
- [x] C7. Rewrite README so it only teaches the new public surface
- [x] C8. Wire `scripts/evaluate.py` to a real public evaluation flow
- [x] C9. Remove remaining maintainer-only legacy code from the public tree
- [x] C10. Remove old duplicated public paths once the new surface is verified

## Design Rules

### 1. Public and Private Must Be Decoupled

The repository must clearly separate:

- the public benchmark interface
- the private maintainer layer

Public code must not import from `hidden/`.

Allowed dependency direction:

- `hidden/` may depend on public code
- public code must not depend on `hidden/`

### 2. Data Layers Must Match Benchmark Use Cases

Public benchmark data is split by use case:

- `examples/`: tiny runnable examples; public reference outputs allowed
- `dev/`: public development set; public reference outputs allowed
- `validation/`: public holdout set; public reference outputs allowed
- `test/`: official benchmark inputs; no public reference outputs

This split exists to support onboarding and development without leaking official
test references.

### 3. Public Interface Must Stay Small

A public user should only need to understand:

- `benchmark_data/`
- `scripts/`
- `docs/`

A public user should not need to understand:

- research-era task runners
- memory selection internals
- raw export internals
- private scoring workflows

### 4. Public Commands Must Be Predictable

The public command surface should remain exactly:

- `scripts/build_data.py`
- `scripts/validate_data.py`
- `scripts/evaluate.py`
- `scripts/smoke_examples.py`
- `scripts/split_public_data.py`

All implementation detail belongs in `src/benchmark/`.

### 5. Task 4 Public Naming Must Be Simplified

The public benchmark surface exposes only:

- `ability1`
- `ability2`
- `ability3`
- `ability4`
- `ability5`

These internal names must not leak into public CLI or public docs:

- `ability2_general`
- `ability2_interaction`

## Target Directory Structure

```text
AlpsBench/
  README.md
  REPOSITORY_REFACTOR_ROADMAP.md
  .gitignore
  requirements.txt
  requirements-dev.txt

  adapter_example/
    README.md
    minimal_adapter.py

  benchmark_data/
    examples/
      task1/
        model_input.jsonl
        reference_output.jsonl
      task2/
        model_input.jsonl
        reference_output.jsonl
      task3/
        d100/
          model_input.jsonl
          reference_output.jsonl
        d300/
          model_input.jsonl
          reference_output.jsonl
        d500/
          model_input.jsonl
          reference_output.jsonl
        d700/
          model_input.jsonl
          reference_output.jsonl
        d1000/
          model_input.jsonl
          reference_output.jsonl
      task4_ability1/
        model_input.jsonl
        reference_output.jsonl
      task4_ability2/
        model_input.jsonl
        reference_output.jsonl
      task4_ability3/
        model_input.jsonl
        reference_output.jsonl
      task4_ability4/
        model_input.jsonl
        reference_output.jsonl
      task4_ability5/
        model_input.jsonl
        reference_output.jsonl
    dev/
      task1/
        model_input.jsonl
        reference_output.jsonl
      task2/
        model_input.jsonl
        reference_output.jsonl
      task3/
        d100/
          model_input.jsonl
          reference_output.jsonl
        d300/
          model_input.jsonl
          reference_output.jsonl
        d500/
          model_input.jsonl
          reference_output.jsonl
        d700/
          model_input.jsonl
          reference_output.jsonl
        d1000/
          model_input.jsonl
          reference_output.jsonl
      task4_ability1/
        model_input.jsonl
        reference_output.jsonl
      task4_ability2/
        model_input.jsonl
        reference_output.jsonl
      task4_ability3/
        model_input.jsonl
        reference_output.jsonl
      task4_ability4/
        model_input.jsonl
        reference_output.jsonl
      task4_ability5/
        model_input.jsonl
        reference_output.jsonl
    validation/
      task1/
        model_input.jsonl
        reference_output.jsonl
      task2/
        model_input.jsonl
        reference_output.jsonl
      task3/
        d100/
          model_input.jsonl
          reference_output.jsonl
        d300/
          model_input.jsonl
          reference_output.jsonl
        d500/
          model_input.jsonl
          reference_output.jsonl
        d700/
          model_input.jsonl
          reference_output.jsonl
        d1000/
          model_input.jsonl
          reference_output.jsonl
      task4_ability1/
        model_input.jsonl
        reference_output.jsonl
      task4_ability2/
        model_input.jsonl
        reference_output.jsonl
      task4_ability3/
        model_input.jsonl
        reference_output.jsonl
      task4_ability4/
        model_input.jsonl
        reference_output.jsonl
      task4_ability5/
        model_input.jsonl
        reference_output.jsonl
    test/
      task1/
        model_input.jsonl
      task2/
        model_input.jsonl
      task3/
        d100/
          model_input.jsonl
        d300/
          model_input.jsonl
        d500/
          model_input.jsonl
        d700/
          model_input.jsonl
        d1000/
          model_input.jsonl
      task4_ability1/
        model_input.jsonl
      task4_ability2/
        model_input.jsonl
      task4_ability3/
        model_input.jsonl
      task4_ability4/
        model_input.jsonl
      task4_ability5/
        model_input.jsonl
    artifacts/
      raw_export_index.json
      build_summary.json
      example_smoke_report.json
      split_manifest.json

  scripts/
    build_data.py
    validate_data.py
    evaluate.py
    smoke_examples.py
    split_public_data.py

  src/
    benchmark/
      __init__.py
      data.py
      runner.py
      metrics.py
      reports.py
      tasks/
        __init__.py
        task1.py
        task2.py
        task3.py
        task4.py

  docs/
    usage.md
    data.md
    evaluation_matrix.md
    metrics.md
    prediction_contract.md
    leaderboard.md
    repository_structure.md

  tests/
    test_data_validation.py
    test_evaluate.py
    test_examples_smoke.py
    test_metrics.py

  hidden/
    private_gold/
    dev_tools/
    legacy_research/
    local_configs/
    notebooks/
    scratch/

  meta_script/
```

## Responsibilities

### `benchmark_data/`

This is the only public data surface.

- `examples/`: tiny runnable samples
- `dev/`: public development set
- `validation/`: public validation set
- `test/`: official public test inputs only
- `artifacts/`: build summaries and validation products

### `scripts/`

This is the only public command surface.

- `build_data.py`: build the public data layout
- `validate_data.py`: validate public schema and split rules
- `evaluate.py`: public evaluation entrypoint
- `smoke_examples.py`: smoke check the shipped examples
- `split_public_data.py`: maintainer split utility for released tracks

### `src/benchmark/`

This is the public implementation layer.

- `data.py`: path rules, split rules, schema validation, build helpers
- `runner.py`: public orchestration
- `metrics.py`: public metric definitions and scoring policy surface
- `reports.py`: shared file IO and reporting helpers
- `tasks/`: task-specific public pipeline descriptors

### `hidden/`

This is the private maintainer layer.

It must be ignored by git and must not be imported by public code.

### `meta_script/`

This is the only place where agent-authored helper scripts should live.

Rule:

- if a helper script is created only to inspect, migrate, or bootstrap the repo,
  it goes in `meta_script/`
- do not scatter temporary scripts across `scripts/`, `src/`, or repo root

## File Contracts

### `model_input.jsonl`

This file contains only model-visible inputs.

It may contain:

- prompts
- dialogue context
- candidate memory sets
- metadata needed to run the benchmark

It must not contain public answers for the official test set.

### `reference_output.jsonl`

This file contains public references used for examples, dev, or validation scoring.

It is allowed only in:

- `benchmark_data/examples/`
- `benchmark_data/dev/`
- `benchmark_data/validation/`

It must not appear in the public `test/` tree.

## Do / Don't

### Do

- keep public names short and literal
- keep only one public path for each concept
- wrap old code before deleting it
- make new public commands stable first
- document every public directory in plain language

### Don't

- add compatibility layers to the new public surface
- expose research-era names in public docs
- leak hidden gold into public files
- mix helper scripts into public directories
- rewrite the README before the public scaffold exists

## Implementation Order

1. Create `meta_script/`
2. Create `src/benchmark/`
3. Create public `scripts/*`
4. Build the new public data layout
5. Add public docs
6. Add tests and CI
7. Rewrite README
8. Wire real public evaluation
9. Move private code out of the public path
10. Remove old duplicated surface

## Acceptance Criteria For The Next Thread

The next implementation checkpoint is acceptable if all of the following are
true:

- `meta_script/` exists and helper scripts are placed there
- `src/benchmark/` exists as the public package root
- `scripts/build_data.py` and `scripts/validate_data.py` work on the new layout
- `benchmark_data/examples`, `benchmark_data/dev`, `benchmark_data/validation`,
  `benchmark_data/test`, and `benchmark_data/artifacts` all exist
- `test/` exposes no public `reference_output.jsonl`
- public docs describe only the new public surface
- public code does not import `hidden/`

The public evaluation flow is allowed to remain partial until checkpoint `C8`.
