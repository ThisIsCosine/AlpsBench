# Repository Structure

This file describes the intended public structure of the repository.

## Public Surface

- `benchmark_data/`: public benchmark data
- `adapter_example/`: standalone adapter skeleton for `--predict-program`
- `scripts/`: public command entrypoints
- `docs/`: public benchmark documentation
- `src/benchmark/`: public implementation layer

## Private Surface

- `hidden/`: maintainer-only code, gold files, local tools, and notebooks

Public code must not import from `hidden/`.

## Helper Script Surface

- `meta_script/`: temporary or migration helper scripts created during refactor

If a helper script becomes part of the public interface, it should be moved into
`scripts/`.

## Public Script Surface

Released user-facing commands:

- `scripts/validate_data.py`
- `scripts/smoke_examples.py`
- `scripts/evaluate.py`

Maintainer-oriented data-layout commands:

- `scripts/build_data.py`
- `scripts/split_public_data.py`
