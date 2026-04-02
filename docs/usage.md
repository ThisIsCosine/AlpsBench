# Usage

This document describes the released user workflow for AlpsBench.

## Typical Workflow

1. Validate the shipped data layout.

```bash
python scripts/validate_data.py
python scripts/smoke_examples.py
```

2. Inspect the example split and confirm your adapter can return prediction
   rows in the public task contract documented in `docs/prediction_contract.md`.
   If you are writing a new adapter from scratch, start from `adapter_example/`.

```bash
python scripts/evaluate.py --task task1 --split examples --oracle
python scripts/evaluate.py --task task3 --split examples --oracle
```

3. Run your model on `dev`.

```bash
python scripts/evaluate.py --task task1 --split dev --predictions my_task1_predictions.jsonl
python scripts/evaluate.py --task task4 --ability ability2 --split dev --predict-program python --predict-arg my_adapter.py
```

4. Lock your configuration and self-check on `validation`.

```bash
python scripts/evaluate.py --task task1 --split validation --predictions my_task1_predictions.jsonl
python scripts/evaluate.py --task task4 --ability ability2 --split validation --predict-program python --predict-arg my_adapter.py
```

5. Generate official `test` predictions.

```bash
python scripts/evaluate.py --task task1 --split test --predictions my_task1_predictions.jsonl
python scripts/evaluate.py --task task4 --ability ability2 --split test --predict-program python --predict-arg my_adapter.py
```

The public `test` split has no public references, so the evaluator packages
predictions without scoring them.

## Evaluation Sources

Exactly one prediction source must be selected for each run.

### 1. Prediction File

Use a prepared JSONL file containing one prediction row per input row.

```bash
python scripts/evaluate.py --task task2 --split dev --predictions my_task2_predictions.jsonl
```

### 2. External Adapter Program

Use this when you want AlpsBench to call your model row by row.

```bash
python scripts/evaluate.py --task task2 --split dev --predict-program python --predict-arg my_adapter.py
```

The repository includes a minimal standalone example in
`adapter_example/minimal_adapter.py`.

Contract:

- the evaluator sends one `model_input` row as JSON on stdin
- the adapter returns one prediction row as JSON on stdout
- that row must follow the task contract in `docs/prediction_contract.md`

`--predict-program` with repeated `--predict-arg` is the recommended interface.
It avoids shell quoting issues on Windows and PowerShell.

Windows note:

- AlpsBench sends adapter stdin as ASCII-safe JSON, so a plain `json.load(sys.stdin)` works even on legacy Windows code pages
- if your adapter writes non-ASCII characters directly to stdout, prefer `python -X utf8 my_adapter.py`
- if you use Python's default `json.dump(...)`, it emits ASCII-safe JSON and does not require extra encoding flags

### 3. Legacy Shell Command

```bash
python scripts/evaluate.py --task task2 --split dev --predict-command "python my_adapter.py"
```

This remains available for compatibility, but it is not the recommended public
path.

### 4. Oracle Mode

Oracle mode is only for public-reference splits.

```bash
python scripts/evaluate.py --task task1 --split examples --oracle
python scripts/evaluate.py --task task4 --ability ability3 --split validation --oracle
```

It is not available on `test`.

## Task And Split Naming

Supported splits:

- `examples`
- `dev`
- `validation`
- `test`

Supported tasks:

- `task1`
- `task2`
- `task3`
- `task4`

Public track mapping:

- `task3` defaults to `task3_d100`
- use `--distractors 100|300|500|700|1000` to select a different released Task
  3 distractor pool
- `task4` requires `--ability ability1..ability5`

Example:

```bash
python scripts/evaluate.py --task task4 --ability ability4 --split validation --predictions my_task4_predictions.jsonl
python scripts/evaluate.py --task task3 --distractors 700 --split validation --predictions my_task3_predictions.jsonl
```

## Output Files

Each run writes to `runs/public/<split>/<track>/` by default unless
`--output-dir` is supplied.

Files written for all runs:

- `run_manifest.json`
- `predictions.jsonl`
- `summary.json`

Additional file written for `examples`, `dev`, and `validation`:

- `scores.jsonl`

For `test`, `summary.json` states that no public scoring is available.

## Submission Guidance

- `reference_output.jsonl` is benchmark ground truth answer and is committed only for `examples`, `dev`, and `validation`
- prediction files are user-produced run artifacts
- user-produced prediction rows must follow the task-specific contract in `docs/prediction_contract.md`
- for `test`, the artifact to send for hidden evaluation is the produced `predictions.jsonl`
