# Adapter Example

This directory contains adapters for AlpsBench `--predict-program`.

- `task1_api_adapter.py`: real Task 1 extraction using the original prompt and a
  configurable HTTP model endpoint. See [setup and usage](../docs/task1_extraction.md).
- `minimal_adapter.py`: standalone stubs illustrating the input/output contract
  for all four tasks. Its Task 1/2 functions intentionally return empty memories.

The minimal adapter does not depend on any internal repo modules. It:

- reads one JSON row from `stdin`
- writes one prediction JSON row to `stdout`
- follows the public prediction contract for `task1` to `task4`

## Quick Run

From the repository root:

```bash
python scripts/evaluate.py --task task4 --ability ability2 --split examples --predict-program python --predict-arg adapter_example/minimal_adapter.py
```

On Windows, if you change the adapter to print non-ASCII text directly, prefer:

```bash
python scripts/evaluate.py --task task4 --ability ability2 --split examples --predict-program python --predict-arg -X --predict-arg utf8 --predict-arg adapter_example/minimal_adapter.py
```

## What To Edit

Open `minimal_adapter.py` and replace the task-specific stub functions:

- `predict_task1`
- `predict_task2`
- `predict_task3`
- `predict_task4`

The rest of the file is just I/O glue and schema-safe output formatting.

## Contract Reminder

The evaluator sends one `model_input.jsonl` row to `stdin`.

Your program must return exactly one JSON object:

- `task1`: `{"benchmark_id": "...", "memory_items": [...]}`
- `task2`: `{"benchmark_id": "...", "memory_items": [...]}`
- `task3`: `{"benchmark_id": "...", "answer": "...", "reason": "...", "selected_memory_id": "..."}`
- `task4`: `{"benchmark_id": "...", "answer": "...", "used_memory_fact": "..."}`

For the full public schema, see `docs/prediction_contract.md`.
