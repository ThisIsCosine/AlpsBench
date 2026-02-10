Task 1: Memory Extraction

This folder contains the task1 pipeline used to evaluate memory extraction from
dialogue. The flow is:

1) Generator reads benchmark/dev records and builds probes that include:
   - dialogue
   - query asking a model to extract memories
   - ground-truth memories (kept only for evaluation)
2) Evaluator runs selected models to extract memories from the probe dialogue.
3) Curator scores model outputs vs ground truth and records results.

Run

From repo root:
  python -m src.agents.task_runner task1

Inputs

- Benchmark records: benchmark/dev/**/*.jsonl
- Config: configs/api.json (model endpoints and keys)

Outputs

Runs are written to:
  runs/task1_batch/<UTC timestamp>/

Key files:
- probes.jsonl     Generated probes
- reports.jsonl    Model outputs (memory_items)
- decisions.jsonl  Curator scores
- dataset.jsonl    Probe + ground-truth dataset snapshot
- events.jsonl     Log events
- errors.jsonl     Errors (if any)
