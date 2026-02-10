# Task 3: Memory Retrieval (Generator + Judger + Evaluator + Grader)

This directory implements the end-to-end **Task 3 memory retrieval benchmark pipeline**, including:

- **Probe construction** (query generation + quality control)
- **Candidate set assembly** (selected memory + distractors)
- **Evaluated-model inference** (answer + attribution + rationale)
- **Automatic grading** and **artifact logging** for reproducibility and auditing

Primary entry points:

- `run_task3_from_jsonl_dir()` in `src/agents/task_runner.py` (batch runner / quickstart)
- `run_task3_pipeline()` in `src/agents/task3_memory_retrieval/curator.py` (Task 3 orchestration)

---

## Architecture at a Glance

Task 3 is structured as a modular pipeline with explicit inputs/outputs at each stage. The core design principle is **separating generation, quality control, evaluation, and grading**, and persisting every intermediate artifact to JSONL to support replay and analysis.

### Data model (high level)

For each seed example:

- **Inputs (seed-time)**: `dialogue` + `selected_memory` (and metadata)
- **Model-time input**: `query` + `candidate_memories` (selected + distractors; shuffled)
- **Evaluated-model output**: `answer` + `selected_memory_id` + `reason`
- **Grading output**: correctness and/or decision signals (stored in `decisions.jsonl`)

---

## Pipeline Stages (Responsibilities and Contracts)

### 1) Generator (`generator.py`)

- **Responsibility**: synthesize a query that is *not trivially copyable* from the selected memory and may require multi-hop reasoning.
- **Consumes**: `dialogue`, `selected_memory`
- **Produces**: `query` (JSON)

### 2) Judger / QC gate (`curator.py` → `Task3Judger`)

- **Responsibility**: enforce probe quality before evaluation.
- **Checks include**:
  - **Leakage**: whether the query reveals the selected memory’s answer/key facts
  - **Dependency**: whether the query actually requires the selected memory (not guessable)
  - **Rewrite loop**: if failing QC, request rewrite and retry

### 3) Candidate Assembly + Distractor Injection (`curator.py` → `run_task3_pipeline`)

- **Responsibility**: build `candidate_memories` by combining:
  - exactly **one** selected memory
  - \(N=\) `distract_n` distractors sampled from an external pool
- **Randomization**: shuffle the candidate list so the selected memory position is not fixed.

### 4) Evaluator (evaluated model) (`evaluator.py`)

- **Responsibility**: call the evaluated model with a fixed input schema and parse a structured JSON response.
- **Consumes**: `query`, `candidate_memories`
- **Produces**: `answer`, `selected_memory_id`, `reason`

### 5) Grader (`curator.py` → `Task3Grader`)

- **Responsibility**: score the evaluated model’s output, prioritizing judge-based signals when available and falling back to lightweight heuristics when necessary.

### 6) Artifact Logging (runs directory)

Every run writes a fully traceable record to a timestamped directory under `runs/task3_batch/`.

---

## What the Evaluated Model Can See

By default, **the evaluated model does not receive `dialogue`** (to focus on memory retrieval rather than dialogue grounding). The model is given:

- `query`
- `candidate_memories` (each entry contains `memory_id`, `value`, and optionally `label`)

Note:

> You may still see `dialogue` in `dataset.jsonl` for audit/replay; it is not necessarily forwarded to the evaluated model.

---

## Key Control Variable: Number of Distractors (`distract_n`)

`distract_n` is the most important difficulty knob:

- too small → search space is tiny; retrieval becomes trivial
- too large → noise increases; outputs may become unstable (especially with cross-category distractors)

### Distractor source (pragmatic trade-off)

In many real datasets, a single user does not have a large number of same-category memories. To keep the benchmark runnable, distractors may be drawn from **other records / other categories**, which can make retrieval easier if semantic distance becomes too large. Treat `distract_n` as a sensitivity-analysis variable.

---

## Running (Windows PowerShell)

From the repository root:

```powershell
$env:TASK3_JSONL_PATH="data\wildchat\memories\selected\task4_final_with_selected_memory_id"
python src\agents\task_runner.py task3 --samples 10 --distractors 20
```

Arguments:

- `--samples`: number of seeds to process
- `--distractors`: number of distractors per seed (i.e., `distract_n`)

Equivalent environment-variable form:

```powershell
$env:TASK3_DISTRACTORS="20"
python src\agents\task_runner.py task3 --samples 10
```

---

## Ablation: Label On vs. Label Off (Recommended)

### Motivation

This ablation tests whether models exploit the category label (`label`) as a shortcut instead of performing semantic retrieval over the `value` field.

### Label OFF

```powershell
$env:TASK3_INCLUDE_LABEL="0"
python src\agents\task_runner.py task3 --samples 10 --distractors 20
```

### Label ON (control)

```powershell
$env:TASK3_INCLUDE_LABEL="1"
python src\agents\task_runner.py task3 --samples 10 --distractors 20
```

Note:

> `dataset.jsonl` matches the model-visible schema: when `TASK3_INCLUDE_LABEL=0`, `label` is omitted from both `candidate_memories` and `selected_memory`.

---

## Outputs (Artifacts)

Each run writes to:

`runs/task3_batch/<UTC timestamp>/`

Key files:

- `dataset.jsonl`: full per-example snapshot (e.g., `query`, `candidate_memories`, audit fields)
- `probes.jsonl`: probe construction/QC artifacts (generation + rewrites)
- `reports.jsonl`: evaluated model outputs (raw + parsed)
- `decisions.jsonl`: QC/grading decisions and scores
- `events.jsonl`: process-level logs (including retries/rewrites)
- `errors.jsonl`: exceptions (e.g., parse errors, network failures)
