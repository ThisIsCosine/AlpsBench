# AlpsBench Benchmark Release Roadmap

## Purpose

This document is the single source of truth for turning the current research-style codebase into a reusable, benchmark-grade, easy-to-understand GitHub repository.

Use this file as the long-running control board:

- When a task is finished, change `- [ ]` to `- [x]`.
- If you want a visual strike-through style, also wrap the task text in `~~ ~~`.
- Do not mark a task as done unless its acceptance criteria are met.

---

## Target State

The final repository should satisfy all 5 conditions below:

1. A new user can install dependencies and run at least one example for each task without guessing hidden paths.
2. A researcher can understand the dataset schema, split strategy, and metric formulas from the docs alone.
3. A practitioner can evaluate a new model with a stable CLI and stable output format.
4. A maintainer can validate changes via tests and CI before publishing.
5. A public GitHub visitor can immediately understand what the benchmark is, how to use it, and what is included in the repo vs hosted externally.

---

## Current Assets We Can Build On

- [X]  Core task code exists under `src/agents/` for Task 1, Task 2, Task 3, and Task 4.
- [X]  Example benchmark data exists under `benchmark_data/examples/`.
- [X]  Initial rubric drafts already exist under `docs/rubrics/`.
- [X]  A shared model API layer already exists in `src/agents/shared.py`.
- [X]  Several baseline or helper scripts already exist under `scripts/`.
- [X]  A structured exported data bundle now exists under `Alps_data_v1/`.

These are useful foundations, but they are not yet enough for a clean public benchmark repository.

---

## Canonical Data Source

Until we normalize and repackage the benchmark, treat `Alps_data_v1/` as the current source of truth for public-data reconstruction.

Observed structure:

- `Alps_data_v1/benchmark_construction_check_data/`
  - Per-task session exports with `manifest.json` and `sessions/<id>/item.json`
  - Contains benchmark-side records, original task payloads, and gold data
- `Alps_data_v1/LLM_as_judge_Human_Alignment_data/`
  - Per-task session exports with model outputs, judge outputs, judge scores, and final scores
  - Best suited for baseline/reference analysis, not the primary public benchmark input format
- `Alps_data_v1/task1_evaluation_data/`
  - Task-1-specific flattened evaluation bundle with dialogue, extraction query, gold answer, model outputs, and judge results
  - Useful as analysis material and metric examples
- `Alps_data_v1/task1/`, `task2/`, `task3/`
  - Currently appear to be log or pool-summary style exports rather than the main benchmark release format
- `Alps_data_v1/task4/`
  - Organized as `ability1` through `ability5`

Concrete implications:

- The future public repository should be built around `Alps_data_v1`, not around the older internal `data/...` and `benchmark/...` assumptions in the current code.
- Task 4 public docs should align to `ability1` through `ability5` as exported in `Alps_data_v1`.
- The `LLM_as_judge_Human_Alignment_data` branch should become an `analysis` or `reference_results` asset, not the main user-facing benchmark input path.
- We should create a normalized public dataset layer from `Alps_data_v1`, rather than asking users to read raw export internals directly.

---

## Recommended Execution Order

Work in this order unless there is a strong reason to deviate:

1. `P0` Make the repo honest and runnable.
2. `P1` Make the repo understandable.
3. `P1` Make the data publishable and structured.
4. `P1` Make the metrics explicit and stable.
5. `P2` Make the baselines and benchmark protocol reusable.
6. `P2` Make the engineering and release process reliable.

---

## Status Notes

This section explains what has already been completed, how it was completed, and what is still open or blocked.

### Workstream Snapshot

These counts reflect the current number of checked checklist items inside each workstream section.


| Workstream                                              | Checked | Total | Current status    |
| ------------------------------------------------------- | ------: | ----: | ----------------- |
| A. Repository Architecture and Code Cleanup             |       5 |    16 | Partially started |
| B. Public README and User Onboarding                    |      12 |    12 | Complete          |
| C. Dataset Packaging, Splits, and Schemas               |      26 |    26 | Complete          |
| D. Metrics, Formulas, and Evaluation Docs               |      18 |    18 | Complete          |
| E. CLI, Scripts, and Output Conventions                 |      10 |    10 | Complete          |
| F. Baselines and Leaderboard Protocol                   |       7 |    10 | Strong progress   |
| G. Testing, CI, and Engineering Quality                 |       2 |     7 | Partially started |
| H. Open-Source Readiness and Release Governance         |       6 |    10 | Strong progress   |

### Completed and Verified

1. `Public README and onboarding docs`
   Completed by replacing the paper-style root README with a public benchmark README, then adding `docs/getting_started.md`, `docs/configuration.md`, `docs/repository_structure.md`, `docs/troubleshooting.md`, and `docs/faq.md`.
   Verification: the public docs now point to files that exist in this repo and reflect the current `Alps_data_v1/` plus `benchmark_data/normalized/` layout.
2. `Canonical raw-data source decision`
   Completed by treating `Alps_data_v1/benchmark_construction_check_data/` as the current canonical benchmark reconstruction source and documenting the role of the other bundles in `docs/benchmark_data_architecture.md`.
   Verification: the roadmap, root README, and data-architecture docs now consistently describe benchmark input data, judge-alignment data, and auxiliary analysis data as separate layers.
3. `Normalized benchmark reconstruction`
   Completed by adding `scripts/data/index_raw_export_bundle.py`, `scripts/data/build_normalized_benchmark.py`, and `scripts/data/validate_normalized_benchmark.py`, then generating the normalized files under `benchmark_data/normalized/`.
   Verification: `benchmark_data/normalized/build_summary.json` and `benchmark_data/build_artifacts/raw_export_index.json` were generated from `Alps_data_v1/`, and `scripts/data/validate_normalized_benchmark.py` passes on the normalized output.
4. `Example-data smoke path`
   Completed by adding `scripts/smoke/check_example_data.py` so that each shipped example task can be checked without API credentials.
   Verification: `scripts/smoke/check_example_data.py --task all --limit 2` was run with the pinned Anaconda environment and wrote `benchmark_data/build_artifacts/example_data_smoke_report.json`.
5. `Optional baseline dependency split`
   Completed by keeping `requirements.txt` focused on the current public data workflow and moving Task 3 baseline extras into `requirements-baselines.txt`.
   Verification: the public docs now explain when `requirements-baselines.txt` is needed, and the split is reflected in the quickstart and troubleshooting docs.
6. `Quickstart path cleanup`
   Completed by updating `src/agents/task_runner.py` quickstart defaults to point at repo-local example files instead of unpublished internal paths, and by fixing the `TASK_QUICKSTARTS` error message bug.
   Verification: `src/agents/task_runner.py` now resolves Task 1 through Task 4 quickstarts against `benchmark_data/examples/`, and the file passes `py_compile`.

## Workstream A: Repository Architecture and Code Cleanup

### A1. Public Entry Points

- [ ]  Define one canonical public CLI entrypoint for evaluation.
- [ ]  Define one canonical public CLI entrypoint for dataset preparation or validation.
- [ ]  Remove ambiguity between `scripts/` runners and `src/agents/task_runner.py`.

### A2. Path Hygiene

- [X]  Remove hard-coded internal directories such as unpublished `data/...` and `benchmark/...` assumptions from default public flows.
- [ ]  Replace hidden internal defaults with explicit arguments, environment variables, or example-safe defaults.
- [ ]  Add safe path validation where file discovery currently assumes internal layout.

### A3. Configuration Hygiene

- [X]  Add `configs/api.example.json`.
- [X]  Document every config field in `configs/api.json`.
- [ ]  Define which options are required, optional, and advanced.
- [ ]  Standardize environment variable fallback behavior.

### A4. Dependency and Packaging

- [ ]  Audit all imported third-party packages used by scripts and core code.
- [X]  Update `requirements.txt` so it actually matches the codebase.
- [X]  Separate runtime dependencies from optional baseline dependencies if needed.

### A5. Code Correctness Cleanup

- [ ]  Fix known broken or suspicious code paths before public release.
- [ ]  Normalize encoding and eliminate garbled text in user-facing docs and prompts where possible.
- [ ]  Audit Task 4 runner and curator code for syntax, output-schema, and control-flow consistency.

Acceptance criteria:

- There is one clear public way to run the benchmark.
- Public code paths do not depend on unreleased local directories.

---

## Workstream B: Public README and User Onboarding

### B1. Root README Rewrite

- [X]  Replace the current paper-style README with a public benchmark-style README.
- [X]  Add a clear "What is included in this repo" section.
- [X]  Add a clear "What is not included in this repo" section.
- [X]  Add a 5-minute quickstart at the top.
- [X]  Add links to deeper docs instead of overloading the root README.

### B2. Essential Docs

- [X]  Add `docs/getting_started.md`.
- [X]  Add `docs/configuration.md`.
- [X]  Add `docs/repository_structure.md`.
- [X]  Add `docs/troubleshooting.md`.
- [X]  Add `docs/faq.md`.

### B3. User Journeys

- [X]  Document the "I only want to run the examples" path.
- [X]  Document the "I want to evaluate my own model" path.

Acceptance criteria:

- A visitor can choose a path and follow it without reading source code first.

---

## Workstream C: Dataset Packaging, Splits, and Schemas

### C1. Canonical Split Design

- [X]  Decide whether `Alps_data_v1/benchmark_construction_check_data/` becomes the canonical benchmark source for public packaging.
- [X]  Decide whether `Alps_data_v1/LLM_as_judge_Human_Alignment_data/` becomes `analysis/reference_results/`.
- [X]  Decide whether `Alps_data_v1/task1_evaluation_data/` becomes an auxiliary analysis bundle instead of benchmark core data.
- [X]  Decide and document the official split policy: `train/dev/test`, `dev/test`, or task-specific variants.
- [X]  State whether the GitHub repo contains only examples or a reduced public split.
- [X]  State where the full benchmark lives and how it should be downloaded.
- [X]  Add version identifiers for dataset releases.

### C2. Dataset Layout

- [X]  Define a normalized repo-facing layout derived from `Alps_data_v1` instead of exposing raw export folders as-is.
- [X]  Define a clean public dataset directory structure.
- [X]  Decide whether `benchmark_data/examples/` remains sample-only or becomes a real split subset.
- [X]  Standardize per-task filenames and suffixes.
- [X]  Standardize naming for distractor settings in Task 3.

### C3. Schema Docs

- [X]  Document the Task 1 input and output schema.
- [X]  Document the Task 2 input and output schema.
- [X]  Document the Task 3 input and output schema.
- [X]  Document the Task 4 ability schemas.

### C4. Dataset Validation

- [X]  Add a script to validate JSON or JSONL schema for each task.
- [X]  Add checks for missing required fields.
- [X]  Add checks for duplicate IDs.
- [X]  Add checks for split leakage where applicable.
- [X]  Add checks for Task 3 candidate set correctness.

### C5. Data Release Docs

- [X]  Add a dataset card.
- [X]  Add source-data provenance notes.
- [X]  Add annotation process notes.
- [X]  Add privacy, redaction, and data-usage constraints.
- [X]  Add instructions for citing the dataset.

Acceptance criteria:

- Every public data file has a documented schema and a validation path.

---

## Workstream D: Metrics, Formulas, and Evaluation Docs

### D1. Task 1 and Task 2 Metrics

- [X]  Write the exact matching and scoring formula used in `compare_memory_records.py`.
- [X]  Explain `label_similarity`, `value_similarity`, `type_match`, and `confidence_score`.
- [X]  Explain `greedy` vs `hungarian` matching.
- [X]  Explain `min_pair_score`.
- [X]  Explain `algo_score`, optional `llm_score`, and final blended score.

### D2. Task 3 Metrics

- [X]  Document the main judge-based scoring path.
- [X]  Document the fallback token-overlap path.
- [X]  Document what the evaluator sees vs what the judge sees.
- [X]  Define the official reported metric for Task 3 leaderboard reporting.
- [X]  Clarify whether `selected_memory_id`, `used_memory`, and `score` play different roles.

### D3. Task 4 Metrics

- [X]  Reconcile current code (`ability2_general`, `ability2_interaction`, `ability6`) with the exported `Alps_data_v1/task4/ability1..ability5` structure.
- [X]  Audit each ability's current judge output schema.
- [X]  Decide on a unified per-ability metric schema.
- [X]  Decide on the official overall Task 4 aggregation rule.
- [X]  Document ability-level scoring dimensions and their final reported scores.
- [X]  Align code and docs so Task 4 does not depend on implicit judge behavior.

### D4. Metric Transparency

- [X]  Add `docs/metrics.md` as the canonical metric doc.
- [X]  State clearly which metrics are heuristic, which are judge-driven, and which are exact.

Acceptance criteria:

- A user can understand every reported score from documented logic.

---

## Workstream E: CLI, Scripts, and Output Conventions

### E1. Public CLI Surface

- [X]  Define a stable command pattern for evaluation, such as `python -m ... evaluate --task ...`.
- [X]  Remove reliance on hidden internal defaults in public examples.
- [X]  Standardize argument names across tasks.

### E2. Output Format

- [X]  Define a canonical output directory structure.
- [X]  Standardize file names such as `reports.jsonl`, `scores.jsonl`, and `summary.json`.
- [X]  Add a machine-readable summary artifact for each run.
- [X]  Document which artifacts are intermediate vs final.

### E3. Example Runs

- [X]  Add one example command per task that works on shipped example data.
- [X]  Document the official baseline references for the current public release.
- [X]  Add one example command for "custom model endpoint" usage.

Acceptance criteria:

- Script behavior is predictable across all tasks.

---

## Workstream F: Baselines and Leaderboard Protocol

### F1. Official Baselines

- [X]  Decide which baselines are officially supported.
- [X]  Document the BM25 baseline as a paper-reference baseline for the current public release.
- [X]  Document the embedding baseline as a paper-reference baseline for the current public release.

### F2. Reference Results

- [ ]  Provide small reference outputs on example data.
- [ ]  Provide expected summary metrics for smoke runs.
- [ ]  Record model name, config, timestamp, and benchmark version in outputs.

### F3. Leaderboard Protocol

- [X]  Define the official metrics to report per task.
- [X]  Define how to aggregate task-level results into tables.
- [X]  Define submission format for external contributors.
- [X]  Add a `docs/leaderboard.md` or equivalent.

Acceptance criteria:

- A third party can understand the reference baselines, submission path, and what numbers to report for the current public release.

---

## Workstream G: Testing, CI, and Engineering Quality

### G1. Automated Tests

- [ ]  Add unit tests for shared scoring logic.
- [ ]  Add unit tests for JSON parsing utilities.
- [X]  Add dataset validation tests for example files.
- [ ]  Add smoke tests for Task 1 through Task 4 example flows.

### G2. CI

- [X]  Add GitHub Actions or equivalent CI.
- [ ]  Run lint, tests, and example smoke checks in CI.

### G3. Quality Gates

- [ ]  Decide on formatting and linting tools.

Acceptance criteria:

- Public changes can be validated automatically before merge.

---

## Workstream H: Open-Source Readiness and Release Governance

### H1. Legal and Usage

- [X]  Add a code license.
- [X]  Add a data license or data usage notice.
- [X]  Clarify whether all data can be redistributed via GitHub.
- [X]  Add privacy and redaction notes for real-dialogue-derived data.

### H2. Community Files

- [X]  Add `CONTRIBUTING.md`.
- [X]  Add `CHANGELOG.md`.

### H3. Release Policy

- [ ]  Define benchmark versioning rules.
- [ ]  Define dataset versioning rules.
- [ ]  Define how breaking metric or schema changes will be communicated.
- [ ]  Define release checklist for public tags.

Acceptance criteria:

- The repo can be maintained publicly without unclear ownership or ambiguous licensing.

---

## Done Definition for This Refactor

Do not call this refactor complete until all of the following are true:

- [X]  The root README is public-ready.
- [ ]  Every public command in the docs is runnable.
- [X]  Example data is validated and documented.
- [X]  Full-data access path is documented.
- [ ]  Every reported metric has a canonical formula doc.
- [X]  Task 4 has a clearly documented official aggregation rule.
- [X]  Tests and CI are in place.
- [X]  Code and data licensing are explicit.

---

## Progress Log

Use this section to record meaningful milestones, not every tiny edit.

- [X]  Initial roadmap created.
- [X]  Canonical raw data source identified as `Alps_data_v1/`.
- [X]  Initial dataset build and validation scaffolding added for `Alps_data_v1`.
- [X]  Repository structure guide added under `docs/`.
- [X]  Full normalized benchmark files generated and validated under `benchmark_data/normalized/`.
- [X]  Public root README, quickstart, configuration, troubleshooting, and FAQ docs added.
- [X]  Public example smoke runner added for Task 1 through Task 4 example files.
- [X]  Public evaluation matrix and metric-surface docs added under `docs/`.
- [X]  Public tests and GitHub Actions scaffold added for the current benchmark surface.
- [X]  `REPOSITORY_REFACTOR_ROADMAP.md` now records how completed work was finished and why unfinished work remains open.
- [X]  Roadmap checklist refreshed against the current public scaffold, released split layout, and runnable public commands.
- [X]  README and evaluation docs were updated to state that `benchmark_data/` is the full public release, Task 3 official reporting follows the paper, and submission or leaderboard details defer to the paper plus maintainer contact.
- [X]  Repository governance files and leaderboard policy docs were added for the current public release.
