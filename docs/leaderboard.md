# Leaderboard And Submission

This document describes the public reporting policy for AlpsBench.

## Official Reporting Surface

- `task1`: report memory extraction metrics from the Task 1 evaluation protocol
- `task2`: report memory update metrics from the Task 2 evaluation protocol
- `task3`: follow the paper's retrieval framing and reported retrieval metrics
- `task4`: report each ability independently; there is no merged public
  `Task4_overall`

For Task 4 specifically:

- `task4_ability1`
- `task4_ability2`
- `task4_ability3`
- `task4_ability4`
- `task4_ability5`

Each ability is scored, ranked, and compared independently.

## Table Aggregation Policy

The current public release follows the paper-style reporting view:

- Task 1 and Task 2 are reported as their own task lines
- Task 3 is reported as its own retrieval task line
- Task 4 abilities are reported as separate benchmark tracks rather than one
  merged aggregate

This means there is no required single scalar across all AlpsBench tasks in the
public release documentation.

## Baselines

The paper reports two reference Task 3 retrieval baselines:

- `nltk + bm25`
- `all-MiniLM-L6-v2`

In the current public repository release, these baselines are reference
baselines from the paper rather than bundled runnable baseline implementations.

## External Submission

External contributors should:

1. run their system on `benchmark_data/test/`
2. produce a prediction file that matches the contract in
   `docs/prediction_contract.md`
3. submit the generated `predictions.jsonl` by contacting the first author
   listed in the paper or by opening a GitHub issue

After verification, accepted results can be added to the benchmark reporting
surface.
