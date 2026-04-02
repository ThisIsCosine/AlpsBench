# Contributing

AlpsBench accepts issue reports, documentation fixes, and benchmark-surface
improvements.

## What To Open

- use an issue for bug reports, unclear docs, schema problems, or submission
  questions
- use a pull request for small scoped fixes that preserve the public benchmark
  surface
- for larger benchmark-policy changes, open an issue first so metric or schema
  changes are discussed before implementation

## Local Checks

Before sending a change, run the lightweight public checks that match the
current repository surface:

```bash
python scripts/validate_data.py
python scripts/smoke_examples.py
python scripts/evaluate.py --task task1 --split examples --oracle
```

If your change touches docs only, explain that in the PR and note which commands
you skipped.

## Scope Rules

- keep `benchmark_data/` contracts stable unless the change is intentionally a
  versioned benchmark update
- do not add hidden-path assumptions back into public scripts or docs
- do not commit private evaluation references or maintainer-only assets
- do not change public task schemas without updating the matching docs

## External Submission Path

If you are not changing the repository itself and only want your model added to
the benchmark, follow the submission instructions in `README.md` and
`docs/leaderboard.md` instead of opening a code PR.
