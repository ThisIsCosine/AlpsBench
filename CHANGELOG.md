# Changelog

All notable public benchmark-surface changes should be recorded here.

## v4

- rebuilt all public splits from `Alps_data_final_v4`
- limited Task 3 to the d100 source track actually present in the v4 bundle
- normalized Task 3 candidate IDs to stable row-local IDs and deterministically
  filled duplicate source candidates from the v4 Task 1 memory pool
- retained Ability 5 as an explicit empty reserved track
- added safe zip import, source checksums, stronger validation, and provenance
  documentation

## Unreleased

- restored the maintainer's original Task 1 personalized-memory extraction prompt
- added a configurable Task 1 HTTP adapter and mock-API integration tests
- documented the distinction between extraction, public proxy scoring, and the
  unavailable legacy curator/judge implementation
- fixed test collection when invoking `pytest` directly and in GitHub Actions

- clarified that `benchmark_data/` is the full public `v1` benchmark release
- documented data provenance, data-usage boundary, and citation path
- documented Task 3 paper-reported metrics versus the released local proxy
- documented leaderboard and submission policy for the current public release
- added repository governance files: `LICENSE`, `DATA_USAGE.md`, and
  `CONTRIBUTING.md`

## v1

- initial public benchmark release built around `benchmark_data/`, `scripts/`,
  `src/benchmark/`, and `docs/`
