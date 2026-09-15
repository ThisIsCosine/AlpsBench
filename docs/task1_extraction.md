# Running Task 1: personalized memory extraction

Task 1 now includes the maintainer-supplied extraction prompt and an HTTP model
adapter. The adapter uses only Python's standard library. It works with endpoints
that accept chat-completions requests and return `choices[0].message.content`.

## Original prompt

[`prompts/task1_extract.txt`](../prompts/task1_extract.txt) contains the exact
runtime text of `DEFAULT_EXTRACT_PROMPT` in the maintainer's original extraction
script. The text sent to the model has SHA-256:

```text
a9cecb7416c11569c1efc96cd98f44e4eb3ffd259a080bbecd8df2d3d94ec090
```

The prompt specifies user-only evidence, long-term personalization, role-play
cautions, repeated curiosity, the label taxonomy, confidence levels, and the
`memory_items` JSON format. It is sent as the system message. The user message
contains only `{"dialogue": [{"role": "user|assistant", "text": "..."}]}`.
The adapter sends the full supplied dialogue without truncation. Benchmark IDs,
metadata, references, and annotation records are not sent to the API.

The original script calls unavailable `Task1Evaluator`, `Task1Curator`, and
`call_model_api` modules. This release restores its prompt and provides a new
public adapter; it does not claim to reproduce any additional instructions,
default sampling parameters, truncation, or judge behavior in those modules.

## Setup and a small real-model run

Clone using Git LFS to obtain actual JSONL data:

```bash
git lfs install
git clone https://github.com/ThisIsCosine/AlpsBench.git
cd AlpsBench
git lfs pull
```

Configure the endpoint, model, and API key in your environment (replace the
placeholders below with your provider's values). `ALPS_API_URL` must be the full
completion endpoint, not just the base URL. A local server on loopback may use
HTTP and omit `ALPS_API_KEY`; remote endpoints require HTTPS and a key.

```bash
export ALPS_API_URL='https://YOUR-HOST/v1/chat/completions'
export ALPS_MODEL='YOUR-MODEL-ID'
export ALPS_API_KEY='YOUR-API-KEY'

python scripts/evaluate.py --task task1 --split examples --limit 2 \
  --predict-program python --predict-arg adapter_example/task1_api_adapter.py \
  --output-dir runs/task1-api-examples
```

PowerShell environment variables use `$env:ALPS_API_URL`, `$env:ALPS_MODEL`, and
`$env:ALPS_API_KEY`; run the evaluation command on one line.

This invokes the model once per input row and then scores the returned memories
with the existing public local scorer. API calls use your provider's quota.
Outputs include `predictions.jsonl`, `scores.jsonl`, and `summary.json`.
The minimal adapter in `adapter_example/` remains a stub; use `task1_api_adapter.py`
for real extraction.

For a larger run, change `--split examples --limit 2` to `--split dev` or
`--split validation`. Use `--split test` to produce predictions for submission;
test has no public gold and is not scored locally.

To inspect the request without calling the API, pass one input JSON row to the
adapter directly:

```bash
head -n 1 benchmark_data/examples/task1/model_input.jsonl | \
  python adapter_example/task1_api_adapter.py --dry-run
```

The dry-run output is a request body, not a prediction; do not use this flag
through `evaluate.py`. It contains dialogue text, so treat saved dry runs as data.

## Parameters and errors

The adapter reads its API key only from `ALPS_API_KEY`. Optional flags include
`--timeout` (seconds, default 120), `--temperature`, and `--max-tokens`.
Sampling parameters are omitted by default because the original API helper was
not supplied and providers support different parameters. To pass these flags
through the evaluator, for example:

```bash
python scripts/evaluate.py --task task1 --split examples --limit 2 \
  --predict-program python --predict-arg adapter_example/task1_api_adapter.py \
  --predict-arg=--timeout --predict-arg 180 \
  --output-dir runs/task1-api-examples-180s
```

Malformed responses, HTTP errors, and truncated completions fail the run instead
of being converted to empty predictions. There are no automatic retries or
resume support. Use a new output directory for each run; the evaluator writes
its prediction file after completing the input batch. For long evaluations,
start with a small limit and ensure the provider supports the full context size.

## Scoring scope

The original script invokes `curator.score_report(..., use_llm=True,
matcher="greedy")`. The curator and judge implementation are not in the supplied
script. The current public scorer is unchanged: it uses an exact multiset match
over `label`, `value`, `type`, `preference_attitude`, `time_scope`, and `emotion`.
The original prompt does not request the last three fields. Missing fields are
therefore compared as empty values and may differ from the annotated gold even
when the memory's meaning is correct. This adapter does not invent or copy those
fields from gold to improve the score. Public scores are debugging proxies and
are not the paper's original semantic matching / judge scores.

Local mock-API tests verify prompt fidelity, request isolation, Unicode output,
failure handling, and integration with dev scoring and test prediction export.
They do not establish the extraction quality of any real model.
