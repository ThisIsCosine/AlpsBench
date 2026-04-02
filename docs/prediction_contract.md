# Prediction Contract

This document defines the required prediction-row format for public evaluation.

## Shared Rules

- every input row must produce exactly one prediction row
- every prediction row must include the matching `benchmark_id`
- no extra top-level keys are accepted
- `benchmark_data/examples/*/reference_output.jsonl` is public gold and sample
  data, not the submission format

## Task 1

Original generator/evaluator contract:

- output a JSON object with `memory_items`

Public prediction row:

```json
{
  "benchmark_id": "sess_x__1",
  "memory_items": []
}
```

`memory_items` should contain the extracted memory objects produced by your
model.

Recommended Task 1 memory item shape:

```json
{
  "memory_id": "m1",
  "type": "direct",
  "label": "Preferences/Food",
  "value": "Prefers spicy noodles",
  "confidence": 0.8,
  "evidence_text": "I really like spicy noodles."
}
```

Task 1 notes:

- each element of `memory_items` should be a JSON object
- the original Task 1 evaluator normalizes around:
  - `memory_id`
  - `type`
  - `label`
  - `value`
  - `confidence`
  - `evidence_text`
- for public scoring, the most important fields are `label`, `value`, and
  `type`

## Task 2

Original generator/evaluator contract:

- output a JSON object with `memory_items`

Public prediction row:

```json
{
  "benchmark_id": "sess_x",
  "memory_items": []
}
```

`memory_items` should be the final updated memory state after applying the new
dialogue.

Typical Task 2 memory item shape:

```json
{
  "memory_id": "m1_updated",
  "type": "direct",
  "label": "Preferences/Entertainment",
  "label_suggestion": null,
  "value": "Prefers realistic emotional animal stories over robot-themed STEM stories",
  "reasoning": "The new dialogue explicitly rejects robots and STEM framing.",
  "evidence": {
    "session_id": "sess_x",
    "utterance_index": 0,
    "text": "I don't want robots in this story."
  },
  "confidence": 0.74,
  "time_scope": "long_term",
  "emotion": null,
  "preference_attitude": "create",
  "updated_at": "2026-02-03T00:00:00Z"
}
```

Task 2 notes:

- each element of `memory_items` should be a JSON object
- the old Task 2 evaluator preserves extra keys, but fills defaults for:
  - `memory_id`
  - `type`
  - `label`
  - `value`
  - `confidence`
- public scoring mainly uses:
  - `label`
  - `value`
  - `type`
  - `preference_attitude`
  - `time_scope`
  - `emotion`
- `label_suggestion`, `reasoning`, `evidence`, and `updated_at` are strongly
  recommended because the released gold uses them heavily

## Task 3

Original generator/evaluator contract:

- output a JSON object with `answer`, `reason`, and `selected_memory_id`

Public prediction row:

```json
{
  "benchmark_id": "sess_x",
  "answer": "...",
  "reason": "...",
  "selected_memory_id": "m3"
}
```

Task 3 notes:

- the same prediction schema applies to every released Task 3 track:
  `task3_d100`, `task3_d300`, `task3_d500`, `task3_d700`, and `task3_d1000`
- `answer` is the model's natural-language answer to the query
- `reason` is the model's explanation of why that memory was used
- `selected_memory_id` should point to one of the candidate memories from the
  input row
- for public scoring, `selected_memory_id` is the decisive field

## Task 4 Ability 1

Original evaluator contract:

- output a JSON object with `answer` and `used_memory_fact`

Public prediction row:

```json
{
  "benchmark_id": "ability1_0",
  "answer": "...",
  "used_memory_fact": "..."
}
```

Task 4 notes:

- `answer` is the user-facing natural-language response
- `used_memory_fact` is the specific remembered fact or constraint the model
  claims it grounded on
- `used_memory_fact` should be concise and should correspond to the relevant
  gold selected memory

## Task 4 Ability 2

Ability 2 uses the same output format:

```json
{
  "benchmark_id": "ability2_0",
  "answer": "...",
  "used_memory_fact": "..."
}
```

## Task 4 Ability 3

Ability 3 uses the same output format:

```json
{
  "benchmark_id": "ability3_0",
  "answer": "...",
  "used_memory_fact": "..."
}
```

## Task 4 Ability 4

Ability 4 uses the same output format:

```json
{
  "benchmark_id": "ability4_0",
  "answer": "...",
  "used_memory_fact": "..."
}
```

## Task 4 Ability 5

Ability 5 uses the same output format:

```json
{
  "benchmark_id": "ability5_0",
  "answer": "...",
  "used_memory_fact": "..."
}
```

## Important Clarification

- `reference_output.jsonl` remains public gold for local scoring on
  `examples`, `dev`, and `validation`
- user submissions must follow the prediction-row schemas above
- do not submit the full public gold row shape as your prediction format
- if you are implementing an adapter, you should treat the examples above as
  the contract you need to emit, not just illustrative placeholders
