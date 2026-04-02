# Evaluation Matrix

This document explains how AlpsBench is intended to evaluate model behavior at the
task level. It is meant to answer four user-facing questions:

1. What input does the evaluated model receive?
2. What output should the model return?
3. How is each prediction scored?
4. Which dimensions matter for each task?

This file summarizes two sources together:

- the user-visible schema in `benchmark_data/examples/`

The lightweight scorer in `src/benchmark/metrics.py` is a public local-scoring
proxy. The matrix below records the fuller benchmark-side intent of the
evaluation protocol.

If you are implementing your own AlpsBench adapter, use this file as the
task-level input/output reference together with `adapter_example/`.

## Official Scoring Note

For Task 1 and Task 2, the official scoring method is based on memory matching
and the precision-recall family of metrics.

- `precision`
- `recall`
- `f1`

These metrics are part of the official scoring surface, not just debugging
statistics. In addition to combined ranking views, we also maintain separate
scoreboard views for pure `precision` and pure `recall`.

For the `LLM-as-Judge` component in Task 1 and Task 2, we do not use an
unconstrained natural-language judgment. We use our own independently designed
structured evaluation schema.

## At A Glance


| Task             | Model-visible input          | Expected model output                   | Main scoring idea                                                                   |
| ---------------- | ---------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------- |
| Task 1           | dialogue                     | extracted memory list                   | memory-to-memory matching with similarity scoring, coverage, and optional LLM judge |
| Task 2           | existing memories + dialogue | updated memory list                     | same matching core as Task 1, but evaluated as update behavior                      |
| Task 3           | query + candidate memories   | answer + rationale + selected memory id | judge whether the selected memory was actually used                                 |
| Task 4 Ability 1 | selected memory + query      | answer + used memory fact               | judge persona grounding and personalization quality                                 |
| Task 4 Ability 2 | dialogue history + query     | answer + used memory fact               | judge preference following, including content and interaction style                 |
| Task 4 Ability 3 | dialogue history + query     | answer + used memory fact               | judge role-play or virtual-context consistency                                      |
| Task 4 Ability 4 | dialogue history + query     | answer + used memory fact               | judge constraint or boundary adherence                                              |
| Task 4 Ability 5 | dialogue history + query     | answer + used memory fact               | judge emotional grounding and empathy quality                                       |

## Shared Memory Schema

Across Task 1 and Task 2, a memory item usually uses a schema like:

```json
{
  "memory_id": "m1",
  "type": "direct",
  "label": "Preferences/Food",
  "value": "Prefers vegetarian meals",
  "reasoning": "Optional explanation in gold data",
  "evidence": {
    "session_id": "sess_x",
    "utterance_index": 3,
    "text": "I avoid meat."
  },
  "confidence": 0.9,
  "time_scope": "long_term",
  "emotion": null,
  "preference_attitude": "like",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

The benchmark data may carry many fields, but the legacy algorithmic matcher is
driven mainly by:

- `label`
- `value`
- `type`
- `confidence`

The label space is hierarchical. Annotation-side taxonomy files show categories
such as:

- `Personal_Background/...`
- `States_Experiences/...`
- `Possessions/...`
- `Preferences/...`
- `Thoughts/...`
- `Plans/...`

## Task 1: Memory Extraction

### What the model receives

Legacy evaluator input is the dialogue only. The extraction instruction is
placed in the system prompt, while the user prompt contains:

```json
{
  "dialogue": [
    {"role": "user", "text": "..."},
    {"role": "assistant", "text": "..."}
  ]
}
```

In user-facing `model_input.jsonl`, the row is wrapped in benchmark metadata,
but the model-visible core is still the dialogue.

### What the model should return

The evaluator accepts either a JSON list or a JSON object:

```json
[
  {
    "memory_id": "p1",
    "type": "direct",
    "label": "Preferences/Food",
    "value": "Prefers vegetarian meals",
    "confidence": 0.9,
    "evidence_text": "I avoid meat."
  }
]
```

or

```json
{
  "memory_items": [
    {
      "memory_id": "p1",
      "type": "direct",
      "label": "Preferences/Food",
      "value": "Prefers vegetarian meals",
      "confidence": 0.9
    }
  ]
}
```

### How scoring works

Task 1 scoring in the legacy implementation is pairwise memory matching plus
coverage and optional LLM judging.

#### Step 1: normalize text

All text fields are lowercased, stripped, punctuation-normalized, and
whitespace-normalized before comparison.

#### Step 2: score every gold-prediction pair

For one gold item `a` and one predicted item `b`:

- `label_similarity`
  - labels are treated as paths like `Preferences/Food`
  - score = `2 * LCP / (len(path_a) + len(path_b))`
  - `LCP` is the longest common prefix length of the path segments
- `value_similarity`
  - blended lexical and character similarity
  - default formula:
  - `0.6 * token_jaccard + 0.4 * sequence_match_ratio`
- `type_match`
  - `1.0` if `type` matches exactly, else `0.0`
- `confidence_score`
  - no penalty when `|conf_a - conf_b| <= 0.2`
  - after that, penalty grows linearly until the score reaches `0`

Default pair score:

```text
item_score
= 0.40 * label_similarity
+ 0.40 * value_similarity
+ 0.10 * type_match
+ 0.10 * confidence_score
```

#### Step 3: find a 1-to-1 matching

Two matchers are implemented:

- `greedy`
- `hungarian`

Only pairs with `item_score >= min_pair_score` are accepted. The default
threshold is `0.30`.

Accepted pairs are counted as matched items. Unmatched gold items are
`missing`. Unmatched predicted items are `extra`.

#### Step 4: compute coverage

If `TP = number of matched pairs`:

- `precision = TP / #predicted_items`
- `recall = TP / #gold_items`
- `f1 = 2PR / (P + R)`

#### Step 5: compute quality

Quality is the mean pair score over matched pairs only:

- `avg_pair_score`
- `avg_label`
- `avg_value`
- `type_match_rate`
- `avg_confidence_score`

#### Step 6: compute algorithmic score

Default weights are:

- `algo_quality_weight = 0.60`
- `algo_coverage_weight = 0.40`

So:

```text
algo_score_100 = 100 * (0.60 * avg_pair_score + 0.40 * f1)
```

#### Step 7: optional LLM score

An optional judge reviews:

- matched pairs
- missing gold items
- extra predicted items

Its penalties include:

- missing item that should have been extracted
- hallucinated extra item
- pair-level issues such as `VALUE_DRIFT`, `HALLUCINATION`, `LABEL_WRONG`,
  `TYPE_WRONG`, `OVERGENERAL`, `UNDERGENERAL`

This `LLM-as-Judge` path is not an unstructured comment-only review. It uses a
benchmark-specific schema designed for this task family. The judge is expected
to return structured fields such as:

- `pair_reviews`
- `missing_reviews`
- `extra_reviews`

The schema is explicitly designed to separate:

- matched-but-imperfect pairs
- missed gold memories
- hallucinated extra memories

The final blended score is:

```text
final_score_100 = (1 - llm_weight) * algo_score_100 + llm_weight * llm_score_100
```

### What dimensions matter

Task 1 is really checking:

- memory coverage: did the model find the right memories?
- label fidelity: did it place the memory in the right semantic bucket?
- value fidelity: did it preserve the user meaning?
- type correctness: direct vs indirect extraction
- confidence calibration: was uncertainty reasonable?
- over-extraction control: did it hallucinate extra memories?

Annotation-side review requirements also emphasize:

- concrete dialogue evidence
- faithful wording
- avoiding unsupported over-inference

## Task 2: Memory Update

### What the model receives

Legacy evaluator input combines old memories with old and new dialogue:

```json
{
  "existing_memories": [
    {"memory_id": "m1", "label": "Preferences/Food", "value": "Prefers tea"}
  ],
  "dialogue": [
    {"role": "user", "text": "... old turn ..."},
    {"role": "assistant", "text": "..."},
    {"role": "user", "text": "... new turn ..."}
  ]
}
```

In user-facing example data, the benchmark row usually distinguishes:

- `old_dialogue`
- `new_dialogue`
- `memory`

### What the model should return

The model returns an updated memory list in the same general schema as Task 1:

```json
[
  {
    "memory_id": "m1_updated",
    "type": "direct",
    "label": "Preferences/Food",
    "value": "Now prefers coffee over tea",
    "confidence": 0.85
  }
]
```

The evaluator accepts either a list or a wrapper object with `memory_items`.

### How scoring works

Task 2 reuses the same core matcher and scoring stack as Task 1:

- pairwise similarity scoring
- greedy or Hungarian matching
- precision, recall, F1
- quality score over matched pairs
- `algo_score_100`
- optional `llm_score_100`

So the same formulas apply.

What changes is the semantic meaning of correctness. In Task 2, the benchmark is
not only asking whether an item matches gold textually, but whether the system
performed the right update action:

- keep unchanged memories unchanged
- add new stable memories when warranted
- modify outdated memories when contradicted
- avoid changing unrelated memories

The optional `LLM-as-Judge` layer for Task 2 follows the same idea as Task 1:
it uses an independently designed structured schema rather than an unconstrained
natural-language opinion. In practice, it separates pair-level errors, missing
updates, and extra or hallucinated updates into different review fields.

### What dimensions matter

From the implementation and annotation requirements, Task 2 focuses on:

- update relevance: was there a real update signal?
- state consistency: does the new memory reflect the new dialogue?
- retention correctness: were unaffected memories preserved?
- edit precision: did the model change only what should change?
- add / modify / overwrite behavior: was the update strength appropriate?

## Task 3: Memory Retrieval

### What the model receives

The evaluated model sees a retrieval problem:

```json
{
  "query": "If I insisted I had firsthand knowledge from outside the normal timeline, what detail about me would make that stance consistent?",
  "candidate_memories": [
    {"memory_id": "m1", "value": "...", "label": "..."},
    {"memory_id": "m2", "value": "...", "label": "..."},
    {"memory_id": "m3", "value": "...", "label": "..."}
  ]
}
```

Important detail:

- the evaluated model is usually not given the dialogue
- dialogue may still be stored in benchmark artifacts for audit
- the real task is to retrieve from the candidate memory set

### What the model should return

The expected response format is:

```json
{
  "answer": "string",
  "reason": "string",
  "selected_memory_id": "string"
}
```

The `reason` field is important because the judge uses it to decide whether the
selected memory was actually used.

### How scoring works

Task 3 has two layers: probe quality control and answer scoring.

#### Probe quality control

Before evaluation, the benchmark checks whether the generated query is valid:

- no leakage of the hidden answer
- real dependence on the selected memory
- not a trivial or guessable question
- not a virtual-world memory when the task expects real personalized retrieval

This is part of benchmark construction quality, not the final model score.

#### Main scoring path: judge-based usage check

After the model answers, the judge receives:

```json
{
  "query": "...",
  "model_response": {
    "answer": "...",
    "reason": "...",
    "selected_memory_id": "m3"
  },
  "selected_memory": {
    "memory_id": "m3",
    "value": "..."
  }
}
```

The judge returns:

```json
{
  "used_memory": true,
  "score": 0.83,
  "reason": "string"
}
```

This means the main per-example Task 3 signal is:

- `used_memory`: binary decision
- `score`: continuous usage quality in `[0, 1]`

#### Official reported metric

The paper frames Task 3 as a retrieval task and reports retrieval-oriented
metrics:

- `precision`
- `recall`

In the released public interface, each example asks the evaluated model to
return one `selected_memory_id` from a candidate set. Because of that public
prediction contract, the runnable local scorer in `src/benchmark/metrics.py`
uses exact selected-memory correctness as a lightweight public proxy.

So the practical reading is:

- paper-style official reporting follows the retrieval framing in the paper
- public local scoring uses `selected_memory_id` match as the released proxy
- judge outputs such as `used_memory` and `score` remain benchmark-side signals
  for checking whether the selected memory was genuinely used

#### Fallback scoring path

If judge output is unavailable or unparsable, the legacy code falls back to a
simple overlap heuristic between the model rationale and the selected memory
text:

- normalize selected memory text and rationale
- compute token overlap score
- mark `used_memory = true` when score `>= 0.2`

### What dimensions matter

Task 3 is checking:

- retrieval correctness: did the model point to the right memory?
- memory dependence: was the answer actually grounded in that memory?
- attribution honesty: does the chosen memory support the answer?
- rationale consistency: does the explanation match the retrieval behavior?

Annotation-side task requirements also emphasize:

- whether the selected memory is genuinely necessary
- whether the query is still answerable without personalization
- whether the chosen memory is the best supporting memory

## Task 4: Memory-Grounded Response Generation

Task 4 measures whether the model can use long-term memory to produce the right
kind of personalized response.

### Common output format

Across Ability 1 to Ability 5, the evaluated model is asked to return:

```json
{
  "answer": "string",
  "used_memory_fact": "string"
}
```

The exact input differs by ability.

### Public note on benchmark-side judging

The released public scorer in `src/benchmark/metrics.py` is only a lightweight
grounding proxy for Task 4. The benchmark-side evaluation logic is richer and
uses ability-specific structured judge outputs.

For open documentation purposes, we describe the Task 4 judge surface at the
dimension level rather than exposing full prompt text, exact decision rules, or
prompt-internal guardrails. This is intentional: the goal is to make the
benchmark understandable without turning the prompt itself into the target.

The public release and the current `src/benchmark/` interface use
`task4_ability1` through `task4_ability5` as the canonical Task 4 tracks. Older
benchmark-side implementation references may still show finer-grained internal
judge modules, but those are background sources rather than separate public
tracks.

Across abilities, the benchmark-side judge is intended to answer two broad
questions:

- did the model actually use the relevant memory in the response?
- was that memory used in the right personalized way for this ability?

In the human-alignment workflow, the object of review is the judge output
itself. Reviewers examine the model input, model response, claimed memory use,
and structured judge fields, then record whether each judge conclusion is
supported, unsupported, or needs revision.

### Task 4 judge dimensions at a glance


| Ability   | High-level judge dimensions                                                                                              | Public interpretation                                                                                                         |
| --------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Ability 1 | memory alignment, naturalness, claim honesty, personalization, overall score                                             | did the answer use the selected persona fact correctly and in a natural personalized way                                      |
| Ability 2 | preference respect, interaction-style following, factual grounding of claimed memory use, response quality / naturalness | did the answer follow the user's remembered preferences and interaction habits rather than only sounding good in the abstract |
| Ability 3 | immersion, in-character grounding, world consistency, overall score                                                      | did the answer stay inside the established role-play or virtual context without breaking character                            |
| Ability 4 | constraint adherence, helpfulness under constraints, overall score                                                       | did the answer respect remembered boundaries while still being useful                                                         |
| Ability 5 | emotional grounding, fact match, empathy quality                                                                         | did the answer remember the relevant emotional context and respond with appropriate empathy                                   |

Two implementation notes matter for interpretation:

- not every internal judge schema maps one-to-one to a public score field; some criteria are merged into broader dimensions such as an overall score or an immersion-style score
- some abilities emit an explicit overall score while others emit only dimension-level scores

So the benchmark-side Task 4 judge should be understood as a structured,
ability-specific evaluator, not one universal scalar judge reused unchanged
across all personalized response settings.

### Official Task 4 reporting policy

Task 4 is officially treated as a family of independent benchmark tracks rather
than one aggregated leaderboard task.

The released policy is:

- each ability is scored independently
- each ability is ranked independently
- each ability is compared independently
- AlpsBench does not define one cross-ability `Task4_overall` public score

This keeps the benchmark honest about what the judge is measuring. The abilities
cover meaningfully different forms of memory-grounded personalization, so a
single merged score would hide important differences in model behavior.

In practice, the public reporting unit for Task 4 is therefore:

- `task4_ability1`
- `task4_ability2`
- `task4_ability3`
- `task4_ability4`
- `task4_ability5`

and not one blended Task 4 total.

---

## Task 4 Ability 1: Persona Grounding

### What the model receives

Ability 1 directly exposes the selected memory:

```json
{
  "selected_memory": {
    "memory_id": "m2",
    "value": "Pursuing a Master's in Human-Centered Design"
  },
  "latest_query": "..."
}
```

### How scoring works

The judge evaluates:

- `alignment`
  - whether the answer uses the selected memory correctly
- `naturalness`
  - whether the answer avoids AI meta-talk and stays in character
- `honesty`
  - whether `used_memory_fact` is actually supported by the answer
- `personalization`
  - whether the response feels specifically tailored
- `overall_weighted_score`

### What dimensions matter

Ability 1 cares about:

- persona fit
- correct memory usage
- natural conversational style
- non-generic personalization

---

## Task 4 Ability 2: Preference Following

### What the model receives

Ability 2 uses dialogue history rather than directly exposing the gold memory:

```json
{
  "dialogue_history": [
    {"role": "user", "text": "..."},
    {"role": "assistant", "text": "..."}
  ],
  "latest_query": "..."
}
```

### How scoring works

Legacy Ability 2 evaluation covers both:

- content preference following
- interaction preference following, such as tone, length, or format

Judge dimensions used in code include:

- `respect_score`
  - whether the answer respects the hidden preference
- `preference_following`
  - whether the answer follows the requested interaction style
- `fact_match`
  - whether `used_memory_fact` matches the gold preference
- `answer_quality`
  - whether the answer is otherwise good
- `naturalness`
  - whether it avoids robotic meta-talk
- `overall_weighted_score`
  - used in the content-preference judge path

### What dimensions matter

Ability 2 mainly measures:

- preference respect
- style following
- factual grounding of the claimed memory use
- response usefulness without breaking personalization

---

## Task 4 Ability 3: Role-Play Or Virtual-Context Consistency

### What the model receives

The model sees dialogue history and the latest query:

```json
{
  "dialogue_history": [...],
  "latest_query": "..."
}
```

### How scoring works

The judge checks:

- `immersion_score`
  - whether the answer maintains the established world or role-play logic
- `consistency_score`
  - whether the answer stays true to the established facts
- `overall_weighted_score`

The prompt also explicitly asks the judge to consider:

- in-character grounding
- immersion guardrails
- creative consistency

Strong penalty target:

- meta-talk and out-of-world narration should strongly hurt immersion

### What dimensions matter

Ability 3 is about:

- staying inside the user's virtual context
- preserving world consistency
- being creative without breaking established facts

---

## Task 4 Ability 4: Constraint Or Boundary Adherence

### What the model receives

Again the model sees dialogue history and a fresh query:

```json
{
  "dialogue_history": [...],
  "latest_query": "..."
}
```

### How scoring works

The judge returns:

- `adherence_score`
  - whether the answer respected the remembered boundary
- `helpfulness_score`
  - whether it stayed useful while respecting that boundary
- `overall_weighted_score`

### What dimensions matter

Ability 4 checks:

- remembered user boundaries
- safe redirection instead of blunt refusal
- natural refusal style instead of robotic policy talk

---

## Task 4 Ability 5: Emotional Grounding And Empathy

### What the model receives

The model sees dialogue history and a high-EQ query:

```json
{
  "dialogue_history": [...],
  "latest_query": "..."
}
```

### How scoring works

The judge returns:

- `emotional_grounding`
  - whether the answer remembers and respects the emotional meaning of the
    selected memory
- `fact_match`
  - whether `used_memory_fact` matches the selected memory
- `empathy_quality`
  - whether the answer is genuinely empathetic instead of cliched or robotic

Unlike some other Task 4 abilities, this judge path returns dimension scores
directly rather than requiring a single explicit overall score field.

### What dimensions matter

Ability 5 measures:

- emotional sensitivity
- whether personalization is emotionally appropriate, not just factually correct
- whether the answer sounds supportive and human

### Human-alignment evaluation of the judge itself

For Task 4, human review is performed on the judge outputs rather than only on
the model answers. The core question is:

- does each judge field correctly reflect the quality of the model's memory use?

At a high level, the review protocol asks annotators to:

- read the model input, the model answer, and the claimed memory use
- inspect the structured judge fields for the relevant ability
- decide whether each field is supported by the evidence
- record disagreements, likely failure modes, and suggested corrections

This means the Task 4 judge-alignment analysis is field-level rather than just
example-level. In practice, the important error modes are:

- field-level score inflation or under-scoring
- reasoning text that does not support the assigned scores
- failure to distinguish memory grounding from general answer quality
- ability-specific confusions such as mixing preference following with factual grounding, or empathy with simple politeness

The public documentation intentionally stays at this level. It explains what the
judge is supposed to measure without publishing the full prompt wording or
prompt-internal scoring logic.

## Summary Of Scoring Dimensions

If you only want the shortest mental model, AlpsBench scores the following:

- Task 1: extract the right memories and express them faithfully
- Task 2: update the right memories while preserving the rest
- Task 3: retrieve the right memory and prove it was really used
- Task 4 Ability 1: use persona facts naturally
- Task 4 Ability 2: follow user preferences and interaction style
- Task 4 Ability 3: stay consistent with the user's virtual context
- Task 4 Ability 4: respect remembered constraints and boundaries
- Task 4 Ability 5: respond with grounded empathy

## Primary Implementation References

- `Alps_code_v1/AlpsBench_source_code/src/agents/compare_memory_records.py`
- `Alps_code_v1/AlpsBench_source_code/src/agents/task1_memory_extraction/evaluator.py`
- `Alps_code_v1/AlpsBench_source_code/src/agents/task2_memory_update/evaluator.py`
- `Alps_code_v1/AlpsBench_source_code/src/agents/task3_memory_retrieval/evaluator.py`
- `Alps_code_v1/AlpsBench_source_code/src/agents/task3_memory_retrieval/curator.py`
- `Alps_code_v1/AlpsBench_source_code/src/agents/task4_memory_grounded_qa/ability1/curator.py`
- `Alps_code_v1/AlpsBench_source_code/src/agents/task4_memory_grounded_qa/ability2_general/curator.py`
- `Alps_code_v1/AlpsBench_source_code/src/agents/task4_memory_grounded_qa/ability2_interaction/curator.py`
- `Alps_code_v1/AlpsBench_source_code/src/agents/task4_memory_grounded_qa/ability3/curator.py`
- `Alps_code_v1/AlpsBench_source_code/src/agents/task4_memory_grounded_qa/ability4/curator.py`
- `Alps_code_v1/AlpsBench_source_code/src/agents/task4_memory_grounded_qa/ability5/curator.py`
- `benchmark_data/examples/`
- `Alps_Human_Alignment_Annotation/DataAnnotation/src/q1-q2/config/requirements.ts`
- `Alps_Human_Alignment_Annotation/DataAnnotation/src/config/memory-taxonomy.ts`
