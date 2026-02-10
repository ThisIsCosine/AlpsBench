"""
Task 3 Generator: Memory Retrieval.
Creates probes that require retrieving specific memory to answer a query.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Callable

try:
    from json_repair import repair_json
except ImportError:  # optional dependency
    repair_json = None

from ..shared import call_model_api


TASK3_GENERATOR_PROMPT = """SYSTEM:
You are Task 3 Generator for memory retrieval.
You will receive:
- dialogue (user+assistant turns)
- selected_memory (the memory that MUST be used)

[Goal]
Write ONE highly implicit query that requires multi-hop reasoning or logical deduction to answer. The answer must be retrievable ONLY from the 'selected_memory'.
It's necessary to mimic the chat style and questioning methods of real users, and initiate inquiries from the user's perspective.

[Deduction/Multi-hop Requirement]
Instead of asking for the fact directly, create a scenario or a secondary question that requires retrieving the fact first and then applying logic.
Example:
- Memory: "User is allergic to peanuts."
- Direct (Bad): "What is the user allergic to?"
- Multi-hop (Good): "If someone were to gift me a box of traditional Snickers bars, why might I have to decline eating them based on my health background?" (Requires retrieving 'peanut allergy' and knowing Snickers contain peanuts).

[CRITICAL REQUIREMENT]
1. PERSONALIZED MEMORY ONLY: The 'selected_memory' must be a personalized/individual memory about the user (e.g., job, family, preferences, real-life experiences).
2. HIGHLY INDIRECT: The query must NOT reveal, paraphrase, or hint at unique details from the selected_memory. 
3. EXCLUSIVITY: A model without access to this specific memory should find it impossible to guess the correct answer.

[Hard constraints]
- The query must NOT include names, dates, locations, titles, or unique phrases from selected_memory.
- The query should ask for a consequence, a reason, or a choice that depends on the retrieved memory.
- Output must be a single JSON object, no markdown.

[Output format]
{
  "task": "memory_retrieval",
  "query": "string",
  "selected_memory_id": "string",
  "selected_memory": {"memory_id": "...", "label": "...", "value": "...", "evidence_text": "..."},
  "metadata": {"session_id": "string", "dialog_id": "string", "language": "Chinese|English"}
}
Return ONLY the JSON object.
"""


class Task3Generator:
    def __init__(self, system_prompt: str = TASK3_GENERATOR_PROMPT) -> None:
        self.system_prompt = system_prompt

    def build_user_prompt(self, seed: Dict[str, Any], api_config: Dict[str, Any] | None = None) -> str:
        selected = seed.get("selected_memory") or None
        if not selected and seed.get("selected_memory_id"):
            for memory in seed.get("memories") or []:
                if memory.get("memory_id") == seed.get("selected_memory_id"):
                    selected = memory
                    break
        payload = {
            "dialogue": seed.get("dialogue") or [],
            "selected_memory_id": seed.get("selected_memory_id"),
            "selected_memory": selected,
            "metadata": seed.get("metadata") or {},
        }
        return json.dumps(payload, ensure_ascii=True)

    def generate_probe(
        self,
        seed: Dict[str, Any],
        call_model,
        model_id: str | None = None,
        api_config: Dict[str, Any] | None = None,
        log_fn: Callable[[str, Dict[str, Any]], None] | None = None,
    ) -> Dict[str, Any]:
        user_prompt = self.build_user_prompt(seed, api_config=api_config)
        system_prompt = self.system_prompt
        feedback = seed.get("generator_feedback")
        if isinstance(feedback, str) and feedback.strip():
            system_prompt = f"{system_prompt}\n\nFollow these rewrite instructions exactly:\n- {feedback.strip()}"
        if log_fn:
            log_fn("task3_generator_start", {"seed_keys": list(seed.keys())})
            log_fn("task3_generator_input", {"system_prompt": system_prompt, "user_prompt": user_prompt})
        if call_model is None:
            if not model_id:
                raise ValueError("generator_model_id_required")
            raw = call_model_api(model_id, "M_generate", system_prompt, user_prompt, api_config)
        else:
            raw = call_model(system_prompt, user_prompt)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("generator_empty_output")
        if log_fn:
            log_fn("task3_generator_raw_output", {"raw_output": raw})
        try:
            probe = json.loads(raw)
        except json.JSONDecodeError:
            if repair_json is None:
                raise
            repaired = repair_json(raw)
            probe = json.loads(repaired)
        if not isinstance(probe, dict):
            raise ValueError("generator_invalid_json")
        if not probe.get("query"):
            print(probe)
            raise ValueError("generator_missing_query")
        # Force selected_memory from seed into the probe (evaluator does not see it).
        seed_selected = seed.get("selected_memory")
        if not seed_selected and seed.get("selected_memory_id"):
            for memory in seed.get("memories") or []:
                if memory.get("memory_id") == seed.get("selected_memory_id"):
                    seed_selected = memory
                    break
        if seed_selected:
            probe["selected_memory"] = seed_selected
            if not probe.get("selected_memory_id"):
                probe["selected_memory_id"] = seed_selected.get("memory_id")
        # Ensure probe metadata carries stable IDs from seed to avoid "string/null" placeholders.
        seed_meta = seed.get("metadata") or {}
        metadata = probe.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        def _valid_id(value: Any) -> bool:
            if value is None:
                return False
            text = str(value).strip()
            if not text:
                return False
            return text.lower() not in {"string", "null", "none"}

        if _valid_id(seed_meta.get("session_id")) and not _valid_id(metadata.get("session_id")):
            metadata["session_id"] = seed_meta.get("session_id")
        if _valid_id(seed_meta.get("dialog_id")) and not _valid_id(metadata.get("dialog_id")):
            metadata["dialog_id"] = seed_meta.get("dialog_id")
        if metadata:
            probe["metadata"] = metadata
        if log_fn:
            log_fn(
                "task3_generator_output_summary",
                {
                    "probe_id": probe.get("metadata", {}).get("dialog_id", ""),
                    "selected_memory_id": probe.get("selected_memory_id"),
                    "query": probe.get("query"),
                    "expected_answer": probe.get("expected_answer"),
                    "memories_count": len(probe.get("memories") or []),
                    "candidate_memories_count": len(probe.get("candidate_memories") or []),
                },
            )
            log_fn("task3_generator_done", {"probe_id": probe.get("metadata", {}).get("dialog_id", "")})
        return probe
