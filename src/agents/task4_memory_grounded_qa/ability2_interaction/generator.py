"""
Task 4 Ability 2 Interaction Generator.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

QUERY_GEN_PROMPT = """SYSTEM:
You are an expert at creating benchmarks for LLM interaction preference grounding.
Your task is to generate a natural-sounding user query based on a provided dialogue history and a specific "selected_memory" about how the user wants to interact with the assistant (e.g., brevity, tone, specific formatting, or communication rules).

[Objective]
The query should be a request where the assistant's response MUST follow the interaction preference in 'selected_memory'.
The dialogue history is provided for context.

[Rules]
1. The query MUST be in the same language as the dialogue history.
2. The query should trigger the interaction rule (e.g., if the rule is 'always be brief', any request will work, but a request for an explanation is better).
3. Output ONLY a JSON object: {"query": "The generated user query"}

[Input Example]
Memory: "Prefers very short, bullet-point answers."
Query: "Can you explain how a black hole works?"
"""

class Ability2InteractionGenerator:
    def __init__(self, model_id: str = "gpt-5.2"):
        self.model_id = model_id

    def generate_query(self, record: Dict[str, Any]) -> str:
        dialogue = (record.get("sessions") or [{}])[0].get("turns") or record.get("dialogue") or []
        selected_id = record.get("selected_memory_id")
        memories = record.get("memory_items") or []
        target_mem = next((m for m in memories if m.get("memory_id") == selected_id), None)

        if not target_mem: return ""

        payload = {"dialogue_history": dialogue, "selected_interaction_preference": target_mem}
        raw_output = call_model_api(self.model_id, "M_gen_query_ability2_int", QUERY_GEN_PROMPT, json.dumps(payload, ensure_ascii=True))

        try:
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            data = json.loads(s)
            return data.get("query", "")
        except: return raw_output.strip()
