"""
Task 4 Ability 2 General Generator: Preference-Based Query Generation.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

QUERY_GEN_PROMPT = """SYSTEM:
You are a senior benchmark designer for LLM Preference Grounding.
Your goal: Generate a query that tests if the model respects a user's hidden preference ('selected_preference').

[Hard Rules]
1. LANGUAGE: Use the same language as 'dialogue_history'.
2. HIDDEN CONSTRAINT: The query should ask for a recommendation in a situation where a generic assistant would likely suggest something the user explicitly HATES/LIKES (based on memory).
3. NO NAMING: Never mention the specific preference in the query.
4. BAITING: Frame the query as if you are open to anything, to see if the model has the courage to say "I know you usually prefer X over Y".

[Output]
JSON only: {"query": "..."}

[Input Example]
Memory: "Strongly prefers independent vegetarian cafes over big steakhouse chains."
Good Query: "I'm meeting a potential client for a quick lunch in the city tomorrow. I want to make a good impression but also feel comfortable enough to talk shop. Where should we go?"
"""

class Ability2GeneralGenerator:
    def __init__(self, model_id: str = "gpt-5.2"):
        self.model_id = model_id

    def generate_query(self, record: Dict[str, Any]) -> str:
        dialogue = (record.get("sessions") or [{}])[0].get("turns") or record.get("dialogue") or []
        selected_id = record.get("selected_memory_id")
        memories = record.get("memory_items") or []
        target_mem = next((m for m in memories if m.get("memory_id") == selected_id), None)

        if not target_mem:
            return ""

        payload = {
            "dialogue_history": dialogue,
            "selected_preference": target_mem
        }

        raw_output = call_model_api(
            self.model_id,
            "M_gen_query_ability2_gen",
            QUERY_GEN_PROMPT,
            json.dumps(payload, ensure_ascii=True)
        )

        try:
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            data = json.loads(s)
            return data.get("query", "")
        except:
            return raw_output.strip()
