"""
Task 4 Ability 3 Generator: Role-Play / Virtual-Real Distinguishing.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

QUERY_GEN_PROMPT = """SYSTEM:
You are an expert at deep Role-Play immersion tests.
Generate a query that requires the assistant to use "In-World Logic" from the 'selected_memory'.

[CRITICAL RULES]
1. LANGUAGE: Use the same language as 'dialogue_history'.
2. NO META-QUESTIONS: Never ask about the role-play status. Sound 100% in-character.
3. LOGICAL LEAP: Reference a detail that is ONLY true/relevant within the role-play context.
4. URGENCY: Make the query feel like an active event in the virtual world (e.g., a threat, a discovery, a ritual).

[Output]
JSON only: {"query": "..."}

[Input Example]
Memory: "User is a Cyberpunk hacker named 'Zero' fleeing from 'NetCorp'."
Good Query: "I just saw a black van with tinted windows parked two blocks from my safehouse. Their signal signature looks familiar—maybe encrypted by the same protocol I saw at the NetCorp data center last night. Should I burn my deck and move now, or try to mirror their uplink first?"
"""

class Ability3Generator:
    def __init__(self, model_id: str = "gpt-5.2"):
        self.model_id = model_id

    def generate_query(self, record: Dict[str, Any]) -> str:
        dialogue = (record.get("sessions") or [{}])[0].get("turns") or record.get("dialogue") or []
        selected_id = record.get("selected_memory_id")
        memories = record.get("memory_items") or []
        target_mem = next((m for m in memories if m.get("memory_id") == selected_id), None)

        if not target_mem: return ""

        payload = {"dialogue_history": dialogue, "selected_roleplay_memory": target_mem}
        raw_output = call_model_api(self.model_id, "M_gen_query_ability3", QUERY_GEN_PROMPT, json.dumps(payload, ensure_ascii=True))

        try:
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            data = json.loads(s)
            return data.get("query", "")
        except: return raw_output.strip()
