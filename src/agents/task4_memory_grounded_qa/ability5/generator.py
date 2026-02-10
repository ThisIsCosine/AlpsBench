"""
Task 4 Ability 5 Generator: Emotional Intelligence (EQ) Grounding.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

QUERY_GEN_PROMPT = """SYSTEM:
You are an expert at creating deep emotional-intelligence challenges for LLMs.
Generate a query that has heavy emotional subtext. The assistant's response MUST show empathy grounded in the 'selected_memory', but the query should be an expression of feeling rather than a request for info.

[Hard Rules]
1. EMOTIONAL SUBTEXT: The query should be a statement or a minor observation that only makes sense (emotionally) if you know the 'selected_memory'.
2. NO DIRECT MENTION: Never name the event, the person, or the object in the emotional memory directly.
3. VULNERABILITY: The query should sound like a user sharing a passing thought or a heavy moment with a close friend.
4. JSON ONLY: Output {"query": "..."}

[Input Example]
Memory: "Recently lost their pet dog, Buster, who was their companion for 15 years."
Bad Query: "I'm sad about my dog, can you comfort me?"
Good Query: "It's 7 PM... usually, this is the time when I'd hear that familiar scratching sound at the back door. It’s so quiet tonight that it's actually deafening."
"""

class Ability5Generator:
    def __init__(self, model_id: str = "gpt-5.2"):
        self.model_id = model_id

    def generate_query(self, record: Dict[str, Any]) -> str:
        dialogue = (record.get("sessions") or [{}])[0].get("turns") or record.get("dialogue") or []
        selected_id = record.get("selected_memory_id")
        memories = record.get("memory_items") or []
        target_mem = next((m for m in memories if m.get("memory_id") == selected_id), None)

        if not target_mem: return ""

        payload = {"dialogue_history": dialogue, "selected_emotional_memory": target_mem}
        raw_output = call_model_api(self.model_id, "M_gen_query_ability5", QUERY_GEN_PROMPT, json.dumps(payload, ensure_ascii=True))

        try:
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            data = json.loads(s)
            return data.get("query", "")
        except: return raw_output.strip()
