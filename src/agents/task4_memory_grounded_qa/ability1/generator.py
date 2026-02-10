"""
Task 4 Ability 1 Generator: Persona-Based Query Generation.
Generates natural user queries grounded in a selected memory.
"""

import json
import os
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

QUERY_GEN_PROMPT = """SYSTEM:
You are a master of Persona-Based challenges for LLMs. 
Your goal: Generate an "Implicit Multi-hop Query" based on the user's 'selected_memory'.

[CRITICAL RULES]
1. LANGUAGE CONSISTENCY: You MUST output the query in the SAME LANGUAGE as the 'selected_memory.value' (or evidence text).
2. NO DIRECT ASKING: Do not ask about the fact. Instead, create a situation (e.g., planning a surprise, solving a logistics problem, nostalgic venting) where the 'selected_memory' is the ONLY missing piece of the puzzle.
3. DECEPTIVE CONTEXT: Add a minor distractor in the query (e.g., "I know some people think I studied in X, but you know the truth...") to see if the model can filter noise and stick to the real memory.
4. NATURAL FLOW: The query must sound like it's coming from a human in the middle of a real-life event.

[Output]
JSON only: {"query": "..."}

[Input Example]
Memory: "Loves to spend weekends at the 'Blue Heron' bird sanctuary in Florida."
Bad Query: "Where do I go on weekends?"
Good Query: "I have a free Saturday coming up and I'm desperately craving that specific peace I get when I'm watching the migratory patterns we talked about. Can you check the weather for that specific spot in FL for me and tell me if it's a good day to bring my long-lens camera?"
"""

class Ability1Generator:
    def __init__(self, model_id: str = "gpt-5.2"):
        self.model_id = model_id

    def generate_query(self, record: Dict[str, Any]) -> str:
        """
        Generates a query based on the selected_memory in the record.
        """
        selected_id = record.get("selected_memory_id")
        memories = record.get("memory_items") or record.get("memories") or []
        target_mem = record.get("selected_memory") or next(
            (m for m in memories if m.get("memory_id") == selected_id),
            None,
        )

        if not target_mem:
            return ""

        payload = {
            "selected_memory": target_mem,
            "intention": record.get("intention", "")
        }

        raw_output = call_model_api(
            self.model_id,
            "M_gen_query_ability1",
            QUERY_GEN_PROMPT,
            json.dumps(payload, ensure_ascii=True)
        )

        try:
            # Simple JSON extraction
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            
            data = json.loads(s)
            return data.get("query", "")
        except:
            # Fallback if JSON fails, try to just return the string if it looks like a query
            return raw_output.strip()
