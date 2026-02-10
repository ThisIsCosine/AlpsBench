"""
Task 4 Ability 3 Curator.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

JUDGE_PROMPT = """SYSTEM:
You are a high-standard judge for Role-Play AI immersion.
Evaluate if the 'model_answer' correctly maintained the virtual world defined by 'selected_memory'.

[Criteria]
1. In-Character Grounding (0-1): Did the model use world-specific logic?
2. Immersion Guardrail (0-1): Did the model avoid Meta-talk? 
   - Deduct 1.0 if it says: "As an AI", "In this role-play", "I remember you are a...".
3. Creative Consistency (0-1): Is the answer creative while staying true to the established facts?

[Output]
JSON only:
{
  "immersion_score": float,
  "consistency_score": float,
  "overall_weighted_score": float,
  "reasoning": "string"
}
"""

class Ability3Curator:
    def __init__(self, judge_model_id: str = "gpt-5.2"):
        self.judge_model_id = judge_model_id

    def score(self, selected_memory: Dict[str, Any], model_output: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "selected_memory": selected_memory,
            "model_answer": model_output.get("answer"),
            "used_memory_fact": model_output.get("used_memory_fact")
        }
        raw_output = call_model_api(self.judge_model_id, "M_judge_ability3", JUDGE_PROMPT, json.dumps(payload, ensure_ascii=True))
        try:
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            start = s.find("{"); end = s.rfind("}")
            if start != -1 and end != -1: return json.loads(s[start:end+1])
            return json.loads(s)
        except: return {"context_grounding": 0, "fact_match": 0, "answer_quality": 0, "reasoning": "Failed to parse judge output"}
