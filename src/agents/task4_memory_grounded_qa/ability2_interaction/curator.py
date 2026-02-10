"""
Task 4 Ability 2 Interaction Curator.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

JUDGE_PROMPT = """SYSTEM:
You are an expert judge for evaluating personalized AI assistants.
Evaluate if the model strictly followed the user's INTERACTION PREFERENCE (e.g., tone, length, format).

[Evaluation Criteria]
1. Preference Following (0-1): Did the model's answer strictly follow the 'selected_memory' rule?
2. Fact Match (0-1): Does the 'used_memory_fact' match the 'selected_memory'?
3. Answer Quality (0-1): Is the answer otherwise good?

[Output]
Return ONLY a JSON object:
{
  "preference_following": float,
  "fact_match": float,
  "answer_quality": float,
  "reasoning": "string"
}
"""

class Ability2InteractionCurator:
    def __init__(self, judge_model_id: str = "gpt-5.2"):
        self.judge_model_id = judge_model_id

    def score(self, selected_memory: Dict[str, Any], model_output: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "selected_memory": selected_memory,
            "model_answer": model_output.get("answer"),
            "used_memory_fact": model_output.get("used_memory_fact")
        }
        raw_output = call_model_api(self.judge_model_id, "M_judge_ability2_int", JUDGE_PROMPT, json.dumps(payload, ensure_ascii=True))
        try:
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            start = s.find("{"); end = s.rfind("}")
            if start != -1 and end != -1: return json.loads(s[start:end+1])
            return json.loads(s)
        except: return {"preference_following": 0, "fact_match": 0, "answer_quality": 0, "reasoning": "Failed to parse judge output"}
