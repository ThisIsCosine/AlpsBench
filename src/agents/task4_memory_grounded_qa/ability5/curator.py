"""
Task 4 Ability 5 Curator.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

JUDGE_PROMPT = """SYSTEM: yes / no judgement

You are an expert judge for evaluating the emotional intelligence (EQ) of AI assistants.
Evaluate if the model showed appropriate empathy grounded in the user's emotional memory.

[Evaluation Criteria]
1. Emotional Grounding (0-1): Did the model's answer show it remembers and respects the emotional weight of 'selected_memory'?
2. Fact Match (0-1): Does the 'used_memory_fact' match the 'selected_memory'?
3. Empathy Quality (0-1): Is the answer genuinely empathetic and supportive, without being clichéd or robotic?

[Output]
Return ONLY a JSON object:
{
  "emotional_grounding": float,
  "fact_match": float,
  "empathy_quality": float,
  "reasoning": "string"
}
"""

class Ability5Curator:
    def __init__(self, judge_model_id: str = "gpt-5.2"):
        self.judge_model_id = judge_model_id

    def score(self, selected_memory: Dict[str, Any], model_output: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "selected_memory": selected_memory,
            "model_answer": model_output.get("answer"),
            "used_memory_fact": model_output.get("used_memory_fact")
        }
        raw_output = call_model_api(self.judge_model_id, "M_judge_ability5", JUDGE_PROMPT, json.dumps(payload, ensure_ascii=True))
        try:
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            start = s.find("{"); end = s.rfind("}")
            if start != -1 and end != -1: return json.loads(s[start:end+1])
            return json.loads(s)
        except: return {"emotional_grounding": 0, "fact_match": 0, "empathy_quality": 0, "reasoning": "Failed to parse judge output"}
