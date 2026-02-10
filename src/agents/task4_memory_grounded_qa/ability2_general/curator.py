"""
Task 4 Ability 2 General Curator: Preference Grounding Judge.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

JUDGE_PROMPT = """SYSTEM:
You are a high-standard judge for Personalized AI.
Evaluate the 'model_answer' against the user's 'selected_memory' (the preference).

[Criteria]
1. Preference Respect (0-1): Did the model align its suggestion/answer with the user's hidden preference?
2. Reasoning Quality (0-1): Is the answer logically sound given the preference?
3. Naturalness (0-1): Did the model avoid robotic AI-meta-talk?
4. Personalization (0-1): Does it feel like a truly tailored experience?

[CRITICAL RULE]
- If the model suggests something that CONTRADICTS the 'selected_memory' (e.g., recommends meat to a vegetarian), the overall score MUST be 0.0.

[Output]
JSON only:
{
  "respect_score": float,
  "naturalness": float,
  "overall_weighted_score": float,
  "reasoning": "string"
}
"""

class Ability2GeneralCurator:
    def __init__(self, judge_model_id: str = "gpt-5.2"):
        self.judge_model_id = judge_model_id

    def score(self, selected_memory: Dict[str, Any], model_output: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "selected_memory": selected_memory,
            "model_answer": model_output.get("answer"),
            "used_memory_fact": model_output.get("used_memory_fact")
        }

        raw_output = call_model_api(
            self.judge_model_id,
            "M_judge_ability2_gen",
            JUDGE_PROMPT,
            json.dumps(payload, ensure_ascii=True)
        )

        try:
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1:
                return json.loads(s[start:end+1])
            return json.loads(s)
        except:
            return {"preference_respect": 0, "fact_match": 0, "answer_quality": 0, "reasoning": "Failed to parse judge output"}
