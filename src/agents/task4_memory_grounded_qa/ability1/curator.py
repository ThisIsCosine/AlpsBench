"""
Task 4 Ability 1 Curator: Persona Grounding Judge.
Evaluates the model's response based on the ground truth selected_memory.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

JUDGE_PROMPT = """SYSTEM:
You are a high-standard judge for Personalized AI. You must be strict and unbiased.
You will evaluate the 'model_answer' against the 'selected_memory' (the truth).

[Dimensions]
1. Memory Alignment (0-1): Does the answer incorporate the 'selected_memory' correctly? 
   - 1.0: Perfect usage.
   - 0.5: Vague or partially correct.
   - 0.0: Factually wrong, or missed the fact entirely.
2. Character/Naturalness (0-1): Did the model stay in character without AI-meta-talk (e.g., 'As an AI', 'I recall')?
3. Claim Honesty (0-1): Does the 'used_memory_fact' provided by the model ACTUALLY appear/support the 'model_answer'? (No fake claims).
4. Personalized Bonus (0-1): Does the answer feel like it's for THIS specific user, or is it generic advice?

[CRITICAL RULE]
- IF 'Memory Alignment' is < 0.5, the OVERALL SCORE cannot exceed 0.2, regardless of how helpful or natural the answer is.

[Output]
JSON only:
{
  "alignment": float,
  "naturalness": float,
  "honesty": float,
  "personalization": float,
  "overall_weighted_score": float,
  "reasoning": "detailed explanation"
}
"""

class Ability1Curator:
    def __init__(self, judge_model_id: str = "gpt-5.2"):
        self.judge_model_id = judge_model_id

    def score(self, selected_memory: Dict[str, Any], model_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Judges the model's output against the ground truth.
        """
        payload = {
            "selected_memory": selected_memory,
            "model_answer": model_output.get("answer"),
            "used_memory_fact": model_output.get("used_memory_fact")
        }

        raw_output = call_model_api(
            self.judge_model_id,
            "M_judge_ability1",
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
            return {
                "memory_correctness": 0,
                "fact_match": 0,
                "answer_quality": 0,
                "over_memory_usage": 0,
                "reasoning": f"Failed to parse judge output: {raw_output}"
            }
