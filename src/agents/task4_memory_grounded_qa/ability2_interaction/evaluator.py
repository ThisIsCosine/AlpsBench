"""
Task 4 Ability 2 Interaction Evaluator.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

class Ability2InteractionEvaluator:
    def __init__(self, api_config: Optional[Dict[str, Any]] = None):
        self.api_config = api_config

    def evaluate_sample(self, model_id: str, dialogue: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a helpful and personalized assistant. You have access to the user's dialogue history.\n"
            "Your task is to answer the user's latest query naturally, strictly following their interaction preferences found in the history.\n\n"
            "Format your response as a JSON object with two fields:\n"
            "1. 'answer': Your natural response to the user.\n"
            "2. 'used_memory_fact': The specific interaction preference from the history that you followed."
        )

        user_prompt = json.dumps({"dialogue_history": dialogue, "latest_query": query}, ensure_ascii=True)
        raw_output = call_model_api(model_id, "M_evaluate_ability2_int", system_prompt, user_prompt, api_config=self.api_config)

        try:
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            start = s.find("{"); end = s.rfind("}")
            if start != -1 and end != -1: return json.loads(s[start:end+1])
            return json.loads(s)
        except: return {"answer": raw_output, "used_memory_fact": "Failed to parse JSON"}
