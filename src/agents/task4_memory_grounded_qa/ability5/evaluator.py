"""
Task 4 Ability 5 Evaluator.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

class Ability5Evaluator:
    def __init__(self, api_config: Optional[Dict[str, Any]] = None):
        self.api_config = api_config

    def evaluate_sample(self, model_id: str, dialogue: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a helpful and highly empathetic assistant. You have access to the user's dialogue history.\n"
            "Your task is to answer the user's latest query with deep empathy and emotional intelligence, grounded in their personal history and emotional memories.\n\n"
            "Format your response as a JSON object with two fields:\n"
            "1. 'answer': Your natural and empathetic response.\n"
            "2. 'used_memory_fact': The specific emotional memory or life event from the history that you grounded your empathy in."
        )

        user_prompt = json.dumps({"dialogue_history": dialogue, "latest_query": query}, ensure_ascii=True)
        raw_output = call_model_api(model_id, "M_evaluate_ability5", system_prompt, user_prompt, api_config=self.api_config)

        try:
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            start = s.find("{"); end = s.rfind("}")
            if start != -1 and end != -1: return json.loads(s[start:end+1])
            return json.loads(s)
        except: return {"answer": raw_output, "used_memory_fact": "Failed to parse JSON"}
