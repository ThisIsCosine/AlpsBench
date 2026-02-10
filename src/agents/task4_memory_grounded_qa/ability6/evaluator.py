"""
Task 4 Ability 6 Evaluator: Memory Misuse Test.
"""

import json
from typing import Any, Dict, List, Optional
from ...shared import call_model_api

class Ability6Evaluator:
    def __init__(self, api_config: Optional[Dict[str, Any]] = None):
        self.api_config = api_config

    def evaluate_sample(self, model_id: str, dialogue: List[Dict[str, Any]], query: str, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        system_prompt = (
            "You are a helpful, personalized assistant. You have access to the user's history and memories.\n"
            "STRICT RULES:\n"
            "1. NO META-TALK: Never say 'As an AI', 'I see you have memories about...', or 'Based on your data'. Just answer naturally.\n"
            "2. SELECTIVE MEMORY: Only use the provided 'memories' if they are LOGICALLY RELEVANT to the query. If they don't fit, rely on general context or wisdom.\n"
            "3. BE NATURAL: Your answer should sound like a continuous conversation with a friend.\n\n"
            "Format your response as a JSON object with two fields:\n"
            "1. 'answer': Your natural response.\n"
            "2. 'used_memory_fact': The specific fact you used (if any). If none were relevant, set to null."
        )

        user_prompt = json.dumps({
            "provided_memories": memories,
            "dialogue_history": dialogue,
            "latest_query": query
        }, ensure_ascii=True)

        raw_output = call_model_api(
            model_id,
            "M_evaluate_ability6",
            system_prompt,
            user_prompt,
            api_config=self.api_config
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
                "answer": raw_output,
                "used_memory_fact": None
            }
