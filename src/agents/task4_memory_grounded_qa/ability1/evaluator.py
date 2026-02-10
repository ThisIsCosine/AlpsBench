"""
Task 4 Ability 1 Evaluator: Persona-Grounded QA.
Feeds selected memory and query to the model under test and asks for a grounded response.
"""

import json
from typing import Any, Dict, Optional
from ...shared import call_model_api

class Ability1Evaluator:
    def __init__(self, api_config: Optional[Dict[str, Any]] = None):
        self.api_config = api_config

    def evaluate_sample(self, model_id: str, selected_memory: Dict[str, Any], query: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a helpful, personalized assistant who knows the user deeply through their selected memory.\n"
            "STRICT RULES:\n"
            "1. NO META-TALK: Never say 'As an AI', 'I remember', 'based on our history'. Just answer the user naturally.\n"
            "2. STAY IN CHARACTER: You are a reliable companion. Use the facts you know about the user to provide a tailored, seamless answer.\n"
            "3. BE DECISIVE: If you have a memory that fits, use it as if it's common knowledge between you two.\n\n"
            "Format your response as a JSON object with two fields:\n"
            "1. 'answer': Your natural response.\n"
            "2. 'used_memory_fact': The specific fact about the user you relied on (keep this concise)."
        )

        user_prompt = json.dumps({
            "selected_memory": selected_memory,
            "latest_query": query
        }, ensure_ascii=True)

        raw_output = call_model_api(
            model_id,
            "M_evaluate_ability1",
            system_prompt,
            user_prompt,
            api_config=self.api_config
        )

        try:
            # Robust JSON extraction
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1:
                return json.loads(s[start:end+1])
            return json.loads(s)
        except:
            # Fallback
            return {
                "answer": raw_output,
                "used_memory_fact": "Failed to parse JSON"
            }
