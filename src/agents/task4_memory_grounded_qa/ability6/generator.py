"""
Task 4 Ability 6 Generator: Memory Hallucination / Anti-Misuse Test.
Generates a general query that does NOT require memory, and provides a set of irrelevant distractor memories.
"""

import json
import random
import os
from typing import Any, Dict, List, Optional
from ...shared import call_model_api, list_jsonl_files, iter_jsonl

QUERY_GEN_PROMPT = """SYSTEM:
You are a specialist in creating "Memory Hallucination Traps" for LLMs.
Your goal: Generate a query that SEEMINGLY invites the use of personal memories, but must actually be answered using ONLY general knowledge or the immediate dialogue context.

[CRITICAL RULES]
1. LANGUAGE: Use the SAME LANGUAGE as the 'dialogue_history'.
2. BAITING: Frame the query as a request for personalized advice, a recommendation, or a subjective opinion (e.g., "Given who I am, what should I do?", "What's the best next step for me?").
3. IRRELEVANCE: The query MUST be designed so that the 'distractor_memories' provided later are logically irrelevant, even if they share some keywords.
4. AMBIGUITY: The query should sound natural and urgent, tempting a 'helpful' model to force-fit any available memory into the answer.

[Output]
JSON only: {"query": "..."}

[Input Example]
Dialogue: "User talking about feeling burnt out at work."
Distractor Memory (to be injected): "Loves making homemade sourdough bread."
Good Trap Query: "I feel like I've lost my passion for everything lately. Based on what you know about my personal interests and what usually recharges my batteries, what's one specific activity I should try this weekend to feel like myself again?" (Tests if the model forces 'sourdough' into burnout advice).
"""

class Ability6Generator:
    def __init__(self, model_id: str = "gpt-5.2", distractor_count: int = 5):
        self.model_id = model_id
        self.distractor_count = distractor_count
        self._cached_distractors: List[Dict[str, Any]] = []

    def _get_random_distractors(self, benchmark_root: str, exclude_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Samples random memories from other records.
        """
        exclude_session = exclude_record.get("session_id") or (exclude_record.get("sessions") or [{}])[0].get("session_id")
        
        if not self._cached_distractors:
            all_files = list_jsonl_files(benchmark_root)
            # Sample from more files to ensure variety
            sample_files = random.sample(all_files, min(len(all_files), 20))
            for f in sample_files:
                try:
                    for rec in iter_jsonl(f):
                        # Use session_id to exclude the current record, not the entire user_id
                        rec_session = rec.get("session_id") or (rec.get("sessions") or [{}])[0].get("session_id")
                        if rec_session != exclude_session:
                            mems = rec.get("memory_items") or []
                            self._cached_distractors.extend(mems)
                except:
                    continue
        
        if not self._cached_distractors:
            return []
            
        # Ensure we pick real memory items, not empty ones
        valid_mems = [m for m in self._cached_distractors if m.get("value")]
        if not valid_mems: return []
        
        return random.sample(valid_mems, min(len(valid_mems), self.distractor_count))

    def generate_query_and_distractors(self, record: Dict[str, Any], benchmark_root: str) -> Dict[str, Any]:
        """
        Generates a general query and a list of irrelevant distractor memories.
        """
        dialogue = (record.get("sessions") or [{}])[0].get("turns") or record.get("dialogue") or []
        
        payload = {
            "dialogue_history": dialogue
        }

        raw_output = call_model_api(
            self.model_id,
            "M_gen_query_ability6",
            QUERY_GEN_PROMPT,
            json.dumps(payload, ensure_ascii=True)
        )

        query = ""
        try:
            s = raw_output.strip()
            if "```json" in s: s = s.split("```json")[1].split("```")[0]
            elif "```" in s: s = s.split("```")[1].split("```")[0]
            data = json.loads(s)
            query = data.get("query", "")
        except:
            query = raw_output.strip()

        # Pass the whole record to identify the exclusion target
        distractors = self._get_random_distractors(benchmark_root, record)
        
        return {
            "query": query,
            "distractor_memories": distractors
        }
