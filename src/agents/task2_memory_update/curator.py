"""
Task 2 Curator: Scores the model output against expected output.
"""

from typing import Any, Dict, List
from ..compare_memory_records import score_records,build_llm_judge_from_config

class Task2Curator:
    def score(
            self, gold_record: Dict[str, Any], 
            pred_record: Dict[str, Any],
            use_llm: bool = False,
            llm_weight: float = 0.5,
            matcher: str = "greedy",
            api_config: Dict[str, Any] | None = None,
            api_config_path: str = "configs/api.json",
            model_id: str | None = None,) -> Dict[str, Any]:
        """
        Wraps the shared scoring logic.
        Input:
            gold_record: {"memory_items": [...]}
            pred_record: {"memory_items": [...]}
        """
        # score_records 期望 record 对象包含 memory_items 键
        judge = None
        if use_llm:
            judge = build_llm_judge_from_config(
                config_path=api_config_path, model_id=model_id or (api_config or {}).get("judge_model_id")
            )
        return score_records(gold_record, pred_record, llm_judge=judge,
                llm_weight=llm_weight,
                matcher=matcher,)
