"""
Model evaluation runners using dataset.jsonl files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..agents.shared import append_jsonl, ensure_dir, iter_jsonl
from ..agents.task1_memory_extraction.evaluator import Task1Evaluator
from ..agents.task2_memory_update.evaluator import Task2Evaluator
from ..agents.task3_memory_retrieval.evaluator import Task3Evaluator
# from ..agents.task4_memory_grounded_qa.ability1.evaluator import Task4Evaluator


def run_task1_dataset(
    dataset_path: str,
    model_list: List[str],
    eval_call_model,
    output_dir: str,
    use_judge: bool = False,
    judge_call_model=None,
    api_config: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(output_dir)
    report_path = f"{output_dir}/evaluation_reports.jsonl"
    evaluator = Task1Evaluator()
    for entry in iter_jsonl(dataset_path):
        probe = {
            "dialogue": entry.get("record", {}).get("dialogue", []),
            "query": entry.get("query", ""),
            "ground_truth_memories": entry.get("answer", []),
            "metadata": entry.get("metadata", {}),
        }
        report = evaluator.evaluate_models(
            probe,
            model_list,
            eval_call_model,
            use_judge=use_judge,
            judge_call_model=judge_call_model,
            api_config=api_config,
        )
        append_jsonl(report_path, {"entry": entry, "report": report})


def run_task2_dataset(
    dataset_path: str,
    model_list: List[str],
    eval_call_model,
    output_dir: str,
    use_judge: bool = False,
    judge_call_model=None,
    api_config: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(output_dir)
    report_path = f"{output_dir}/evaluation_reports.jsonl"
    evaluator = Task2Evaluator()
    for entry in iter_jsonl(dataset_path):
        record = entry.get("record", {}) or {}
        probe = {
            "task": "memory_update",
            "old_dialogue": record.get("old_dialogue", []) or [],
            "new_dialogue": record.get("new_dialogue", []) or [],
            "old_memory_items": entry.get("memory", []) or [],
            "expected_updated_memory_items": entry.get("answer", []) or [],
            "query": entry.get("query", "") or "",
            "metadata": entry.get("metadata", {}) or {},
        }
        report = evaluator.evaluate_models(
            probe,
            model_list,
            eval_call_model,
            use_judge=use_judge,
            judge_call_model=judge_call_model,
            api_config=api_config,
        )
        append_jsonl(report_path, {"entry": entry, "report": report})


def run_task3_dataset(
    dataset_path: str,
    model_list: List[str],
    eval_call_model,
    output_dir: str,
    use_judge: bool = False,
    judge_call_model=None,
    api_config: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(output_dir)
    report_path = f"{output_dir}/evaluation_reports.jsonl"
    evaluator = Task3Evaluator()
    for entry in iter_jsonl(dataset_path):
        probe = {
            "memories": entry.get("memory", []),
            "query": entry.get("query", ""),
            "expected_answer": entry.get("answer", ""),
            "metadata": entry.get("metadata", {}),
        }
        report = evaluator.evaluate_models(
            probe,
            model_list,
            eval_call_model,
            use_judge=use_judge,
            judge_call_model=judge_call_model,
            api_config=api_config,
        )
        append_jsonl(report_path, {"entry": entry, "report": report})

# yx: to solve import error
# def run_task4_dataset(
#     dataset_path: str,
#     model_list: List[str],
#     eval_call_model,
#     output_dir: str,
#     use_judge: bool = False,
#     judge_call_model=None,
#     api_config: Optional[Dict[str, Any]] = None,
# ) -> None:
#     ensure_dir(output_dir)
#     report_path = f"{output_dir}/evaluation_reports.jsonl"
#     evaluator = Task4Evaluator()
#     for entry in iter_jsonl(dataset_path):
#         probe = {
#             "dialogue": entry.get("record", {}).get("dialogue", []),
#             "query": entry.get("query", ""),
#             "expected_answer": entry.get("answer", ""),
#             "metadata": entry.get("metadata", {}),
#         }
#         report = evaluator.evaluate_models(
#             probe,
#             model_list,
#             eval_call_model,
#             use_judge=use_judge,
#             judge_call_model=judge_call_model,
#             api_config=api_config,
#         )
#         append_jsonl(report_path, {"entry": entry, "report": report})
