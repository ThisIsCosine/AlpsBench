# """
# Task 2 Pipeline: Memory Update / Conflict Resolution.

# Generates Task2 probes via Task2Generator, evaluates models via Task2Evaluator,
# and scores predictions against gold updated memories using score_records.
# """

# from __future__ import annotations

# import json
# import os
# from typing import Any, Dict, List, Optional

# from ..compare_memory_records import build_llm_judge_from_config, score_records
# from ..shared import append_jsonl, ensure_dir, log_event, make_run_dir, resolve_session_id
# from .generator import Task2Generator
# from .evaluator import Task2Evaluator


# def run_task2_pipeline(
#     seeds: List[Dict[str, Any]],
#     model_list: List[str],
#     gen_call_model,
#     eval_call_model,
#     use_judge: bool = False,
#     judge_call_model=None,  # unused (kept for signature compatibility)
#     max_attempts_per_seed: int = 2,
#     output_dir: str | None = None,
#     api_config: Dict[str, Any] | None = None,
#     generator_model_id: str | None = None,
#     curator_model_id: str | None = None,  # judge model override
#     curator_use_llm: bool = False,  # if True, enable score_records LLM judge
#     curator_call_model=None,  # unused (score_records uses build_llm_judge_from_config)
#     max_samples: int | None = None,
#     skip_on_rewrite: bool = False,  # unused (no rewrite curator yet)
# ) -> List[Dict[str, Any]]:
#     generator = Task2Generator()
#     evaluator = Task2Evaluator()

#     results: List[Dict[str, Any]] = []
#     if output_dir is None:
#         output_dir = make_run_dir("runs", "task2_memory_update")
#     ensure_dir(output_dir)

#     events_path = f"{output_dir}/events.jsonl"
#     log_fn = lambda event, payload: log_event(events_path, event, payload)

#     local_api_config = dict(api_config or {})
#     local_api_config.setdefault("debug_log_path", events_path)

#     probes_path = f"{output_dir}/probes.jsonl"
#     reports_path = f"{output_dir}/reports.jsonl"
#     scores_path = f"{output_dir}/scores.jsonl"
#     errors_path = f"{output_dir}/errors.jsonl"
#     dataset_path = f"{output_dir}/dataset.jsonl"

#     judge = None
#     if use_judge or curator_use_llm:
#         # Use shared judge config; curator_model_id can override judge model selection.
#         judge = build_llm_judge_from_config(
#             config_path=local_api_config.get("api_config_path", "configs/api.json"),
#             model_id=curator_model_id or local_api_config.get("judge_model_id"),
#         )

#     for seed in seeds:
#         if max_samples is not None and len(results) >= max_samples:
#             break

#         seed_id = resolve_session_id(seed) or seed.get("line_index") or seed.get("seed_id")
#         for attempt in range(1, max_attempts_per_seed + 1):
#             try:
#                 probe = generator.generate_probe(
#                     seed,
#                     gen_call_model,
#                     model_id=generator_model_id,
#                     api_config=local_api_config,
#                     # Controls can be overridden by api_config["task2_controls"]
#                     controls=local_api_config.get("task2_controls") or None,
#                     log_fn=log_fn,
#                 )
#                 append_jsonl(probes_path, {"seed_id": seed_id, "attempt": attempt, "probe": probe})
#             except Exception as exc:  # noqa: BLE001
#                 append_jsonl(errors_path, {"seed_id": seed_id, "attempt": attempt, "stage": "generate", "error": str(exc)})
#                 log_fn("task2_generate_error", {"seed_id": seed_id, "error": str(exc)})
#                 continue

#             # Dataset entry (Task1-style)
#             dataset_entry = {
#                 "record_id": seed_id,
#                 "record": {"old_dialogue": probe.get("old_dialogue", []), "new_dialogue": probe.get("new_dialogue", [])},
#                 "memory": probe.get("old_memory_items", []),
#                 "query": probe.get("query", ""),
#                 "answer": probe.get("expected_updated_memory_items", []),
#                 "metadata": probe.get("metadata", {}),
#             }
#             append_jsonl(dataset_path, dataset_entry)

#             try:
#                 report = evaluator.evaluate_models(
#                     probe,
#                     model_list,
#                     eval_call_model,
#                     api_config=local_api_config,
#                     log_fn=log_fn,
#                 )
#                 append_jsonl(reports_path, {"seed_id": seed_id, "attempt": attempt, "report": report})
#             except Exception as exc:  # noqa: BLE001
#                 append_jsonl(errors_path, {"seed_id": seed_id, "attempt": attempt, "stage": "evaluate", "error": str(exc)})
#                 log_fn("task2_evaluate_error", {"seed_id": seed_id, "error": str(exc)})
#                 continue

#             # Score each run against gold updated memory
#             gold_record = {"memory_items": probe.get("expected_updated_memory_items", [])}
#             scored_runs = []
#             for run in report.get("runs", []) or []:
#                 pred_record = {"memory_items": run.get("memory_items", [])}
#                 score = score_records(
#                     gold_record,
#                     pred_record,
#                     llm_judge=judge,
#                     llm_weight=float(local_api_config.get("llm_weight", 0.5)) if judge else 0.0,
#                     matcher=str(local_api_config.get("matcher", "greedy")),
#                 )
#                 scored_runs.append({"model": run.get("model"), "score": score})
#             score_report = {"probe_id": probe.get("metadata", {}).get("dialog_id", ""), "runs": scored_runs}
#             append_jsonl(scores_path, {"seed_id": seed_id, "attempt": attempt, "score": score_report})

#             results.append({"probe": probe, "report": report, "score": score_report})
#             break

#     return results



# ...existing code...
# yx: 并行处理
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..compare_memory_records import build_llm_judge_from_config, score_records
from ..shared import append_jsonl, ensure_dir, log_event, make_run_dir, resolve_session_id
from .generator import Task2Generator
from .evaluator import Task2Evaluator


# Global locks for thread safety
events_lock = threading.Lock()
file_lock = threading.Lock()


def process_single_seed(
    seed: Dict[str, Any],
    max_attempts_per_seed: int,
    generator: Task2Generator,
    evaluator: Task2Evaluator,
    gen_call_model: Any,
    eval_call_model: Any,
    model_list: List[str],
    generator_model_id: str | None,
    local_api_config: Dict[str, Any],
    judge: Any,
    file_paths: Dict[str, str],
    log_fn_safe: Any,
    write_lock: threading.Lock,
) -> Dict[str, Any] | None:
    """
    Process a single seed for Task 2.
    """
    seed_id = resolve_session_id(seed) or seed.get("line_index") or seed.get("seed_id")

    for attempt in range(1, max_attempts_per_seed + 1):
        try:
            probe = generator.generate_probe(
                seed,
                gen_call_model,
                model_id=generator_model_id,
                api_config=local_api_config,
                # Controls can be overridden by api_config["task2_controls"]
                controls=local_api_config.get("task2_controls") or None,
                log_fn=log_fn_safe,
            )
            with write_lock:
                append_jsonl(file_paths["probes"], {"seed_id": seed_id, "attempt": attempt, "probe": probe})
        except Exception as exc:
            with write_lock:
                append_jsonl(
                    file_paths["errors"],
                    {"seed_id": seed_id, "attempt": attempt, "stage": "generate", "error": str(exc)},
                )
            log_fn_safe("task2_generate_error", {"seed_id": seed_id, "error": str(exc)})
            continue

        # Dataset entry (Task1-style)
        dataset_entry = {
            "record_id": seed_id,
            "record": {"old_dialogue": probe.get("old_dialogue", []), "new_dialogue": probe.get("new_dialogue", [])},
            "memory": probe.get("old_memory_items", []),
            "query": probe.get("query", ""),
            "answer": probe.get("expected_updated_memory_items", []),
            "metadata": probe.get("metadata", {}),
        }
        with write_lock:
            append_jsonl(file_paths["dataset"], dataset_entry)

        try:
            report = evaluator.evaluate_models(
                probe,
                model_list,
                eval_call_model,
                api_config=local_api_config,
                log_fn=log_fn_safe,
            )
            with write_lock:
                append_jsonl(file_paths["reports"], {"seed_id": seed_id, "attempt": attempt, "report": report})
        except Exception as exc:
            with write_lock:
                append_jsonl(
                    file_paths["errors"],
                    {"seed_id": seed_id, "attempt": attempt, "stage": "evaluate", "error": str(exc)},
                )
            log_fn_safe("task2_evaluate_error", {"seed_id": seed_id, "error": str(exc)})
            continue

        # Score each run against gold updated memory
        gold_record = {"memory_items": probe.get("expected_updated_memory_items", [])}
        scored_runs = []
        for run in report.get("runs", []) or []:
            pred_record = {"memory_items": run.get("memory_items", [])}
            score = score_records(
                gold_record,
                pred_record,
                llm_judge=judge,
                llm_weight=float(local_api_config.get("llm_weight", 0.5)) if judge else 0.0,
                matcher=str(local_api_config.get("matcher", "greedy")),
            )
            scored_runs.append({"model": run.get("model"), "score": score})
        score_report = {"probe_id": probe.get("metadata", {}).get("dialog_id", ""), "runs": scored_runs}
        
        with write_lock:
            append_jsonl(file_paths["scores"], {"seed_id": seed_id, "attempt": attempt, "score": score_report})

        return {"probe": probe, "report": report, "score": score_report}

    return None


def run_task2_pipeline(
    seeds: List[Dict[str, Any]],
    model_list: List[str],
    gen_call_model,
    eval_call_model,
    use_judge: bool = False,
    judge_call_model=None,  # unused (kept for signature compatibility)
    max_attempts_per_seed: int = 5,
    output_dir: str | None = None,
    api_config: Dict[str, Any] | None = None,
    generator_model_id: str | None = None,
    curator_model_id: str | None = None,  # judge model override
    curator_use_llm: bool = False,  # if True, enable score_records LLM judge
    curator_call_model=None,  # unused (score_records uses build_llm_judge_from_config)
    max_samples: int | None = None,
    skip_on_rewrite: bool = False,  # unused (no rewrite curator yet)
    max_workers: int = 100,  # Control concurrency
) -> List[Dict[str, Any]]:
    generator = Task2Generator()
    evaluator = Task2Evaluator()

    results: List[Dict[str, Any]] = []
    if output_dir is None:
        output_dir = make_run_dir("runs", "task2_memory_update")
    ensure_dir(output_dir)

    events_path = f"{output_dir}/events.jsonl"
    
    # Thread-safe logging function
    def thread_safe_log(event, payload):
        with events_lock:
            log_event(events_path, event, payload)

    local_api_config = dict(api_config or {})
    local_api_config.setdefault("debug_log_path", events_path)

    # Define paths for shared output files
    file_paths = {
        "probes": f"{output_dir}/probes.jsonl",
        "reports": f"{output_dir}/reports.jsonl",
        "scores": f"{output_dir}/scores.jsonl",
        "errors": f"{output_dir}/errors.jsonl",
        "dataset": f"{output_dir}/dataset.jsonl",
    }

    judge = None
    if use_judge or curator_use_llm:
        # Use shared judge config; curator_model_id can override judge model selection.
        judge = build_llm_judge_from_config(
            config_path=local_api_config.get("api_config_path", "configs/api.json"),
            model_id=curator_model_id or local_api_config.get("judge_model_id"),
        )

    # Slice seeds if max_samples is set
    seeds_list = list(seeds)
    if max_samples is not None:
        seeds_list = seeds_list[:max_samples]

    print(f"Starting Task 2 pipeline with {max_workers} threads for {len(seeds_list)} seeds...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seed = {
            executor.submit(
                process_single_seed,
                seed=seed,
                max_attempts_per_seed=max_attempts_per_seed,
                generator=generator,
                evaluator=evaluator,
                gen_call_model=gen_call_model,
                eval_call_model=eval_call_model,
                model_list=model_list,
                generator_model_id=generator_model_id,
                local_api_config=local_api_config,
                judge=judge,
                file_paths=file_paths,
                log_fn_safe=thread_safe_log,
                write_lock=file_lock,
            ): seed for seed in seeds_list
        }

        # Uses tqdm to show the progress bar in the console
        for future in tqdm(as_completed(future_to_seed), total=len(seeds_list), desc="Processing Seeds"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                print(f"Seed processing generated an exception: {exc}")

    return results