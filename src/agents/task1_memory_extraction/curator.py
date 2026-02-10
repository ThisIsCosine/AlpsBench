"""
Task 1 Curator: Memory Extraction.
Decides whether a probe is acceptable for evaluation.
"""

from __future__ import annotations

import json
import os

from typing import Any, Dict, Iterable, List

from ..shared import append_jsonl, ensure_dir, call_model_api, log_event, resolve_session_id, make_run_dir
from ..compare_memory_records import score_records, build_llm_judge_from_config
from .generator import Task1Generator
from .evaluator import Task1Evaluator


TASK1_CURATOR_PROMPT = """SYSTEM:
You are Task 1 Curator. Decide whether to accept, rewrite, or discard a memory
extraction probe based on clarity and scorable output.
Return JSON only.
"""

TASK1_CURATOR_LLM_PROMPT = """SYSTEM:
You are Task 1 Curator. Review the probe and return JSON only:
{
  "probe_id": "string",
  "verdict": "ACCEPT|REWRITE|DISCARD",
  "reasons": ["..."],
  "feedback_for_generator": "string"
}
If invalid or ambiguous, choose REWRITE with specific feedback.
"""


class Task1Curator:
    def __init__(self, system_prompt: str = TASK1_CURATOR_PROMPT) -> None:
        self.system_prompt = system_prompt

    def score_report(
        self,
        probe: Dict[str, Any],
        report: Dict[str, Any],
        use_llm: bool = False,
        llm_weight: float = 0.5,
        matcher: str = "greedy",
        api_config: Dict[str, Any] | None = None,
        api_config_path: str = "configs/api.json",
        model_id: str | None = None,
    ) -> Dict[str, Any]:
        gold_record = {"memory_items": probe.get("ground_truth_memories", [])}
        judge = None
        if use_llm:
            judge = build_llm_judge_from_config(
                config_path=api_config_path, model_id=model_id or (api_config or {}).get("judge_model_id")
            )
        runs = []
        for run in report.get("runs", []):
            pred_record = {"memory_items": run.get("memory_items", [])}
            score = score_records(
                gold_record,
                pred_record,
                llm_judge=judge,
                llm_weight=llm_weight,
                matcher=matcher,
            )
            runs.append(
                {
                    "model": run.get("model"),
                    "score": score,
                }
            )
        return {"probe_id": probe.get("metadata", {}).get("dialog_id", ""), "runs": runs}
    def decide(
        self,
        probe: Dict[str, Any],
        use_llm: bool = False,
        curator_model_id: str | None = None,
        curator_call_model=None,
        api_config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if use_llm:
            if curator_call_model is None and not curator_model_id:
                raise ValueError("curator_model_id_required")
            payload = {
                "probe_id": resolve_session_id(probe) or probe.get("metadata", {}).get("dialog_id", ""),
                "query": probe.get("query", ""),
                "ground_truth_memories": probe.get("ground_truth_memories", []),
                "dialogue": probe.get("dialogue", []),
                "metadata": probe.get("metadata", {}),
            }
            if curator_call_model is None:
                raw = call_model_api(
                    curator_model_id, "M_curate", TASK1_CURATOR_LLM_PROMPT, json.dumps(payload), api_config
                )
            else:
                raw = curator_call_model("curator", "M_curate", TASK1_CURATOR_LLM_PROMPT, json.dumps(payload))
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {
                    "probe_id": resolve_session_id(probe) or probe.get("metadata", {}).get("dialog_id", ""),
                    "verdict": "REWRITE",
                    "reasons": ["curator_invalid_output"],
                    "feedback_for_generator": "Curator failed to return valid JSON. Regenerate with clearer structure.",
                }
        reasons: List[str] = []
        verdict = "ACCEPT"
        if not probe.get("dialogue"):
            verdict = "DISCARD"
            reasons.append("missing_dialogue")
        if not probe.get("ground_truth_memories"):
            verdict = "DISCARD"
            reasons.append("missing_ground_truth")
        if not probe.get("query"):
            verdict = "REWRITE"
            reasons.append("missing_query")
        return {
            "probe_id": resolve_session_id(probe) or probe.get("metadata", {}).get("dialog_id", ""),
            "verdict": verdict,
            "reasons": reasons,
            "feedback_for_generator": "",
            "rewritten_probe": None,
        }


# def run_task1_pipeline(
#     records: Iterable[Dict[str, Any]],
#     model_list: List[str],
#     gen_call_model,
#     eval_call_model,
#     use_judge: bool = False,
#     judge_call_model=None,
#     max_attempts_per_record: int = 2,
#     output_dir: str | None = None,
#     api_config: Dict[str, Any] | None = None,
#     generator_model_id: str | None = None,
#     curator_model_id: str | None = None,
#     curator_use_llm: bool = False,
#     curator_call_model=None,
#     max_samples: int | None = None,
#     skip_on_rewrite: bool = False,
# ) -> List[Dict[str, Any]]:
#     generator = Task1Generator()
#     curator = Task1Curator()
#     evaluator = Task1Evaluator()
#     results: List[Dict[str, Any]] = []
#     if output_dir is None:
#         output_dir = make_run_dir("runs", "task1_batch")
#     ensure_dir(output_dir)
#     events_path = f"{output_dir}/events.jsonl"
#     log_fn = lambda event, payload: log_event(events_path, event, payload)
#     local_api_config = dict(api_config or {})
#     local_api_config.setdefault("debug_log_path", events_path)
#     for record in records:
#         if max_samples is not None and len(results) >= max_samples:
#             break
#         record_id = resolve_session_id(record) or record.get("line_index")
#         source_rel = record.get("_source_rel")
#         if source_rel:
#             rel_dir = os.path.splitext(source_rel)[0]
#             record_dir = os.path.join(output_dir, rel_dir)
#         else:
#             record_dir = output_dir
#         ensure_dir(record_dir)
#         probes_path = f"{record_dir}/probes.jsonl"
#         decisions_path = f"{record_dir}/decisions.jsonl"
#         reports_path = f"{record_dir}/reports.jsonl"
#         errors_path = f"{record_dir}/errors.jsonl"
#         dataset_path = f"{record_dir}/dataset.jsonl"
#         for attempt in range(1, max_attempts_per_record + 1):
#             try:
#                 probe = generator.generate_probe(
#                     record, gen_call_model, model_id=generator_model_id, api_config=local_api_config, log_fn=log_fn
#                 )
#                 append_jsonl(probes_path, {"record_id": record_id, "attempt": attempt, "probe": probe})
#             except Exception as exc:  # noqa: BLE001
#                 append_jsonl(errors_path, {"record_id": record_id, "attempt": attempt, "stage": "generate", "error": str(exc)})
#                 log_fn("task1_generate_error", {"record_id": record_id, "error": str(exc)})
#                 continue

#             dataset_entry = {
#                 "record_id": record_id,
#                 "record": {"dialogue": probe.get("dialogue", [])},
#                 "memory": probe.get("ground_truth_memories", []),
#                 "query": probe.get("query", ""),
#                 "answer": probe.get("ground_truth_memories", []),
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
#                 append_jsonl(reports_path, {"record_id": record_id, "attempt": attempt, "report": report})
#             except Exception as exc:  # noqa: BLE001
#                 append_jsonl(errors_path, {"record_id": record_id, "attempt": attempt, "stage": "evaluate", "error": str(exc)})
#                 log_fn("task1_evaluate_error", {"record_id": record_id, "error": str(exc)})
#                 continue

#             score_report = curator.score_report(
#                 probe,
#                 report,
#                 use_llm=use_judge,
#                 llm_weight=0.5,
#                 matcher="greedy",
#                 api_config=local_api_config,
#             )
#             append_jsonl(decisions_path, {"record_id": record_id, "attempt": attempt, "score": score_report})
#             log_fn("task1_curator_score", {"record_id": record_id})

#             results.append({"probe": probe, "report": report, "score": score_report})
#             break
#     return results


import os
import threading
from typing import Dict, Any, List, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 建议安装 tqdm: pip install tqdm

# 全局锁，用于同步写入共享的 events.jsonl
events_lock = threading.Lock()

def process_single_record(
    record: Dict[str, Any],
    output_dir: str,
    max_attempts_per_record: int,
    model_list: List[str],
    gen_call_model,
    eval_call_model,
    generator,
    curator,
    evaluator,
    api_config: Dict[str, Any],
    generator_model_id: str | None,
    use_judge: bool,
    # 辅助函数
    resolve_session_id_fn,
    ensure_dir_fn,
    append_jsonl_fn,
    log_event_fn,
    events_path: str
) -> Dict[str, Any] | None:
    """
    处理单条记录的逻辑，从主函数中提取出来以支持并行化。
    """
    
    # 定义线程安全的 log_fn
    def thread_safe_log(event, payload):
        with events_lock:
            log_event_fn(events_path, event, payload)

    # 这里的 local_api_config 已经是通过 copy 传入的，或者是线程私有的
    local_api_config = dict(api_config or {})
    local_api_config.setdefault("debug_log_path", events_path)

    record_id = resolve_session_id_fn(record) or record.get("line_index")
    source_rel = record.get("_source_rel")
    
    if source_rel:
        rel_dir = os.path.splitext(source_rel)[0]
        record_dir = os.path.join(output_dir, rel_dir)
    else:
        record_dir = output_dir
        
    # 注意：ensure_dir 可能存在竞态条件，如果 create_dir 不是原子的。
    # 但通常 os.makedirs(exist_ok=True) 是线程安全的。
    ensure_dir_fn(record_dir)

    probes_path = f"{record_dir}/probes.jsonl"
    decisions_path = f"{record_dir}/decisions.jsonl"
    reports_path = f"{record_dir}/reports.jsonl"
    errors_path = f"{record_dir}/errors.jsonl"
    dataset_path = f"{record_dir}/dataset.jsonl"

    final_result = None

    for attempt in range(1, max_attempts_per_record + 1):
        try:
            # Step 1: Generate
            probe = generator.generate_probe(
                record, gen_call_model, model_id=generator_model_id, api_config=local_api_config, log_fn=thread_safe_log
            )
            # 写入文件最好加锁，或者假设每个 record_dir 互不冲突
            # 这里假设 record_id 唯一，所以 file path 唯一，无需加锁
            append_jsonl_fn(probes_path, {"record_id": record_id, "attempt": attempt, "probe": probe})
        except Exception as exc:
            append_jsonl_fn(errors_path, {"record_id": record_id, "attempt": attempt, "stage": "generate", "error": str(exc)})
            thread_safe_log("task1_generate_error", {"record_id": record_id, "error": str(exc)})
            continue

        dataset_entry = {
            "record_id": record_id,
            "record": {"dialogue": probe.get("dialogue", [])},
            "memory": probe.get("ground_truth_memories", []),
            "query": probe.get("query", ""),
            "answer": probe.get("ground_truth_memories", []),
            "metadata": probe.get("metadata", {}),
        }
        append_jsonl_fn(dataset_path, dataset_entry)

        try:
            # Step 2: Evaluate
            report = evaluator.evaluate_models(
                probe,
                model_list,
                eval_call_model,
                api_config=local_api_config,
                log_fn=thread_safe_log,
            )
            append_jsonl_fn(reports_path, {"record_id": record_id, "attempt": attempt, "report": report})
        except Exception as exc:
            append_jsonl_fn(errors_path, {"record_id": record_id, "attempt": attempt, "stage": "evaluate", "error": str(exc)})
            thread_safe_log("task1_evaluate_error", {"record_id": record_id, "error": str(exc)})
            continue

        # Step 3: Curate/Score
        score_report = curator.score_report(
            probe,
            report,
            use_llm=use_judge,
            llm_weight=0.5,
            matcher="greedy",
            api_config=local_api_config,
        )
        append_jsonl_fn(decisions_path, {"record_id": record_id, "attempt": attempt, "score": score_report})
        thread_safe_log("task1_curator_score", {"record_id": record_id})

        final_result = {"probe": probe, "report": report, "score": score_report}
        break # 成功则跳出重试循环

    return final_result

def run_task1_pipeline(
    records: Iterable[Dict[str, Any]],
    model_list: List[str],
    gen_call_model,
    eval_call_model,
    use_judge: bool = False,
    judge_call_model=None,
    max_attempts_per_record: int = 5,
    output_dir: str | None = None,
    api_config: Dict[str, Any] | None = None,
    generator_model_id: str | None = None,
    curator_model_id: str | None = None,
    curator_use_llm: bool = False,
    curator_call_model=None,
    max_samples: int | None = None,
    skip_on_rewrite: bool = False,
    max_workers: int = 100,  # 新增：控制并发数量
) -> List[Dict[str, Any]]:
    
    # 初始化工具类
    generator = Task1Generator()
    curator = Task1Curator()
    evaluator = Task1Evaluator()
    results: List[Dict[str, Any]] = []

    if output_dir is None:
        output_dir = make_run_dir("runs", "task1_batch")
    ensure_dir(output_dir)
    
    events_path = f"{output_dir}/events.jsonl"
    
    # 处理 max_samples 切片
    # 将 records 转换为 list 以便切片，如果 records 非常大导致内存溢出，需使用 itertools.islice
    records_list = list(records)
    if max_samples is not None:
        records_list = records_list[:max_samples]

    print(f"Starting pipeline with {max_workers} threads for {len(records_list)} records...")

    # 使用 ThreadPoolExecutor 并行执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_record = {
            executor.submit(
                process_single_record,
                record=record,
                output_dir=output_dir,
                max_attempts_per_record=max_attempts_per_record,
                model_list=model_list,
                gen_call_model=gen_call_model,
                eval_call_model=eval_call_model,
                generator=generator,
                curator=curator,
                evaluator=evaluator,
                api_config=api_config,
                generator_model_id=generator_model_id,
                use_judge=use_judge,
                # 传入上下文中的全局函数
                resolve_session_id_fn=resolve_session_id,
                ensure_dir_fn=ensure_dir,
                append_jsonl_fn=append_jsonl,
                log_event_fn=log_event,
                events_path=events_path
            ): record for record in records_list
        }

        # 获取结果（带有进度条）
        for future in tqdm(as_completed(future_to_record), total=len(records_list), desc="Processing Records"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                # 捕获线程中未处理的异常，防止整个程序崩溃
                record = future_to_record[future]
                print(f"Record processing generated an exception: {exc}")

    return results