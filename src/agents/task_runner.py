"""
Task runner to orchestrate task pipelines.
Runs a single selected task per call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import importlib
import json
import os
import sys

try:
    from .task1_memory_extraction import run_task1_pipeline
    from .task2_memory_update import run_task2_pipeline
    from .task3_memory_retrieval import run_task3_pipeline
    from .shared import iter_jsonl, list_jsonl_files, make_run_dir
except ImportError:  # Allow direct script execution.
    current_dir = os.path.dirname(__file__)
    src_dir = os.path.dirname(current_dir)
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    from agents.task1_memory_extraction import run_task1_pipeline
    from agents.task2_memory_update import run_task2_pipeline
    from agents.task3_memory_retrieval import run_task3_pipeline
    from agents.shared import iter_jsonl, list_jsonl_files, make_run_dir


TASK_PIPELINES = {
    "task1_memory_extraction": run_task1_pipeline,
    "task2_memory_update": run_task2_pipeline,
    "task3_memory_retrieval": run_task3_pipeline,
}

TASK4_ABILITIES = {
    "ability1",
    "ability2_general",
    "ability2_interaction",
    "ability3",
    "ability4",
    "ability5",
    "ability6",
}


def _resolve_task4_ability(task4_ability: Optional[str], api_config: Optional[Dict[str, Any]]) -> str:
    ability = (task4_ability or "").strip()
    if not ability and api_config:
        ability = str(api_config.get("task4_ability", "")).strip()
    if not ability:
        ability = os.environ.get("TASK4_ABILITY", "").strip()
    if not ability:
        raise ValueError("task4_ability_required (set param, api_config['task4_ability'], or TASK4_ABILITY)")
    if ability not in TASK4_ABILITIES:
        raise ValueError(f"Unknown task4 ability: {ability}. Choose from {sorted(TASK4_ABILITIES)}")
    return ability


def _load_task4_pipeline(task4_ability: str):
    module_root = __package__ or "agents"
    module_path = f"{module_root}.task4_memory_grounded_qa.{task4_ability}.curator"
    module = importlib.import_module(module_path)
    pipeline = getattr(module, "run_task4_pipeline", None)
    if pipeline is None:
        raise ImportError(f"run_task4_pipeline not found in {module_path}")
    return pipeline


def _resolve_repo_path(rel_path: str) -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.normpath(os.path.join(repo_root, rel_path))


def _iter_jsonl_paths(target_path: str) -> List[str]:
    if os.path.isfile(target_path):
        return [target_path]
    if os.path.isdir(target_path):
        return list_jsonl_files(target_path)
    raise FileNotFoundError(f"jsonl_dir not found: {target_path}")


def _build_dialogue_from_record(record: Dict[str, Any]) -> List[Dict[str, str]]:
    sessions = record.get("sessions") or []
    if not sessions:
        return []
    turns = sessions[0].get("turns") or []
    return [{"role": t.get("role", ""), "text": t.get("text", "")} for t in turns]


def run_task_pipeline(
    task_name: str,
    inputs: List[Dict[str, Any]],
    model_list: List[str],
    gen_call_model,
    eval_call_model,
    use_judge: bool = False,
    judge_call_model=None,
    max_attempts: int = 5,
    output_dir: Optional[str] = None,
    api_config: Optional[Dict[str, Any]] = None,
    task4_ability: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if task_name not in TASK_PIPELINES:
        if task_name != "task4_memory_grounded_qa":
            raise ValueError(f"Unknown task_name: {task_name}. Choose one task per run.")

    if task_name == "task1_memory_extraction":
        return TASK_PIPELINES[task_name](
            inputs,
            model_list,
            gen_call_model,
            eval_call_model,
            use_judge=use_judge,
            judge_call_model=judge_call_model,
            max_attempts_per_record=max_attempts,
            output_dir=output_dir or make_run_dir("runs", "task1_memory_extraction"),
            api_config=api_config,
        )

    if task_name == "task4_memory_grounded_qa":
        ability = _resolve_task4_ability(task4_ability, api_config)
        pipeline = _load_task4_pipeline(ability)
        return pipeline(
            inputs,
            model_list,
            gen_call_model,
            eval_call_model,
            use_judge=use_judge,
            judge_call_model=judge_call_model,
            max_attempts_per_seed=max_attempts,
            output_dir=output_dir or make_run_dir("runs", f"task4_{ability}"),
            api_config=api_config,
        )

    return TASK_PIPELINES[task_name](
        inputs,
        model_list,
        gen_call_model,
        eval_call_model,
        use_judge=use_judge,
        judge_call_model=judge_call_model,
        max_attempts_per_seed=max_attempts,
        output_dir=output_dir or make_run_dir("runs", task_name),
        api_config=api_config,
    )


def run_selected_task(
    task_name: str,
    inputs: List[Dict[str, Any]],
    model_list: List[str],
    gen_call_model,
    eval_call_model,
    use_judge: bool = False,
    judge_call_model=None,
    max_attempts: int = 5,
    output_dir: Optional[str] = None,
    api_config: Optional[Dict[str, Any]] = None,
    task4_ability: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Explicit single-task entry point to avoid accidental multi-task runs.
    """
    return run_task_pipeline(
        task_name=task_name,
        inputs=inputs,
        model_list=model_list,
        gen_call_model=gen_call_model,
        eval_call_model=eval_call_model,
        use_judge=use_judge,
        judge_call_model=judge_call_model,
        max_attempts=max_attempts,
        output_dir=output_dir,
        api_config=api_config,
        task4_ability=task4_ability,
    )


def run_task1_from_jsonl_dir(
    jsonl_dir: str,
    max_samples: int,
    model_list: List[str],
    gen_call_model,
    eval_call_model,
    use_judge: bool = False,
    judge_call_model=None,
    max_attempts_per_record: int = 5,
    output_dir: Optional[str] = None,
    api_config: Optional[Dict[str, Any]] = None,
    api_config_path: str = "configs/api.json",
    generator_model_id: Optional[str] = None,
    curator_model_id: Optional[str] = None,
    curator_use_llm: bool = False,
    curator_call_model=None,
) -> List[Dict[str, Any]]:
    if api_config is None:
        with open(api_config_path, "r", encoding="utf-8") as handle:
            api_config = json.load(handle)
    else:
        api_config = dict(api_config)
    api_config["memory_gate_model_id"] = "gpt-5.2"
    if output_dir is None:
        output_dir = make_run_dir("runs", "task1_batch")

    def record_stream():
        # Use the provided jsonl_dir as the base for relative paths.
        # This avoids path traversal (e.g., relpaths starting with "..") when users pass custom data dirs.
        base_dir = os.path.abspath(jsonl_dir)
        if os.path.isfile(base_dir):
            base_dir = os.path.dirname(base_dir)
        for path in _iter_jsonl_paths(jsonl_dir):
            for record in iter_jsonl(path):
                record["_source_path"] = path
                try:
                    rel = os.path.relpath(path, base_dir)
                    # Defensive: never allow ".." to escape output_dir when used for output subfolders.
                    if rel.startswith("..") or os.path.isabs(rel):
                        rel = os.path.basename(path)
                    record["_source_rel"] = rel
                except ValueError:
                    record["_source_rel"] = os.path.basename(path)
                yield record

    return run_task1_pipeline(
        records=record_stream(),
        model_list=model_list,
        gen_call_model=gen_call_model,
        eval_call_model=eval_call_model,
        use_judge=use_judge,
        judge_call_model=judge_call_model,
        max_attempts_per_record=max_attempts_per_record,
        output_dir=output_dir,
        api_config=api_config,
        generator_model_id=generator_model_id,
        curator_model_id=curator_model_id,
        curator_use_llm=curator_use_llm,
        curator_call_model=curator_call_model,
        max_samples=max_samples,
        skip_on_rewrite=True,
    )


def run_task2_from_jsonl_dir(
    jsonl_dir: str,
    model_list: List[str],
    gen_call_model,
    eval_call_model,
    use_judge: bool = False,
    judge_call_model=None,
    max_attempts_per_seed: int = 5,
    output_dir: Optional[str] = None,
    api_config: Optional[Dict[str, Any]] = None,
    api_config_path: str = "configs/api.json",
    generator_model_id: Optional[str] = None,
    curator_model_id: Optional[str] = None,
    curator_use_llm: bool = False,
    curator_call_model=None,
    max_samples: Optional[int] = None,
    skip_on_rewrite: bool = False,
) -> List[Dict[str, Any]]:
    if api_config is None:
        with open(api_config_path, "r", encoding="utf-8") as handle:
            api_config = json.load(handle)
    if output_dir is None:
        output_dir = make_run_dir("runs", "task2_batch")

    # # Resume logic: Load existing IDs from dataset.jsonl
    # existing_ids = set()
    # dataset_path = "E:\\USTC\\Implicit Memory\\implicit-memory-benchmark\\implicit-memory-benchmark\\runs\\task2_batch\\20260201T080255Z\\dataset.jsonl"
    # if os.path.exists(dataset_path):
    #     print(f"Resuming Task 2: Checking existing records in {dataset_path}...")
    #     for line in iter_jsonl(dataset_path):
    #         # Pipeline saves 'record_id' in dataset.jsonl
    #         rid = line.get("record_id")
    #         if rid is not None:
    #             existing_ids.add(str(rid))
    #     print(f"Found {len(existing_ids)} completed records. These will be skipped.")

    # def seed_stream():
    #     for path in _iter_jsonl_paths(jsonl_dir):
    #         for record in iter_jsonl(path):
    #             # Resolve ID to check against existing_ids (matching logic in pipeline.py)
    #             current_id = record.get("session_id") or record.get("line_index") or record.get("seed_id")
                
    #             # If ID exists, skip yielding
    #             if current_id is not None and str(current_id) in existing_ids:
    #                 continue

    #             # Preferred seed format: benchmark records with dialogue + memory_items + selected_memory_id
    #             if record.get("sessions") and record.get("memory_items") and record.get("selected_memory_id"):
    #                 yield record
    #                 continue
    #             if record.get("past_memory"):
    #                 yield record
    #                 continue
    #             past_memory = record.get("memory_items") or []
    #             seed = {
    #                 "past_memory": past_memory,
    #                 "metadata": {"session_id": record.get("session_id"), "dialog_id": record.get("line_index")},
    #             }
    #             yield seed
    print(jsonl_dir)
    def seed_stream():
        for path in _iter_jsonl_paths(jsonl_dir):
            for record in iter_jsonl(path):
                # Preferred seed format: benchmark records with dialogue + memory_items + selected_memory_id
                if record.get("sessions") and record.get("memory_items") and record.get("selected_memory_id"):
                    yield record
                    continue
                if record.get("past_memory"):
                    yield record
                    continue
                past_memory = record.get("memory_items") or []
                seed = {
                    "past_memory": past_memory,
                    "metadata": {"session_id": record.get("session_id"), "dialog_id": record.get("line_index")},
                }
                yield seed

    return run_task2_pipeline(
        seeds=seed_stream(),
        model_list=model_list,
        gen_call_model=gen_call_model,
        eval_call_model=eval_call_model,
        use_judge=use_judge,
        judge_call_model=judge_call_model,
        max_attempts_per_seed=max_attempts_per_seed,
        output_dir=output_dir,
        api_config=api_config,
        generator_model_id=generator_model_id,
        curator_model_id=curator_model_id,
        curator_use_llm=curator_use_llm,
        curator_call_model=curator_call_model,
        max_samples=max_samples,
        skip_on_rewrite=skip_on_rewrite,
    )


def run_task3_from_jsonl_dir(
    jsonl_dir: str,
    model_list: List[str],
    gen_call_model,
    eval_call_model,
    use_judge: bool = True,
    judge_call_model=None,
    max_attempts_per_seed: int = 10,
    output_dir: Optional[str] = None,
    api_config: Optional[Dict[str, Any]] = None,
    api_config_path: str = "configs/api.json",
    generator_model_id: Optional[str] = None,
    curator_model_id: Optional[str] = None,
    curator_use_llm: bool = False,
    curator_call_model=None,
    max_samples: Optional[int] = None,
    skip_on_rewrite: bool = False,
    distract_n: int = 10,
) -> List[Dict[str, Any]]:
    if api_config is None:
        with open(api_config_path, "r", encoding="utf-8") as handle:
            api_config = json.load(handle)
    
    other_memories = []
    # Always load memories from other categories to use as distractors
    # Assuming benchmark/dev_with_selected_memory_id structure
    base_dir = os.path.dirname(jsonl_dir.rstrip("/\\"))
    if os.path.basename(base_dir).startswith("ability"):
        # We are likely in a sub-category folder, go up one more level to reach ability root
        ability_root = base_dir
        parent_root = os.path.dirname(ability_root)
    else:
        ability_root = jsonl_dir
        parent_root = os.path.dirname(ability_root)

    # Collect from all jsonl files NOT in the current ability_root
    for root, dirs, files in os.walk(parent_root):
        # Correctly skip the current ability directory and its subdirectories
        normalized_root = os.path.normpath(root)
        normalized_ability_root = os.path.normpath(ability_root)
        if normalized_root.startswith(normalized_ability_root):
            continue
            
        for file in files:
            if file.endswith(".jsonl") and not file.startswith("_"):
                full_path = os.path.join(root, file)
                for record in iter_jsonl(full_path):
                    # Distractors can come from 'memories' or 'memory_items'
                    mems = record.get("memories") or record.get("memory_items") or []
                    other_memories.extend(mems)
                    if len(other_memories) > 1000: # Cap it to avoid huge memory usage
                        break
        if len(other_memories) > 1000:
            break

    if output_dir is None:
        output_dir = make_run_dir("runs", "task3_batch")

    def seed_stream():
        for path in _iter_jsonl_paths(jsonl_dir):
            for record in iter_jsonl(path):
                # Check for Task 4 memory retrieval format (with memory_items and selected_memory_id)
                if record.get("memory_items") and record.get("selected_memory_id"):
                    memories = record.get("memory_items")
                    # If dialogue is missing, build it from sessions
                    dialogue = record.get("dialogue")
                    if not dialogue:
                        dialogue = _build_dialogue_from_record(record)
                    
                    seed = {
                        "dialogue": dialogue,
                        "memories": memories,
                        "selected_memory_id": record.get("selected_memory_id"),
                        "metadata": {
                            "session_id": record.get("session_id") or (record.get("sessions") or [{}])[0].get("session_id"),
                            "dialog_id": record.get("line_index") or record.get("dialog_id")
                        },
                    }
                    if record.get("selected_memory"):
                        seed["selected_memory"] = record.get("selected_memory")
                    yield seed
                    continue

                if record.get("memories"):
                    memories = record.get("memories") or []
                    seed = {
                        "dialogue": _build_dialogue_from_record(record),
                        "memories": memories,
                        "metadata": {"session_id": record.get("session_id"), "dialog_id": record.get("line_index")},
                    }
                    if record.get("selected_memory_id"):
                        seed["selected_memory_id"] = record.get("selected_memory_id")
                    if record.get("selected_memory"):
                        seed["selected_memory"] = record.get("selected_memory")
                    yield seed
                    continue

    return run_task3_pipeline(
        seeds=seed_stream(),
        model_list=model_list,
        gen_call_model=gen_call_model,
        eval_call_model=eval_call_model,
        use_judge=use_judge,
        judge_call_model=judge_call_model,
        max_attempts_per_seed=max_attempts_per_seed,
        output_dir=output_dir,
        api_config=api_config,
        generator_model_id=generator_model_id,
        curator_model_id=curator_model_id,
        curator_use_llm=curator_use_llm,
        curator_call_model=curator_call_model,
        max_samples=max_samples,
        skip_on_rewrite=skip_on_rewrite,
        other_memories=other_memories,
        distract_n=distract_n,
    )


def run_task4_from_jsonl_dir(
    jsonl_dir: str,
    model_list: List[str],
    gen_call_model,
    eval_call_model,
    use_judge: bool = False,
    judge_call_model=None,
    max_attempts_per_seed: int = 5,
    output_dir: Optional[str] = None,
    api_config: Optional[Dict[str, Any]] = None,
    api_config_path: str = "configs/api.json",
    generator_model_id: Optional[str] = None,
    curator_model_id: Optional[str] = None,
    curator_use_llm: bool = False,
    curator_call_model=None,
    max_samples: Optional[int] = None,
    skip_on_rewrite: bool = False,
    task4_ability: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if api_config is None:
        with open(api_config_path, "r", encoding="utf-8") as handle:
            api_config = json.load(handle)
    ability = _resolve_task4_ability(task4_ability, api_config)
    if output_dir is None:
        output_dir = make_run_dir("runs", f"task4_{ability}")

    def seed_stream():
        for path in _iter_jsonl_paths(jsonl_dir):
            for record in iter_jsonl(path):
                if record.get("dialogue"):
                    yield record
                    continue
                seed = {
                    "dialogue": _build_dialogue_from_record(record),
                    "memories": record.get("memory_items") or [],
                    "metadata": {"session_id": record.get("session_id"), "dialog_id": record.get("line_index")},
                }
                yield seed

    pipeline = _load_task4_pipeline(ability)
    return pipeline(
        seeds=seed_stream(),
        model_list=model_list,
        gen_call_model=gen_call_model,
        eval_call_model=eval_call_model,
        use_judge=use_judge,
        judge_call_model=judge_call_model,
        max_attempts_per_seed=max_attempts_per_seed,
        output_dir=output_dir,
        api_config=api_config,
        generator_model_id=generator_model_id,
        curator_model_id=curator_model_id,
        curator_use_llm=curator_use_llm,
        curator_call_model=curator_call_model,
        max_samples=max_samples,
        skip_on_rewrite=skip_on_rewrite,
    )


def run_task1_quickstart() -> List[Dict[str, Any]]:
    jsonl_dir = os.environ.get("TASK1_JSONL_PATH") or _resolve_repo_path("data/human_annotation_with_selected_memory_id/task4_final_human_annotation_with_selected_memory_id")
    output_dir = None
    max_samples = 10000
    max_attempts_per_record = 5
    skip_on_rewrite = True

    generator_model_id = ""
    eval_model_list = ["gpt-4.1-mini"]
    curator_model_id = ""
    curator_use_llm = False
    use_judge = False

    if not os.path.exists(jsonl_dir):
        raise FileNotFoundError(f"Set jsonl_dir to a valid task1 records path: {jsonl_dir}")

    return run_task1_from_jsonl_dir(
        jsonl_dir=jsonl_dir,
        max_samples=max_samples,
        model_list=eval_model_list,
        gen_call_model=None,
        eval_call_model=None,
        use_judge=use_judge,
        judge_call_model=None,
        max_attempts_per_record=max_attempts_per_record,
        output_dir=output_dir,
        api_config=None,
        api_config_path="configs/api.json",
        generator_model_id=generator_model_id,
        curator_model_id=curator_model_id,
        curator_use_llm=curator_use_llm,
        curator_call_model=None,
    )


def run_task2_quickstart() -> List[Dict[str, Any]]:
    # Default to benchmark records that include dialogue + memory_items + selected_memory_id
    jsonl_dir = os.environ.get("TASK2_JSONL_PATH") or _resolve_repo_path("benchmark/dev_with_selected_memory_id")
    output_dir = None
    max_samples = 10
    max_attempts_per_seed = 10
    skip_on_rewrite = True

    generator_model_id = "gpt-5.2"
    eval_model_list = ["o3-mini", "gpt-4.1-mini", "gpt-5.2"]
    curator_model_id = (os.environ.get("TASK3_CURATOR_MODEL_ID") or "").strip() or "gpt-4o-mini"
    curator_use_llm = False
    use_judge = False

    if not os.path.exists(jsonl_dir):
        raise FileNotFoundError(f"Set jsonl_dir to a valid task2 seed path: {jsonl_dir}")

    return run_task2_from_jsonl_dir(
        jsonl_dir=jsonl_dir,
        model_list=eval_model_list,
        gen_call_model=None,
        eval_call_model=None,
        use_judge=use_judge,
        judge_call_model=None,
        max_attempts_per_seed=max_attempts_per_seed,
        output_dir=output_dir,
        api_config=None,
        api_config_path="configs/api.json",
        generator_model_id=generator_model_id,
        curator_model_id=curator_model_id,
        curator_use_llm=curator_use_llm,
        curator_call_model=None,
        max_samples=max_samples,
        skip_on_rewrite=skip_on_rewrite,
    )


def run_task3_quickstart() -> List[Dict[str, Any]]:
    jsonl_dir = os.environ.get("TASK3_JSONL_PATH") or _resolve_repo_path("benchmark/dev_with_selected_memory_id")
    output_dir = None
    max_samples = int(os.environ.get("MAX_SAMPLES", "5"))
    max_attempts_per_seed = 5
    # Allow multiple attempts when the curator requests a rewrite.
    skip_on_rewrite = False
    distract_n = int(os.environ.get("TASK3_DISTRACTORS", "10"))

    generator_model_id = "gpt-5.2"
    eval_model_list = ["o3-mini"]
    curator_model_id = "gpt-4o-mini"
    curator_use_llm = True
    use_judge = True

    if not os.path.exists(jsonl_dir):
        raise FileNotFoundError(f"Set jsonl_dir to a valid task3 seed path: {jsonl_dir}")

    return run_task3_from_jsonl_dir(
        jsonl_dir=jsonl_dir,
        model_list=eval_model_list,
        gen_call_model=None,
        eval_call_model=None,
        use_judge=use_judge,
        judge_call_model=None,
        max_attempts_per_seed=max_attempts_per_seed,
        output_dir=output_dir,
        api_config=None,
        api_config_path="configs/api.json",
        generator_model_id=generator_model_id,
        curator_model_id=curator_model_id,
        curator_use_llm=curator_use_llm,
        curator_call_model=None,
        max_samples=max_samples,
        skip_on_rewrite=skip_on_rewrite,
        distract_n=distract_n,
    )


def run_task4_quickstart() -> List[Dict[str, Any]]:
    jsonl_dir = "data\wildchat\memories\selected\\task4_final_with_selected_memory_id"
    output_dir = None
    max_samples = 10
    max_attempts_per_seed = 5
    skip_on_rewrite = True

    generator_model_id = "gpt-4.1-mini"
    eval_model_list = ["gpt-4.1-mini"]
    curator_model_id = None
    curator_use_llm = False
    use_judge = False
    task4_ability = os.environ.get("TASK4_ABILITY", "").strip() or "ability1"

    if not os.path.isdir(jsonl_dir):
        raise FileNotFoundError(f"Set jsonl_dir to a valid task4 seed dir: {jsonl_dir}")

    return run_task4_from_jsonl_dir(
        jsonl_dir=jsonl_dir,
        model_list=eval_model_list,
        gen_call_model=None,
        eval_call_model=None,
        use_judge=use_judge,
        judge_call_model=None,
        max_attempts_per_seed=max_attempts_per_seed,
        output_dir=output_dir,
        api_config=None,
        api_config_path="configs/api.json",
        generator_model_id=generator_model_id,
        curator_model_id=curator_model_id,
        curator_use_llm=curator_use_llm,
        curator_call_model=None,
        max_samples=max_samples,
        skip_on_rewrite=skip_on_rewrite,
        task4_ability=task4_ability,
    )


TASK_QUICKSTARTS = {
    "task1": run_task1_quickstart,
    "task2": run_task2_quickstart,
    "task3": run_task3_quickstart,
    "task4": run_task4_quickstart,
}

TASK_TO_RUN = "task2"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task runner to orchestrate task pipelines.")
    parser.add_argument("task", nargs="?", default=TASK_TO_RUN, help=f"Task to run. Options: {sorted(TASK_QUICKSTARTS)}")
    parser.add_argument("--samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--distractors", type=int, help="Number of distractors for Task 3")
    
    args = parser.parse_args()
    
    selected = os.environ.get("TASK_QUICKSTART") or args.task
    if selected not in TASK_QUICKSTARTS:
        raise ValueError(f"Unknown TASK_QUICKSTART: {selected}. Options: {sorted(TASK_QUICKSTART)}")
    
    if args.samples:
        os.environ["MAX_SAMPLES"] = str(args.samples)
    
    if args.distractors is not None:
        os.environ["TASK3_DISTRACTORS"] = str(args.distractors)

    TASK_QUICKSTARTS[selected]()
