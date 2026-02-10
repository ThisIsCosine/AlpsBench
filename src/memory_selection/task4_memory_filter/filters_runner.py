'''
Ability1: Persona extraction and following. --> 20 selected records for each memory category (indirect first than indirect for each categories)
Ability2: General preferences following. --> 20 selected records for each memory category (indirect first than indirect for each category), if this category  does not have enough records, just skip it
Ability2: Interaction preferences following. --> 200 selected records for each memory category （As just 1 category）
Ability3: Real–Virtual Distinguishing. --> 30 selected records for each memory category (indirect first than indirect for each category)
Ability4: Constraint-Grounded Answering. --> 50 selected records for each memory category (indirect first than indirect for each category

Of course, some record will be fail, so write iteration until we have enough records for each category/ability
'''

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import ability1 as ability1_filter
from . import ability2_general as ability2_general_filter
from . import ability2_interaction as ability2_interaction_filter
from . import ability3 as ability3_filter
from . import ability4 as ability4_filter


@dataclass
class RunnerConfig:
    abilities: Optional[List[str]] = None
    per_label: Dict[str, int] = field(default_factory=dict)
    default_target: int = 20
    model_id: Optional[str] = None
    input_root: str = "data/wildchat/memories/selected/task4"
    output_root: str = "data/wildchat/memories/filtered/task4"
    api_config_path: str = "configs/api.json"
    seed: int = 42
    max_passes: int = 5


def run_filters(config: RunnerConfig) -> None:
    """Entry point for running one or more ability filters."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_root = os.path.join(config.output_root, run_id)
    os.makedirs(run_output_root, exist_ok=True)
    run_config = replace(config, output_root=run_output_root)
    _preflight_model_api(run_config)
    abilities = config.abilities or [
        "ability1",
        "ability2_general",
        "ability2_interaction",
        "ability3",
        "ability4",
    ]
    for ability_name in abilities:
        if ability_name == "ability1":
            run_ability1(run_config)
        elif ability_name == "ability2_general":
            run_ability2_general(run_config)
        elif ability_name == "ability2_interaction":
            run_ability2_interaction(run_config)
        elif ability_name == "ability3":
            run_ability3(run_config)
        elif ability_name == "ability4":
            run_ability4(run_config)
        else:
            _raise_not_implemented(ability_name)


def run_ability1(config: RunnerConfig) -> None:
    """Run Ability 1 filtering with per-label sampling and retries."""
    _run_filter_module(ability1_filter, config, "ability1")


def run_ability2_general(config: RunnerConfig) -> None:
    """Run Ability 2 general filtering with per-label sampling and retries."""
    _run_filter_module(ability2_general_filter, config, "ability2_general")


def run_ability2_interaction(config: RunnerConfig) -> None:
    """Run Ability 2 interaction filtering with per-label sampling and retries."""
    _run_filter_module(ability2_interaction_filter, config, "ability2_interaction")


def run_ability3(config: RunnerConfig) -> None:
    """Run Ability 3 filtering with per-label sampling and retries."""
    _run_filter_module(ability3_filter, config, "ability3")


def run_ability4(config: RunnerConfig) -> None:
    """Run Ability 4 filtering with per-label sampling and retries."""
    _run_filter_module(ability4_filter, config, "ability4")


def _run_filter_module(filter_module: Any, config: RunnerConfig, ability_name: str) -> None:
    input_dir = os.path.join(config.input_root, ability_name)
    output_dir = os.path.join(config.output_root, ability_name)
    os.makedirs(output_dir, exist_ok=True)

    target_labels = _get_target_labels(filter_module)

    jsonl_paths = filter_module.iter_jsonl_paths(input_dir)
    if not jsonl_paths:
        raise RuntimeError(f"No jsonl files found under: {input_dir}")

    rng = random.Random(config.seed)
    candidates = _collect_candidates(filter_module, jsonl_paths, target_labels)
    if not candidates:
        raise RuntimeError(f"No candidates found under: {input_dir}")
    if not target_labels:
        target_labels = sorted({candidate["label"] for candidate in candidates})
    per_label_targets = _resolve_label_targets(target_labels, config.per_label, config.default_target)
    candidate_queues = _group_candidates(candidates, rng)

    filter_module.OUTPUT_DIR = output_dir
    chosen_model_id = config.model_id or os.environ.get("TASK4_FILTER_MODEL_ID") or filter_module.DEFAULT_MODEL_ID
    api_config = filter_module._load_api_config(config.api_config_path)
    error_path = os.path.join(output_dir, "errors.log")
    log_path = os.path.join(output_dir, "run.log")

    handles: Dict[str, "filter_module.TextIO"] = {}
    kept_counts = {label: 0 for label in target_labels}
    drop_counts = {label: 0 for label in target_labels}
    error_counts = {label: 0 for label in target_labels}
    index_map = {key: 0 for key in candidate_queues}
    try:
        with open(error_path, "w", encoding="utf-8") as error_handle, open(
            log_path, "w", encoding="utf-8"
        ) as log_handle:
            print(f"[{ability_name}] input_dir={input_dir}", flush=True)
            print(f"[{ability_name}] output_dir={output_dir}", flush=True)
            print(f"[{ability_name}] model_id={chosen_model_id}", flush=True)
            print(f"[{ability_name}] files={len(jsonl_paths)}", flush=True)
            for path in jsonl_paths:
                print(f"[{ability_name}] file={path}", flush=True)
            log_handle.write(f"input_dir={input_dir}\n")
            log_handle.write(f"output_dir={output_dir}\n")
            log_handle.write(f"model_id={chosen_model_id}\n")
            log_handle.write(f"files={len(jsonl_paths)}\n")
            for path in jsonl_paths:
                log_handle.write(f"file\t{path}\n")
            log_handle.write(f"candidates={len(candidates)}\n")
            log_handle.write(f"labels={len(target_labels)}\n")
            for label in sorted(target_labels):
                log_handle.write(
                    f"label_target\t{label}\t{per_label_targets.get(label, 0)}\n"
                )
            _flush_logs(log_handle, error_handle)

            for pass_idx in range(config.max_passes):
                progress = False
                kept_before = sum(kept_counts.values())
                attempted_counts: Dict[str, int] = {}
                for label in sorted(target_labels):
                    target = per_label_targets.get(label, 0)
                    if target <= 0 or kept_counts[label] >= target:
                        continue
                    for mem_type in ("indirect", "direct"):
                        key = (label, mem_type)
                        queue = candidate_queues.get(key, [])
                        while kept_counts[label] < target and index_map.get(key, 0) < len(queue):
                            candidate = queue[index_map[key]]
                            index_map[key] += 1
                            attempted_counts[label] = attempted_counts.get(label, 0) + 1
                            progress = True
                            result = _process_candidate(
                                filter_module,
                                candidate,
                                chosen_model_id,
                                api_config,
                                error_handle,
                                log_handle,
                            )
                            status, payload = result
                            if status == "keep" and payload:
                                output_entry, mem_type = payload
                                handle = filter_module._get_output_handle(label, mem_type, handles)
                                handle.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                                kept_counts[label] += 1
                                handle.flush()
                            elif status == "drop":
                                drop_counts[label] += 1
                            else:
                                error_counts[label] += 1
                        if kept_counts[label] >= target:
                            break
                kept_after = sum(kept_counts.values())
                dropped_total = sum(drop_counts.values())
                error_total = sum(error_counts.values())
                target_total = sum(per_label_targets.values())
                remaining_total = max(target_total - kept_after, 0)
                print(
                    f"[{ability_name}] pass={pass_idx} kept={kept_after} "
                    f"drop={dropped_total} error={error_total} "
                    f"remaining={remaining_total} delta={kept_after - kept_before}"
                , flush=True)
                log_handle.write(
                    f"pass={pass_idx}\tkept={kept_after}\tdrop={dropped_total}"
                    f"\terror={error_total}\tremaining={remaining_total}"
                    f"\tdelta={kept_after - kept_before}\n"
                )
                for label, tried in sorted(attempted_counts.items()):
                    log_handle.write(
                        f"pass_label\t{pass_idx}\t{label}\ttried={tried}"
                        f"\tkept={kept_counts[label]}\tdrop={drop_counts[label]}"
                        f"\terror={error_counts[label]}"
                        f"\tremaining={max(per_label_targets.get(label, 0) - kept_counts[label], 0)}\n"
                    )
                for label in sorted(target_labels):
                    print(
                        f"[{ability_name}] pass={pass_idx} label={label} "
                        f"kept={kept_counts[label]} drop={drop_counts[label]} "
                        f"error={error_counts[label]} "
                        f"remaining={max(per_label_targets.get(label, 0) - kept_counts[label], 0)}"
                    , flush=True)
                if not progress:
                    print(f"[{ability_name}] pass={pass_idx} no_more_candidates", flush=True)
                    log_handle.write(f"pass={pass_idx} no_more_candidates\n")
                    break
                _flush_logs(log_handle, error_handle)
            print(
                f"[{ability_name}] done kept={sum(kept_counts.values())} "
                f"drop={sum(drop_counts.values())} error={sum(error_counts.values())} "
                f"labels={len(target_labels)}"
            , flush=True)
            log_handle.write(
                f"done kept={sum(kept_counts.values())} drop={sum(drop_counts.values())} "
                f"error={sum(error_counts.values())} labels={len(target_labels)}\n"
            )
            for label in sorted(target_labels):
                log_handle.write(
                    f"label_done\t{label}\tkept={kept_counts[label]}"
                    f"\tdrop={drop_counts[label]}\terror={error_counts[label]}"
                    f"\ttarget={per_label_targets.get(label, 0)}\n"
                )
            _flush_logs(log_handle, error_handle)
    finally:
        for handle in handles.values():
            handle.close()


def _collect_candidates(
    filter_module: Any,
    jsonl_paths: Iterable[str],
    target_labels: Optional[Iterable[str]],
) -> List[dict]:
    candidates: List[dict] = []
    labels = set(target_labels or [])
    for path in jsonl_paths:
        for record in filter_module.iter_records(path):
            session_id = filter_module._extract_session_id(record)
            dialogue_turns = filter_module._extract_dialogue_text(record)
            for item in record.get("memory_items") or []:
                label = item.get("label")
                if labels and label not in labels:
                    continue
                mem_type = filter_module._normalize_memory_type(item.get("type"))
                if not mem_type:
                    continue
                candidates.append(
                    {
                        "session_id": session_id,
                        "dialogue_turns": dialogue_turns,
                        "memory_id": str(item.get("memory_id") or ""),
                        "label": label,
                        "mem_type": mem_type,
                        "memory_value": str(item.get("value") or ""),
                        "evidence_text": filter_module._extract_evidence_text(item),
                    }
                )
    return candidates


def _group_candidates(
    candidates: List[dict],
    rng: random.Random,
) -> Dict[Tuple[str, str], List[dict]]:
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for candidate in candidates:
        key = (candidate["label"], candidate["mem_type"])
        grouped.setdefault(key, []).append(candidate)
    for queue in grouped.values():
        rng.shuffle(queue)
    return grouped


def _resolve_label_targets(
    target_labels: Iterable[str],
    per_label: Dict[str, int],
    default_target: int,
) -> Dict[str, int]:
    targets = {label: default_target for label in target_labels}
    for label, value in per_label.items():
        targets[label] = value
    return targets


def _get_target_labels(filter_module: Any) -> Optional[List[str]]:
    parse_fn = getattr(filter_module, "parse_labels_from_docstring", None)
    if not callable(parse_fn):
        return None
    labels = list(parse_fn() or [])
    return labels or None


def _preflight_model_api(config: RunnerConfig) -> None:
    model_id = (
        config.model_id
        or os.environ.get("TASK4_FILTER_MODEL_ID")
        or ability1_filter.DEFAULT_MODEL_ID
    )
    try:
        api_config = ability1_filter._load_api_config(config.api_config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight] failed to load api config: {exc}", flush=True)
        return
    try:
        _ = ability1_filter.call_model_api(
            model_id,
            "M_filter",
            "You are a helpful assistant.",
            "ping",
            api_config=api_config,
        )
        print(f"[preflight] model_id={model_id} ok", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight] model_id={model_id} failed: {exc}", flush=True)


def _flush_logs(log_handle, error_handle) -> None:
    try:
        log_handle.flush()
    except Exception:
        pass
    try:
        error_handle.flush()
    except Exception:
        pass


def _process_candidate(
    filter_module: Any,
    candidate: dict,
    model_id: str,
    api_config: dict,
    error_handle,
    log_handle,
) -> Tuple[str, Optional[Tuple[dict, str]]]:
    prompt = filter_module._build_prompt(
        session_id=candidate["session_id"],
        memory_id=candidate["memory_id"],
        label=candidate["label"],
        memory_value=candidate["memory_value"],
        evidence_text=candidate["evidence_text"],
        dialogue_turns=candidate["dialogue_turns"],
    )
    try:
        raw = filter_module.call_model_api(
            model_id,
            "M_filter",
            "You are a helpful assistant.",
            prompt,
            api_config=api_config,
        )
    except Exception as exc:  # noqa: BLE001
        error_handle.write(
            f"{candidate['session_id']}\t{candidate['memory_id']}\tapi_error\t{exc}\n"
        )
        return "error_api", None

    parsed = filter_module._parse_model_output(raw or "")
    if parsed is None:
        error_handle.write(
            f"{candidate['session_id']}\t{candidate['memory_id']}\tinvalid_output\t{(raw or '').strip()}\n"
        )
        return "error_invalid", None

    if parsed.get("decision") == "DROP":
        return "drop", None

    queries = parsed.get("queries") or []
    if not isinstance(queries, list) or not queries:
        error_handle.write(
            f"{candidate['session_id']}\t{candidate['memory_id']}\tmissing_queries\t{(raw or '').strip()}\n"
        )
        return "error_missing_queries", None

    output_entry = {
        "session_id": parsed.get("session_id") or candidate["session_id"],
        "memory_id": parsed.get("memory_id") or candidate["memory_id"],
        "label": candidate["label"],
        "type": candidate["mem_type"],
        "queries": queries,
    }
    log_handle.write(
        "keep\t"
        f"{candidate['session_id']}\t{candidate['memory_id']}\t"
        f"{candidate['label']}\t{candidate['mem_type']}\n"
    )
    return "keep", (output_entry, candidate["mem_type"])


def _raise_not_implemented(ability_name: str) -> None:
    raise NotImplementedError(f"{ability_name} is not implemented in filters_runner yet.")