from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, Mapping


SCORING_POLICY = {
    "examples": {"reference_output_public": True, "local_scoring_expected": True},
    "dev": {"reference_output_public": True, "local_scoring_expected": True},
    "validation": {"reference_output_public": True, "local_scoring_expected": True},
    "test": {"reference_output_public": False, "local_scoring_expected": False},
}

_WS_RE = re.compile(r"\s+")


def supports_local_scoring(split: str) -> bool:
    return bool(SCORING_POLICY.get(split, {}).get("local_scoring_expected"))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _WS_RE.sub(" ", text.strip().lower())


def _round(value: float) -> float:
    return round(float(value), 6)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return _round(sum(values) / len(values))


def _extract_payload(prediction: Mapping[str, Any]) -> Any:
    if "prediction" in prediction:
        return prediction["prediction"]
    return prediction


def _coerce_memory_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _canonical_memory_item(item: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        normalize_text(item.get("label")),
        normalize_text(item.get("value")),
        normalize_text(item.get("type")),
        normalize_text(item.get("preference_attitude")),
        normalize_text(item.get("time_scope")),
        normalize_text(item.get("emotion")),
    )


def _multiset_f1(pred_items: list[dict[str, Any]], ref_items: list[dict[str, Any]]) -> Dict[str, Any]:
    pred_counter = Counter(_canonical_memory_item(item) for item in pred_items)
    ref_counter = Counter(_canonical_memory_item(item) for item in ref_items)
    overlap = pred_counter & ref_counter
    true_positive = sum(overlap.values())
    pred_total = sum(pred_counter.values())
    ref_total = sum(ref_counter.values())

    if pred_total == 0 and ref_total == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = true_positive / pred_total if pred_total else 0.0
        recall = true_positive / ref_total if ref_total else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "exact_match": 1.0 if pred_counter == ref_counter else 0.0,
        "pred_count": pred_total,
        "reference_count": ref_total,
    }


def _task1_prediction_items(prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _extract_payload(prediction)
    if isinstance(payload, list):
        return _coerce_memory_items(payload)
    if isinstance(payload, Mapping):
        return _coerce_memory_items(payload.get("memory_items"))
    return []


def _task2_prediction_items(prediction: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _extract_payload(prediction)
    if isinstance(payload, list):
        return _coerce_memory_items(payload)
    if isinstance(payload, Mapping):
        return _coerce_memory_items(payload.get("answer") or payload.get("memory_items"))
    return []


def _prediction_selected_memory_id(prediction: Mapping[str, Any]) -> str:
    payload = _extract_payload(prediction)
    if not isinstance(payload, Mapping):
        return ""
    direct = payload.get("selected_memory_id") or payload.get("memory_key")
    if direct:
        return normalize_text(direct)
    selected_memory = payload.get("selected_memory")
    if isinstance(selected_memory, Mapping):
        return normalize_text(selected_memory.get("memory_id"))
    return ""


def _prediction_selected_memory_value(prediction: Mapping[str, Any]) -> str:
    payload = _extract_payload(prediction)
    if not isinstance(payload, Mapping):
        return ""
    selected_memory = payload.get("selected_memory")
    if isinstance(selected_memory, Mapping):
        return normalize_text(selected_memory.get("value"))
    direct = payload.get("selected_memory_value")
    if direct:
        return normalize_text(direct)
    return ""


def _prediction_response_text(prediction: Mapping[str, Any]) -> str:
    payload = _extract_payload(prediction)
    if not isinstance(payload, Mapping):
        return ""
    return normalize_text(payload.get("answer") or payload.get("response"))


def _prediction_used_memory_fact(prediction: Mapping[str, Any]) -> str:
    payload = _extract_payload(prediction)
    if not isinstance(payload, Mapping):
        return ""
    return normalize_text(payload.get("used_memory_fact"))


def score_prediction_row(
    *,
    task: str,
    prediction: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    if task == "task1":
        pred_items = _task1_prediction_items(prediction)
        ref_items = _coerce_memory_items(reference.get("gold", {}).get("memory_items"))
        metrics = _multiset_f1(pred_items, ref_items)
        metrics["main_score"] = metrics["f1"]
        return metrics

    if task == "task2":
        pred_items = _task2_prediction_items(prediction)
        ref_items = _coerce_memory_items(reference.get("gold", {}).get("answer"))
        metrics = _multiset_f1(pred_items, ref_items)
        metrics["main_score"] = metrics["f1"]
        return metrics

    if task == "task3":
        ref_memory = reference.get("gold", {}).get("selected_memory") or {}
        ref_id = normalize_text(reference.get("gold", {}).get("selected_memory_id"))
        ref_value = normalize_text(ref_memory.get("value"))
        pred_id = _prediction_selected_memory_id(prediction)
        pred_value = _prediction_selected_memory_value(prediction)
        id_match = 1.0 if ref_id and pred_id == ref_id else 0.0
        value_match = 1.0 if ref_value and pred_value == ref_value else 0.0
        accuracy = 1.0 if id_match or value_match else 0.0
        return {
            "accuracy": accuracy,
            "memory_id_match": id_match,
            "memory_value_match": value_match,
            "main_score": accuracy,
        }

    if task == "task4":
        ref_memory = reference.get("gold", {}).get("selected_memory") or {}
        ref_value = normalize_text(ref_memory.get("value"))
        pred_fact = _prediction_used_memory_fact(prediction)
        response_present = 1.0 if _prediction_response_text(prediction) else 0.0
        fact_match = 0.0
        if ref_value and pred_fact:
            if pred_fact == ref_value or pred_fact in ref_value or ref_value in pred_fact:
                fact_match = 1.0
        grounding_accuracy = fact_match
        return {
            "grounding_accuracy": grounding_accuracy,
            "used_memory_fact_match": fact_match,
            "response_present": response_present,
            "main_score": grounding_accuracy,
        }

    raise ValueError(f"Unsupported task for scoring: {task}")


def summarize_score_rows(*, task: str, score_rows: list[Mapping[str, Any]]) -> Dict[str, Any]:
    if not score_rows:
        return {
            "status": "ok",
            "task": task,
            "num_rows": 0,
            "main_metric": "main_score",
            "main_score": 0.0,
        }

    if task in {"task1", "task2"}:
        return {
            "status": "ok",
            "task": task,
            "num_rows": len(score_rows),
            "main_metric": "mean_f1",
            "mean_precision": _mean(row["precision"] for row in score_rows),
            "mean_recall": _mean(row["recall"] for row in score_rows),
            "mean_f1": _mean(row["f1"] for row in score_rows),
            "exact_match_rate": _mean(row["exact_match"] for row in score_rows),
            "main_score": _mean(row["f1"] for row in score_rows),
        }

    if task == "task3":
        return {
            "status": "ok",
            "task": task,
            "num_rows": len(score_rows),
            "main_metric": "accuracy",
            "accuracy": _mean(row["accuracy"] for row in score_rows),
            "memory_id_match_rate": _mean(row["memory_id_match"] for row in score_rows),
            "memory_value_match_rate": _mean(row["memory_value_match"] for row in score_rows),
            "main_score": _mean(row["accuracy"] for row in score_rows),
        }

    if task == "task4":
        return {
            "status": "ok",
            "task": task,
            "num_rows": len(score_rows),
            "main_metric": "grounding_accuracy",
            "grounding_accuracy": _mean(row["grounding_accuracy"] for row in score_rows),
            "used_memory_fact_match_rate": _mean(row["used_memory_fact_match"] for row in score_rows),
            "response_present_rate": _mean(row["response_present"] for row in score_rows),
            "main_score": _mean(row["grounding_accuracy"] for row in score_rows),
        }

    raise ValueError(f"Unsupported task for score summary: {task}")


def metric_surface() -> Dict[str, Any]:
    return {
        "status": "ready",
        "task_metrics": {
            "task1": "set-based memory-item precision/recall/f1 using canonicalized memory fields",
            "task2": "set-based updated-memory precision/recall/f1 using canonicalized memory fields",
            "task3_d100": "selected-memory accuracy via selected_memory_id match",
            "task3_d300": "selected-memory accuracy via selected_memory_id match",
            "task3_d500": "selected-memory accuracy via selected_memory_id match",
            "task3_d700": "selected-memory accuracy via selected_memory_id match",
            "task3_d1000": "selected-memory accuracy via selected_memory_id match",
            "task4_ability1": "grounding proxy via used_memory_fact match against the gold selected memory",
            "task4_ability2": "grounding proxy via used_memory_fact match against the gold selected memory",
            "task4_ability3": "grounding proxy via used_memory_fact match against the gold selected memory",
            "task4_ability4": "grounding proxy via used_memory_fact match against the gold selected memory",
            "task4_ability5": "grounding proxy via used_memory_fact match against the gold selected memory",
        },
        "prediction_contract": {
            "task1": {"required": ["benchmark_id", "memory_items"]},
            "task2": {"required": ["benchmark_id", "memory_items"]},
            "task3_d100": {"required": ["benchmark_id", "answer", "reason", "selected_memory_id"]},
            "task3_d300": {"required": ["benchmark_id", "answer", "reason", "selected_memory_id"]},
            "task3_d500": {"required": ["benchmark_id", "answer", "reason", "selected_memory_id"]},
            "task3_d700": {"required": ["benchmark_id", "answer", "reason", "selected_memory_id"]},
            "task3_d1000": {"required": ["benchmark_id", "answer", "reason", "selected_memory_id"]},
            "task4": {"required": ["benchmark_id", "answer", "used_memory_fact"]},
        },
    }
