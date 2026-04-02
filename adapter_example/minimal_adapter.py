from __future__ import annotations

import json
import sys
from typing import Any


def _candidate_memories(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row.get("input", {}).get("candidate_memories") or []


def _selected_memory_from_task4(row: dict[str, Any]) -> dict[str, Any]:
    model_input = row.get("input", {}).get("model_input", {})
    selected = model_input.get("selected_memory")
    if isinstance(selected, dict):
        return selected
    return {}


def predict_task1(row: dict[str, Any]) -> dict[str, Any]:
    # Replace this stub with your extraction model.
    return {
        "benchmark_id": row["benchmark_id"],
        "memory_items": [],
    }


def predict_task2(row: dict[str, Any]) -> dict[str, Any]:
    # Replace this stub with your memory-update model.
    return {
        "benchmark_id": row["benchmark_id"],
        "memory_items": [],
    }


def predict_task3(row: dict[str, Any]) -> dict[str, Any]:
    # Replace this stub with your retrieval logic.
    candidates = _candidate_memories(row)
    first_candidate = candidates[0] if candidates else {}
    return {
        "benchmark_id": row["benchmark_id"],
        "answer": str(first_candidate.get("value") or ""),
        "reason": "Stub adapter selected the first candidate memory.",
        "selected_memory_id": str(first_candidate.get("memory_id") or ""),
    }


def predict_task4(row: dict[str, Any]) -> dict[str, Any]:
    # Replace this stub with your personalized response model.
    selected_memory = _selected_memory_from_task4(row)
    used_memory_fact = str(selected_memory.get("value") or "")
    return {
        "benchmark_id": row["benchmark_id"],
        "answer": "Stub personalized response.",
        "used_memory_fact": used_memory_fact,
    }


def predict(row: dict[str, Any]) -> dict[str, Any]:
    task = row.get("task")
    if task == "task1":
        return predict_task1(row)
    if task == "task2":
        return predict_task2(row)
    if task == "task3":
        return predict_task3(row)
    if task == "task4":
        return predict_task4(row)
    raise ValueError(f"Unsupported task: {task}")


def main() -> None:
    row = json.load(sys.stdin)
    prediction = predict(row)
    # ASCII-safe JSON keeps the adapter reliable on Windows terminals/code pages.
    json.dump(prediction, sys.stdout, ensure_ascii=True)


if __name__ == "__main__":
    main()
