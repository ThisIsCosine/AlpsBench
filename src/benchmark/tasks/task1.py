from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "task1_extract.txt"


def extraction_prompt() -> str:
    # The supplied DEFAULT_EXTRACT_PROMPT has no trailing newline.
    return PROMPT_PATH.read_text(encoding="utf-8").removesuffix("\n")


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if row.get("task") != "task1" or not row.get("benchmark_id"):
        raise ValueError("Expected a Task 1 model_input row with a benchmark_id.")
    payload = row.get("input")
    if not isinstance(payload, dict):
        raise ValueError("Task 1 input must be an object.")
    dialogue = payload.get("dialogue")
    if not dialogue:
        sessions = payload.get("sessions") or []
        dialogue = sessions[0].get("turns") if sessions else None
    if not isinstance(dialogue, list) or not dialogue:
        raise ValueError("Task 1 requires a nonempty dialogue.")
    turns = []
    for turn in dialogue:
        if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}:
            raise ValueError("Dialogue turns must have user or assistant roles.")
        if not isinstance(turn.get("text"), str):
            raise ValueError("Dialogue turn text must be a string.")
        turns.append({"role": turn["role"], "text": turn["text"]})
    # Send only the conversation. Benchmark metadata and any references stay local.
    return [
        {"role": "system", "content": extraction_prompt()},
        {"role": "user", "content": json.dumps({"dialogue": turns}, ensure_ascii=False)},
    ]


def parse_prediction(content: str, benchmark_id: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model returned no text content.")
    text = content.strip()
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].lower() in {"```", "```json"} and lines[-1] == "```":
        text = "\n".join(lines[1:-1])
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("Model response is not valid JSON.") from None
    if not isinstance(result, dict) or set(result) != {"memory_items"}:
        raise ValueError("Model must return an object containing only memory_items.")
    memories = result["memory_items"]
    if not isinstance(memories, list):
        raise ValueError("Model memory_items must be an array.")
    for item in memories:
        if not isinstance(item, dict):
            raise ValueError("Each predicted memory must be an object.")
        for key in ("type", "label", "value", "evidence_text"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Predicted memory requires a nonempty string field: {key}.")
        if "label_suggestion" not in item or (
            item["label_suggestion"] is not None and not isinstance(item["label_suggestion"], str)
        ):
            raise ValueError("label_suggestion must be a string or null.")
        confidence = item.get("confidence")
        if type(confidence) not in (float, int) or not 0 <= confidence <= 1:
            raise ValueError("Memory confidence must be a number between 0 and 1.")
    # Preserve the model's predictions; never fill fields using reference outputs.
    return {"benchmark_id": benchmark_id, "memory_items": memories}


def describe_task() -> dict[str, str]:
    return {
        "task": "task1",
        "public_dir": "task1",
        "purpose": "Memory extraction benchmark track.",
        "status": "Extraction prompt, API adapter, and public evaluation available.",
    }
