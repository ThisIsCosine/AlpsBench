"""
One-call generator: produce exactly ONE new memory item from an existing record.

Goal:
- Read dialogue/memory_items from a jsonl record
- Call LLM exactly once
- Output ONE new memory item that follows the same schema as record["memory_items"]
- Default output language: English (evidence text stays verbatim from dialogue)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..shared import call_model_api, iter_jsonl


TASK2_ONE_MEMORY_PROMPT = """SYSTEM:
You are a memory writer.

You will receive a JSON payload with:
- dialogue: a list of turns [{"role": "...", "text": "...", ...}, ...]
- memory_items: existing memory items (already stored)
- selected_memory_id: optional id of a memory item to focus on
- schema_keys: the exact list of keys each memory item MUST have
- output_language: language for the generated fields (value/reasoning/label if needed)

Task:
- Generate EXACTLY ONE new memory item that is supported by the dialogue but is NOT already present in memory_items.
- The new item MUST follow the same schema as existing memory_items:
  - Output must be JSON only.
  - Use STRICT JSON values: null/true/false (NOT Python None/True/False).
  - Output must contain exactly these keys (no more, no fewer): schema_keys.
  - "evidence.text" MUST be a verbatim quote/snippet from the dialogue (keep original language).
  - "value" and "reasoning" MUST be written in output_language.
  - "memory_id" MUST be "m_new".
  - "confidence" must be a float in [0, 1].
  - "time_scope" must be "short_term" or "long_term".
  - Prefer concise, specific memories with clear evidence.

Return ONLY the memory item JSON object (not wrapped in any other keys).
"""


def _extract_dialogue(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (record.get("sessions") or [{}])[0].get("turns") or record.get("dialogue") or []


def _infer_schema_keys(record: Dict[str, Any]) -> List[str]:
    items = record.get("memory_items") or record.get("memories") or []
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return sorted(items[0].keys())
    # Fallback (best-effort)
    return sorted(
        [
            "memory_id",
            "type",
            "label",
            "label_suggestion",
            "value",
            "reasoning",
            "evidence",
            "confidence",
            "time_scope",
            "emotion",
            "preference_attitude",
            "updated_at",
        ]
    )


def _normalize_memory_item(item: Dict[str, Any], schema_keys: List[str]) -> Dict[str, Any]:
    # Keep only schema keys; fill missing with None
    normalized: Dict[str, Any] = {k: item.get(k, None) for k in schema_keys}

    # Force required fields / sane defaults
    normalized["memory_id"] = "m_new"

    # confidence -> float in [0,1]
    conf = normalized.get("confidence")
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.7
    if conf_f < 0.0:
        conf_f = 0.0
    if conf_f > 1.0:
        conf_f = 1.0
    normalized["confidence"] = conf_f

    # updated_at -> now
    if "updated_at" in normalized:
        normalized["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # evidence must be dict if present
    if "evidence" in normalized and not isinstance(normalized.get("evidence"), dict):
        normalized["evidence"] = None

    # time_scope must be one of allowed
    if "time_scope" in normalized:
        ts = normalized.get("time_scope")
        if ts not in ("short_term", "long_term"):
            normalized["time_scope"] = "long_term"

    return normalized


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of a JSON object from model output.
    Handles:
    - code fences
    - leading/trailing commentary
    - common Python-isms (None/True/False)
    """
    if not isinstance(text, str):
        raise ValueError("model_output_not_string")
    s = text.strip()
    # Strip code fences if present
    if s.startswith("```"):
        s = s.strip("`").strip()
    # 1) direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) try substring between first '{' and last '}'
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        for attempt in (candidate, candidate.replace("None", "null").replace("True", "true").replace("False", "false")):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    raise ValueError("invalid_json_object")


def generate_one_new_memory_item(
    record: Dict[str, Any],
    *,
    model_id: str,
    output_language: str = "English",
    api_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dialogue = _extract_dialogue(record)
    memory_items = record.get("memory_items") or record.get("memories") or []
    schema_keys = _infer_schema_keys(record)

    payload = {
        "dialogue": dialogue,
        "memory_items": memory_items,
        "selected_memory_id": record.get("selected_memory_id"),
        "schema_keys": schema_keys,
        "output_language": output_language,
    }
    user_prompt = json.dumps(payload, ensure_ascii=True)
    raw = call_model_api(model_id, "M_generate_memory", TASK2_ONE_MEMORY_PROMPT, user_prompt, api_config=api_config)
    try:
        item = _extract_json_object(raw)
    except Exception as exc:
        raise ValueError(f"invalid_llm_output: {exc}") from exc
    if not isinstance(item, dict):
        raise ValueError("invalid_llm_output_type")
    got_keys = sorted(item.keys())
    if got_keys != schema_keys:
        raise ValueError(
            "invalid_llm_output_schema_mismatch: "
            f"expected_keys={json.dumps(schema_keys, ensure_ascii=True)} "
            f"got_keys={json.dumps(got_keys, ensure_ascii=True)}"
        )
    return _normalize_memory_item(item, schema_keys)


def _get_record_at_line(jsonl_path: str, line: int) -> Tuple[int, Dict[str, Any]]:
    if line < 0:
        raise ValueError("line must be >= 0")
    for i, rec in enumerate(iter_jsonl(jsonl_path)):
        if i == line:
            return i, rec
    raise IndexError(f"jsonl has fewer than {line+1} lines: {jsonl_path}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate ONE new memory item (one model call).")
    parser.add_argument("jsonl_path", help="Path to a jsonl record file.")
    parser.add_argument("--line", type=int, default=0, help="0-based line index to load (default: 0).")
    parser.add_argument("--model-id", default="gpt-4.1-mini", help="Model id to use (default: gpt-4.1-mini).")
    parser.add_argument("--lang", default="English", help="Output language for value/reasoning (default: English).")
    args = parser.parse_args()

    _, record = _get_record_at_line(args.jsonl_path, args.line)
    item = generate_one_new_memory_item(record, model_id=args.model_id, output_language=args.lang, api_config=None)

    # ASCII-safe to avoid Windows console encoding crashes.
    print(json.dumps(item, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    _main()

