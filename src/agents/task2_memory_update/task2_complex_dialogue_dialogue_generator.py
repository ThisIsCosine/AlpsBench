"""
Task 2 helper: complex dialogue and expected memory generator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..shared import call_model_api, iter_jsonl


@dataclass
class ConflictMemorySpec:
    memory: Dict[str, Any]
    conflict_with_index: int
    conflict_count: int = 1


TASK2_COMPLEX_DIALOGUE_PROMPT = """SYSTEM:
You are a Task 2 dialogue and memory state generator.
You will receive a JSON input with:
- dialogue: original conversation turns
- memory_items: existing structured memories (gold)
- selected_memory_id: the specific memory ID that MUST be contradicted/updated if K > 0
- target_memory_to_update: (Optional) The specific memory object content that needs to be contradicted.
- controls: fake_memory (N), new_memory (M), conflict_memory (K)

[Goal]
1. Generate a NEW, realistic dialogue that imitates the original style.
2. Based on this NEW dialogue, output the UPDATED structured memory list (updated_memory_items).

[Dialogue Rules]
- fake_memory (N): weave N plausible but IRRELEVANT facts into the dialogue. These are "noise" and MUST NOT be added to memory.
- new_memory (M): introduce EXACTLY M new significant facts about the user.
- conflict_memory (K): if K > 0, the dialogue MUST contain info that EXPLICITLY CONTRADICTS the 'target_memory_to_update'. 

[Memory Update Rules for 'updated_memory_items']
- 1. START with ALL items from the original 'memory_items'.
- 2. DO NOT DELETE any original items unless they match 'selected_memory_id'.
- 3. ADD EXACTLY M items for the 'new_memory' facts introduced in the NEW dialogue. (IDs: m_new_1, m_new_2...)
- 4. If conflict_memory (K > 0):
     - FIND the item where memory_id == 'selected_memory_id'.
     - REMOVE it.
     - ADD a NEW replacement item reflecting the updated truth. (ID: "[selected_memory_id]_updated").
- 5. DO NOT ADD items for 'fake_memory'.
- 6. MATH CHECK: Final count MUST BE Original_Count + M.
- 7. Every item MUST follow the exact schema of the input items.

Return ONLY a valid JSON object. No other text.
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Robust extraction of JSON from LLM output."""
    s = text.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    
    try:
        return json.loads(s)
    except:
        # Try finding the first '{' and last '}'
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1:
            candidate = s[start:end+1]
            # Replace Pythonisms
            candidate = candidate.replace("None", "null").replace("True", "true").replace("False", "false")
            return json.loads(candidate)
    raise ValueError("Could not parse JSON from model output")


def _build_user_payload(
    record: Dict[str, Any],
    fake_memory: int,
    new_memory: int,
    conflict_count: int,
    conflict_spec: ConflictMemorySpec | None,
) -> Dict[str, Any]:
    payload = {
        "dialogue": (record.get("sessions") or [{}])[0].get("turns")
        or record.get("dialogue")
        or [],
        "memory_items": record.get("memory_items") or record.get("memories") or [],
        "selected_memory_id": record.get("selected_memory_id"),
        "controls": {
            "fake_memory": fake_memory,
            "new_memory": new_memory,
            "conflict_memory": conflict_count
        },
    }
    if conflict_count > 0:
        selected_id = record.get("selected_memory_id")
        target = next((m for m in payload["memory_items"] if m.get("memory_id") == selected_id), None)
        if target:
            payload["target_memory_to_update"] = {"label": target.get("label"), "old_value": target.get("value")}
    return payload


def _load_api_config(api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if api_config is not None:
        return dict(api_config)
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "api.json"))
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except:
        pass
    return {}


def generate_complex_dialogue_memory(
    record: Dict[str, Any],
    *,
    fake_memory: int,
    new_memory: int,
    conflict_count: int,
    conflict_spec: ConflictMemorySpec | None = None,
    model_id: Optional[str] = None,
    call_model=None,
    api_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = _build_user_payload(record, fake_memory, new_memory, conflict_count, conflict_spec)
    user_prompt = json.dumps(payload, ensure_ascii=True)
    cfg = _load_api_config(api_config)
    
    if call_model is None:
        raw = call_model_api(model_id or "gpt-4.1-mini", "M_generate", TASK2_COMPLEX_DIALOGUE_PROMPT, user_prompt, cfg)
    else:
        raw = call_model(TASK2_COMPLEX_DIALOGUE_PROMPT, user_prompt)

    result = _extract_json_object(raw)

    if not isinstance(result, dict) or "updated_memory_items" not in result:
        raise ValueError("invalid_llm_output_format")

    # Count check
    orig_count = len(payload["memory_items"])
    expected_count = orig_count + new_memory
    actual_count = len(result["updated_memory_items"])
    if actual_count != expected_count:
        print(f"    [WARN] Count mismatch: Expected {expected_count}, got {actual_count}")

    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for item in result["updated_memory_items"]:
        if "updated_at" in item and not item["updated_at"]:
            item["updated_at"] = now_ts

    # 强制把 controls 补齐到结果中，防止模型漏掉
    result["controls"] = result.get("controls") or payload["controls"]

    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate complex dialogue and expected memory state.")
    parser.add_argument("jsonl_path", nargs='?', 
                        default="benchmark/dev_with_selected_memory_id/ability1/Personal_Background/Education/direct.jsonl")
    parser.add_argument("--line", type=int, default=0)
    parser.add_argument("--model-id", default="gpt-4.1-mini")
    parser.add_argument("--fake-memory", type=int, default=0)
    parser.add_argument("--new-memory", type=int, default=1)
    parser.add_argument("--conflict", type=int, default=0)
    args = parser.parse_args()

    record = None
    for i, item in enumerate(iter_jsonl(args.jsonl_path)):
        if i == args.line:
            record = item
            break
    
    if record:
        selected_id = record.get("selected_memory_id")
        orig_mems = record.get("memory_items") or record.get("memories") or []
        selected_mem = next((m for m in orig_mems if m.get("memory_id") == selected_id), None)
        conflict_spec = None
        if args.conflict > 0 and selected_mem:
            conflict_spec = ConflictMemorySpec(memory=selected_mem, conflict_with_index=0, conflict_count=args.conflict)

        out = generate_complex_dialogue_memory(
            record,
            fake_memory=args.fake_memory,
            new_memory=args.new_memory,
            conflict_count=args.conflict,
            conflict_spec=conflict_spec,
            model_id=args.model_id
        )
        print(json.dumps(out, ensure_ascii=True, indent=2))

if __name__ == "__main__":
    _main()
