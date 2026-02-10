"""
Task 2 Evaluator: Memory Update / Conflict Resolution.

Runs the evaluated model on a Task2 probe and parses its updated memory list.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from ..shared import call_model_api


def _extract_json(text: str) -> Any:
    if not isinstance(text, str):
        return None
    s = text.strip()
    # Strip code fences if present
    if "```json" in s:
        s = s.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in s:
        s = s.split("```", 1)[1].split("```", 1)[0]
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # Best-effort: find list or object substrings
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _parse_memory_items(raw_output: str) -> List[Dict[str, Any]]:
    data = _extract_json(raw_output)
    if isinstance(data, dict):
        items = data.get("memory_items") or data.get("memories") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    parsed: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        # Keep original keys but ensure core fields exist for scoring.
        out = dict(item)
        out.setdefault("memory_id", f"p{idx+1}")
        out.setdefault("type", "direct")
        out.setdefault("label", "UNMAPPED")
        out.setdefault("value", "")
        out.setdefault("confidence", None)
        parsed.append(out)
    return parsed


class Task2Evaluator:
    def __init__(self) -> None:
        pass

    def build_user_prompt(self, probe: Dict[str, Any]) -> str:
        full_dialogue = (probe.get("old_dialogue") or []) + (probe.get("new_dialogue") or [])
        payload = {
            "existing_memories": probe.get("old_memory_items") or [],
            "dialogue": full_dialogue,
        }
        return json.dumps(payload, ensure_ascii=True)

    def evaluate_once(
        self,
        probe: Dict[str, Any],
        model_id: str,
        call_model,
        api_config: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        system_prompt = probe.get("query") or "You are a helpful assistant."
        user_prompt = self.build_user_prompt(probe)
        if call_model is None:
            raw_output = call_model_api(model_id, "M_update", system_prompt, user_prompt, api_config=api_config)
        else:
            raw_output = call_model(model_id, "M_update", system_prompt, user_prompt)
        return {
            "probe_id": probe.get("metadata", {}).get("dialog_id", ""),
            "model": model_id,
            "raw_output": raw_output,
            "memory_items": _parse_memory_items(raw_output),
        }

    def evaluate_models(
        self,
        probe: Dict[str, Any],
        model_list: List[str],
        call_model,
        use_judge: bool = False,  # unused (kept for runner compatibility)
        judge_call_model=None,  # unused
        api_config: Dict[str, Any] | None = None,
        log_fn: Callable[[str, Dict[str, Any]], None] | None = None,
    ) -> Dict[str, Any]:
        runs = []
        if log_fn:
            log_fn("task2_evaluator_start", {"models": model_list})
        for model_id in model_list:
            try:
                result = self.evaluate_once(probe, model_id, call_model, api_config)
                runs.append(result)
                if log_fn:
                    log_fn("task2_evaluator_result", {"model": model_id, "memories": len(result.get("memory_items") or [])})
            except Exception as exc:  # noqa: BLE001
                runs.append(
                    {
                        "probe_id": probe.get("metadata", {}).get("dialog_id", ""),
                        "model": model_id,
                        "raw_output": None,
                        "memory_items": [],
                        "error": str(exc),
                    }
                )
                if log_fn:
                    log_fn("task2_evaluator_error", {"model": model_id, "error": str(exc)})
        if log_fn:
            log_fn("task2_evaluator_done", {"runs": len(runs)})
        return {"probe_id": probe.get("metadata", {}).get("dialog_id", ""), "runs": runs}

    # Backward-compatible API (older callers)
    def evaluate_sample(self, model_id: str, sample: Dict[str, Any], api_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        probe = {
            "old_dialogue": sample.get("old_dialogue") or [],
            "new_dialogue": sample.get("new_dialogue") or [],
            "old_memory_items": sample.get("old_memory_items") or [],
            "query": sample.get("query") or "You are a helpful assistant. Return JSON list only.",
            "metadata": sample.get("metadata") or {},
        }
        out = self.evaluate_once(probe, model_id, call_model=None, api_config=api_config)
        return out.get("memory_items") or []
