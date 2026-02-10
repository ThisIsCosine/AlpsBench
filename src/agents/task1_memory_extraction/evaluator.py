"""
Task 1 Evaluator: Memory Extraction.
Compares model-extracted memories with ground truth using F1/recall or a judge.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Callable

from ..shared import call_model_api


TASK1_EVALUATOR_PROMPT = """SYSTEM:
You are Task 1 Evaluator. Given a memory-extraction probe, run the model on the
dialogue and extract memories. Return JSON only.
"""

class Task1Evaluator:
    def __init__(self, system_prompt: str = TASK1_EVALUATOR_PROMPT) -> None:
        self.system_prompt = system_prompt

    def build_user_prompt(self, probe: Dict[str, Any]) -> str:
        # We pass the extraction specification via system prompt (probe["query"]).
        # User prompt contains only the untrusted dialogue data.
        payload = {"dialogue": probe.get("dialogue") or []}
        return json.dumps(payload, ensure_ascii=True)

    def _extract_json(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None

    def _parse_memory_items(self, raw_output: str) -> List[Dict[str, Any]]:
        data = self._extract_json(raw_output)
        if isinstance(data, dict):
            items = data.get("memory_items") or data.get("memories") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []
        parsed = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            evidence_text = item.get("evidence_text", "")
            # Some models may output nested evidence objects; accept best-effort extraction.
            if not evidence_text and isinstance(item.get("evidence"), dict):
                evidence_text = (item.get("evidence") or {}).get("text", "") or ""
            parsed.append(
                {
                    "memory_id": item.get("memory_id") or f"p{idx+1}",
                    "type": item.get("type", "direct"),
                    "label": item.get("label", "UNMAPPED"),
                    "value": item.get("value", ""),
                    "confidence": item.get("confidence", None),
                    "evidence_text": evidence_text,
                }
            )
        return parsed

    def evaluate_once(
        self,
        probe: Dict[str, Any],
        model_id: str,
        call_model,
        api_config: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        # Use Task1 query/spec as system prompt to enforce schema and rules.
        system_prompt = probe.get("query") or self.system_prompt
        user_prompt = self.build_user_prompt(probe)
        if call_model is None:
            raw_output = call_model_api(model_id, "M_extract", system_prompt, user_prompt, api_config)
        else:
            raw_output = call_model(model_id, "M_extract", system_prompt, user_prompt)
        return {
            "probe_id": probe.get("metadata", {}).get("dialog_id", ""),
            "model": model_id,
            "raw_output": raw_output,
            "memory_items": self._parse_memory_items(raw_output),
        }

    def evaluate_models(
        self,
        probe: Dict[str, Any],
        model_list: List[str],
        call_model,
        api_config: Dict[str, Any] | None = None,
        log_fn: Callable[[str, Dict[str, Any]], None] | None = None,
    ) -> Dict[str, Any]:
        runs = []
        if log_fn:
            log_fn("task1_evaluator_start", {"models": model_list})
        for model_id in model_list:
            try:
                result = self.evaluate_once(probe, model_id, call_model, api_config)
                if log_fn:
                    log_fn(
                        "task1_evaluator_result",
                        {
                            "model": model_id,
                            "memories": len(result["memory_items"]),
                        },
                    )
                runs.append(result)
            except Exception as exc:  # noqa: BLE001
                # print(str(exc))
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
                    log_fn("task1_evaluator_error", {"model": model_id, "error": str(exc)})
        if log_fn:
            log_fn("task1_evaluator_done", {"runs": len(runs)})
        return {"probe_id": probe.get("metadata", {}).get("dialog_id", ""), "runs": runs}
