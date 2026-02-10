"""
Task 3 Evaluator: Memory Retrieval.
Runs model on dialogue + query, then judges memory usage.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Callable

from ..shared import call_model_api, normalize_text, truncate_dialogue, extract_response_text


TASK3_EVALUATOR_PROMPT = """SYSTEM:
You are a helpful assistant. Answer the query using the provided candidate memories.
One of the candidate memories is the correct one needed to answer the query.

Return JSON only:
{
  "answer": "string",
  "reason": "string",
  "selected_memory_id": "string"
}
The "reason" should explain how you arrived at the answer.
The "selected_memory_id" must be the ID of the memory you used from the candidates.
"""

TASK3_JUDGE_PROMPT = """SYSTEM:
You are a judge for memory retrieval. Determine whether the model used the selected memory.
Use only the model's reason and the selected_memory content.
Return JSON only:
{
  "used_memory": true|false,
  "score": 0.0,
  "reason": "string"
}
Score in [0,1]. If the reason clearly uses selected_memory, score should be high.
"""


class Task3Evaluator:
    def __init__(self, system_prompt: str = TASK3_EVALUATOR_PROMPT) -> None:
        self.system_prompt = system_prompt

    def build_user_prompt(self, probe: Dict[str, Any]) -> str:
        include_label = True
        env_flag = os.environ.get("TASK3_INCLUDE_LABEL")
        if env_flag is not None and env_flag.strip() != "":
            include_label = env_flag.strip() not in {"0", "false", "False", "no", "NO"}

        candidates = probe.get("candidate_memories") or []
        if not candidates and probe.get("selected_memory"):
            # Fallback for easy mode or when candidates are not explicitly set
            candidates = [probe["selected_memory"]]
        
        candidate_payload = []
        for m in candidates:
            item = {
                "memory_id": m.get("memory_id"),
                "value": m.get("value"),
            }
            if include_label:
                item["label"] = m.get("label")
            candidate_payload.append(item)

        payload = {
            "query": probe.get("query"),
            "candidate_memories": candidate_payload,
        }
        # print(payload)
        return json.dumps(payload, ensure_ascii=True)

    def _score_judge(
        self,
        predicted: Dict[str, Any],
        selected_memory: Dict[str, Any],
        query: str,
        judge_call_model,
        api_config: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        payload = {
            "query": query,
            "model_response": predicted,
            "selected_memory": selected_memory,
        }
        if judge_call_model is not None:
            raw = judge_call_model("judge", "M_judge", TASK3_JUDGE_PROMPT, json.dumps(payload, ensure_ascii=True))
        else:
            judge_model_id = (api_config or {}).get("judge_model_id", "judge")
            # print(judge_model_id)
            raw = call_model_api(
                judge_model_id,
                "M_judge",
                TASK3_JUDGE_PROMPT,
                json.dumps(payload, ensure_ascii=True),
                api_config=api_config,
                return_raw=False
            )
            # print(raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "used_memory": False,
                # "utterance_match": False,
                # "memory_match": False,
                "score": 0.0,
                "reason": "invalid_judge_output",
            }

    @staticmethod
    def _parse_model_output(raw_output: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {
            "answer": raw_output,
            "reason": "",
        }

    def evaluate_once(
        self,
        probe: Dict[str, Any],
        model_id: str,
        call_model,
        api_config: Dict[str, Any] | None,
        get_usage: bool = False
    ) -> Dict[str, Any]:
        if api_config:
            dialogue = truncate_dialogue(
                probe.get("dialogue", []),
                max_turns=api_config.get("dialogue_max_turns"),
                max_chars=api_config.get("dialogue_max_chars"),
            )
            probe = dict(probe)
            probe["dialogue"] = dialogue
        user_prompt = self.build_user_prompt(probe)
        if call_model is None:
            raw_output = call_model_api(model_id, "M_retrieve", self.system_prompt, user_prompt, api_config, return_raw=get_usage)

        else:
            raw_output = call_model(model_id, "M_retrieve", self.system_prompt, user_prompt, return_raw=get_usage)
        # parsed = self._parse_model_output(raw_output)
        # return {
        #     "probe_id": probe.get("metadata", {}).get("dialog_id", ""),
        #     "model": model_id,
        #     "raw_output": raw_output,
        #     "parsed_response": parsed,
        # }
        token_usage = {}
        text_output = ""

        if isinstance(raw_output, dict):
            # 1. 提取 Usage
            token_usage = raw_output.get("usage") or raw_output.get("token_usage") or {}
            
            # 2. 提取文本内容 (使用 shared 中导入的工具函数)
            text_output = extract_response_text(raw_output)
        else:
            # 兼容旧逻辑或 mock，如果是字符串则直接使用
            text_output = str(raw_output)

        # 3. 解析模型输出的 JSON 业务逻辑
        parsed = self._parse_model_output(text_output)
        
        return {
            "probe_id": probe.get("metadata", {}).get("dialog_id", ""),
            "model": model_id,
            "raw_output": text_output, # 仅保留文本内容供后续查看
            "parsed_response": parsed,
            "token_usage": token_usage, # <--- 新增字段
        }


    def evaluate_models(
        self,
        probe: Dict[str, Any],
        model_list: List[str],
        call_model,
        use_judge: bool = False,
        judge_call_model=None,
        api_config: Dict[str, Any] | None = None,
        log_fn: Callable[[str, Dict[str, Any]], None] | None = None,
        get_usage: bool = False
    ) -> Dict[str, Any]:
        runs = []
        if log_fn:
            log_fn("task3_evaluator_start", {"models": model_list})
        for model_id in model_list:
            try:
                if log_fn:
                    log_fn(
                        "task3_evaluator_input",
                        {
                            "model": model_id,
                            "probe_id": probe.get("metadata", {}).get("dialog_id", ""),
                            "query": probe.get("query"),
                            "has_dialogue": bool(probe.get("dialogue")),
                            "selected_memory_id": (probe.get("selected_memory") or {}).get("memory_id"),
                        },
                    )
                result = self.evaluate_once(probe, model_id, call_model, api_config, get_usage)
                if use_judge:
                    # print(judge_call_model)
                    result["judge"] = self._score_judge(
                        result.get("parsed_response") or {},
                        probe.get("selected_memory") or {},
                        probe.get("query") or "",
                        judge_call_model,
                        api_config,
                    )
                    result["used_memory"] = bool(result["judge"].get("used_memory"))
                if log_fn:
                    log_fn(
                        "task3_evaluator_result",
                        {
                            "model": model_id,
                            "used_memory": result.get("used_memory"),
                        },
                    )
                runs.append(result)
            except Exception as exc:  # noqa: BLE001
                runs.append(
                    {
                        "probe_id": probe.get("metadata", {}).get("dialog_id", ""),
                        "model": model_id,
                        "raw_output": None,
                        "error": str(exc),
                    }
                )
                if log_fn:
                    log_fn("task3_evaluator_error", {"model": model_id, "error": str(exc)})
        if log_fn:
            log_fn("task3_evaluator_done", {"runs": len(runs)})
        return {"probe_id": probe.get("metadata", {}).get("dialog_id", ""), "runs": runs}
