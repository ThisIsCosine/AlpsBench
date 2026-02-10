"""
Shared utilities for agent tasks.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple, Callable


_DEFAULT_API_CONFIG_CACHE: Dict[str, Any] | None = None


def _load_default_api_config() -> Dict[str, Any]:
    """
    Best-effort default api config loader.

    Priority:
    1) API_CONFIG_PATH / MODEL_API_CONFIG_PATH env var
    2) <repo>/configs/api.json (resolved relative to this file)

    Returns {} if no config can be loaded.
    """
    global _DEFAULT_API_CONFIG_CACHE
    if _DEFAULT_API_CONFIG_CACHE is not None:
        return _DEFAULT_API_CONFIG_CACHE

    candidates: List[str] = []
    env_path = (os.environ.get("API_CONFIG_PATH") or os.environ.get("MODEL_API_CONFIG_PATH") or "").strip()
    if env_path:
        candidates.append(env_path)
    # shared.py is at <repo>/src/agents/shared.py
    candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "api.json")))

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                _DEFAULT_API_CONFIG_CACHE = json.load(handle)
                if not isinstance(_DEFAULT_API_CONFIG_CACHE, dict):
                    _DEFAULT_API_CONFIG_CACHE = {}
                return _DEFAULT_API_CONFIG_CACHE
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            continue

    _DEFAULT_API_CONFIG_CACHE = {}
    return _DEFAULT_API_CONFIG_CACHE


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def resolve_session_id(payload: Dict[str, Any]) -> str | None:
    if not payload:
        return None
    session_id = payload.get("session_id")
    if session_id:
        return str(session_id)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("session_id"):
        return str(metadata.get("session_id"))
    for key in ("dialog_id", "seed_id", "record_id"):
        if payload.get(key):
            return str(payload.get(key))
    return None


def parse_option_choice(raw_output: str, options: List[str]) -> int | None:
    if not raw_output or not options:
        return None
    text = raw_output.strip().lower()
    for idx in range(1, len(options) + 1):
        if f"{idx}" == text or text.startswith(f"{idx}.") or text.startswith(f"{idx})"):
            return idx - 1
    letters = "abcdefghijklmnopqrstuvwxyz"
    for idx in range(len(options)):
        letter = letters[idx]
        if text == letter or text.startswith(f"{letter}.") or text.startswith(f"{letter})"):
            return idx
    for idx, opt in enumerate(options):
        opt_text = normalize_text(opt)
        if opt_text and opt_text in normalize_text(raw_output):
            return idx
    return None

def _extract_memory_fields(memory: Dict[str, Any]) -> Dict[str, Any]:
    try:
        confidence = float(memory.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    value = str(memory.get("value", "")).strip()
    evidence = ""
    if isinstance(memory.get("evidence"), dict):
        evidence = str(memory.get("evidence", {}).get("text", "")).strip()
    if isinstance(memory.get("evidence_text"), str):
        evidence = memory.get("evidence_text", "").strip()
    label = str(memory.get("label", "")).strip()
    label_suggestion = str(memory.get("label_suggestion", "")).strip()
    return {
        "confidence": confidence,
        "value": value,
        "evidence": evidence,
        "label": label,
        "label_suggestion": label_suggestion,
    }


def _memory_gate_prompt(task_type: str) -> str:
    return (
        "You are a strict memory quality gate. Decide which memories are usable to generate a high-quality "
        f"{task_type} query. Return JSON only: "
        '{"usable_indices": [0,1,...], "reason": "string"}'
    )


def is_memory_usable(
    memory: Dict[str, Any],
    min_confidence: float = 0.6,
    min_value_chars: int = 6,
    min_evidence_chars: int = 6,
    task_type: str | None = None,
    api_config: Dict[str, Any] | None = None,
) -> bool:
    fields = _extract_memory_fields(memory)
    confidence = fields["confidence"]
    value = fields["value"]
    evidence = fields["evidence"]
    label = fields["label"].lower()
    label_suggestion = fields["label_suggestion"].lower()
    curiosity_markers = ("curiosity", "curious", "curiou")
    if any(marker in label for marker in curiosity_markers) or any(
        marker in label_suggestion for marker in curiosity_markers
    ):
        return False
    basic_ok = confidence >= min_confidence and len(value) >= min_value_chars and len(evidence) >= min_evidence_chars
    if basic_ok:
        return True

    gate_model_id = (api_config or {}).get("memory_gate_model_id")
    if not gate_model_id:
        return False

    payload = {
        "task_type": task_type or "unknown_task",
        "memory": {
            "type": memory.get("type", ""),
            "label": memory.get("label", "UNMAPPED"),
            "value": value,
            "evidence_text": evidence,
            "confidence": confidence,
        },
    }
    raw = call_model_api(
        gate_model_id,
        "M_gate",
        _memory_gate_prompt(task_type or "unknown_task"),
        json.dumps(payload, ensure_ascii=True),
        api_config=api_config,
    )
    try:
        verdict = json.loads(raw)
        return bool(verdict.get("usable"))
    except json.JSONDecodeError:
        return False


def filter_memories_for_task(
    memories: List[Dict[str, Any]],
    task_type: str,
    api_config: Dict[str, Any] | None = None,
    min_confidence: float = 0.6,
    min_value_chars: int = 6,
    min_evidence_chars: int = 6,
) -> List[Dict[str, Any]]:
    if not memories:
        return []

    # Stage 1: basic filter only (no LLM)
    basic_candidates = [
        m
        for m in memories
        if is_memory_usable(
            m,
            min_confidence=min_confidence,
            min_value_chars=min_value_chars,
            min_evidence_chars=min_evidence_chars,
            task_type=task_type,
            api_config=None,
        )
    ]
    if not basic_candidates:
        return []

    # Stage 2: LLM gate only if we already have candidates
    gate_model_id = (api_config or {}).get("memory_gate_model_id")
    if not gate_model_id:
        return basic_candidates

    payload = {
        "task_type": task_type,
        "memories": [
            {
                "type": m.get("type", ""),
                "label": m.get("label", "UNMAPPED"),
                "value": _extract_memory_fields(m)["value"],
                "evidence_text": _extract_memory_fields(m)["evidence"],
                "confidence": _extract_memory_fields(m)["confidence"],
            }
            for m in basic_candidates
        ],
    }
    raw = call_model_api(
        gate_model_id,
        "M_gate",
        _memory_gate_prompt(task_type),
        json.dumps(payload, ensure_ascii=True),
        api_config=api_config,
    )
    try:
        verdict = json.loads(raw)
        indices = verdict.get("usable_indices") or []
        selected = []
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(basic_candidates):
                selected.append(basic_candidates[idx])
        return selected
    except json.JSONDecodeError:
        return []


def split_items(text: str) -> List[str]:
    items = []
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        if line:
            items.append(line)
    if not items:
        items = [s.strip() for s in text.split(";") if s.strip()]
    return items


def f1_recall_precision(pred_items: Iterable[str], gold_items: Iterable[str]) -> Tuple[float, float, float]:
    pred_norm = {normalize_text(x) for x in pred_items if x}
    gold_norm = {normalize_text(x) for x in gold_items if x}
    if not pred_norm and not gold_norm:
        return 1.0, 1.0, 1.0
    if not pred_norm or not gold_norm:
        return 0.0, 0.0, 0.0
    true_pos = len(pred_norm & gold_norm)
    precision = true_pos / len(pred_norm) if pred_norm else 0.0
    recall = true_pos / len(gold_norm) if gold_norm else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, recall, precision


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_event(log_path: str, event: str, payload: Dict[str, Any]) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
    }
    # print(json.dumps(entry, ensure_ascii=True))
    append_jsonl(log_path, entry)


def truncate_dialogue(
    dialogue: List[Dict[str, str]],
    max_turns: int | None = None,
    max_chars: int | None = None,
) -> List[Dict[str, str]]:
    if not dialogue:
        return []
    trimmed = dialogue[-max_turns:] if max_turns else list(dialogue)
    if not max_chars:
        return trimmed
    total = 0
    result: List[Dict[str, str]] = []
    for turn in reversed(trimmed):
        text = turn.get("text", "")
        if total + len(text) > max_chars:
            remain = max_chars - total
            if remain <= 0:
                break
            text = text[-remain:]
        result.append({"role": turn.get("role", ""), "text": text})
        total += len(text)
        if total >= max_chars:
            break
    return list(reversed(result))

def make_run_dir(base_dir: str, task_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(base_dir, task_name, timestamp)


def list_jsonl_files(root_dir: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            # Skip report/manifest helper files (convention: leading underscore).
            # These are often not actual benchmark records and can break seed loading.
            if name.startswith("_"):
                continue
            if name.endswith(".jsonl"):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_response_text(response_json: Dict[str, Any]) -> str:
    if isinstance(response_json.get("output"), str):
        return response_json["output"]
    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        message = choice.get("message") or {}
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(choice.get("text"), str):
            return choice["text"]
    if isinstance(response_json.get("text"), str):
        return response_json["text"]
    raise ValueError("Unsupported response format")


def _extract_stream_text(response) -> str:
    """
    Parse SSE-style streaming responses and accumulate content.
    Supports OpenAI/DeepSeek chat.completions delta format.
    """
    chunks: List[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            chunks.append(delta["content"])
            continue
        if isinstance(choice.get("text"), str):
            chunks.append(choice["text"])
            continue
        message = choice.get("message") or {}
        if isinstance(message.get("content"), str):
            chunks.append(message["content"])
    return "".join(chunks)


def call_model_api(
    model_id: str,
    condition: str,
    system_prompt: str,
    user_prompt: str,
    api_config: Dict[str, Any] | None = None,
    return_raw: bool = False, # return raw for token-cosuming statistic
) -> str | Dict[str, Any]:
    import time
    cfg = dict(_load_default_api_config()) if api_config is None else dict(api_config)
    endpoints = cfg.get("model_endpoints") or {}
    base_url = endpoints.get(model_id) or cfg.get("base_url") or os.environ.get("MODEL_API_BASE_URL")
    if not base_url:
        raise RuntimeError("MODEL_API_BASE_URL is not set and no base_url provided")
    model_keys = cfg.get("model_keys") or {}
    # print(model_keys)
    api_key = (
        model_keys.get(model_id, "")
    )
    # print(api_key)
    # print(model_id)
    timeout = int(cfg.get("timeout") or os.environ.get("MODEL_API_TIMEOUT") or "240")
    debug_log_path = cfg.get("debug_log_path")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    # print(payload)
    stream = bool(cfg.get("stream", False))
    if stream:
        payload["stream"] = True
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    max_retries = 10 # yx: more retries
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(base_url, data=data, headers=headers, method="POST")
            # --- 新增：打印完整请求信息 ---
            # print("\n" + "="*50)
            # print(f"DEBUG: [Attempt {attempt + 1}] Sending Request")
            # print(f"URL: {request.get_full_url()}")
            # print(f"Headers: {dict(request.headers)}")
            # print(f"Payload: {data.decode('utf-8')}")
            # print("="*50 + "\n")
            # ---------------------------
            if debug_log_path:
                log_event(
                    debug_log_path,
                    "api_call_start",
                    {"model": model_id, "base_url": base_url, "timeout": timeout, "attempt": attempt + 1},
                )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if stream:
                    body = _extract_stream_text(response)
                    if debug_log_path:
                        log_event(
                            debug_log_path,
                            "api_call_done",
                            {"model": model_id, "status": getattr(response, "status", None), "stream": True},
                        )
                    return body
                body = response.read().decode("utf-8")
                # print(body)
            if debug_log_path:
                log_event(
                    debug_log_path,
                    "api_call_done",
                    {"model": model_id, "status": getattr(response, "status", None), "stream": False},
                )
            response_json = json.loads(body)
            # print(response_json)
            if return_raw:
                # print(response_json)
                return response_json
            return extract_response_text(response_json)
            
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            is_last = (attempt == max_retries - 1)
            if debug_log_path:
                log_event(
                    debug_log_path,
                    "api_call_retry",
                    {"model": model_id, "attempt": attempt + 1, "error": str(exc), "will_retry": not is_last},
                )
            if is_last:
                raise
            time.sleep(retry_delay * (2 ** attempt))
            
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8") if exc.fp else ""
            if debug_log_path:
                log_event(
                    debug_log_path,
                    "api_call_error",
                    {"model": model_id, "status": exc.code, "error": error_body},
                )
            # Don't retry on 4xx errors except 429
            if exc.code == 429 and attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
    
    raise RuntimeError("API call failed after retries")
