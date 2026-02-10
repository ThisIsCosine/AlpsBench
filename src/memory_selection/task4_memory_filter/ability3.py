"""Ability 3: Real-virtual distinguishing ability.
    [All the candidate memories are from the 60000 dataset]
    
    Give the dialogue, ask personalized questions without saying anything about role-play, 
    if the LLM could identify the role-play from the dialogue, not real conversation, PASS
    otherwise, fail
"""

from __future__ import annotations

Filter_PROMPT = """
You are a dataset curator. Your job is to FILTER memory items for Ability 3: Real–Virtual Distinguishing.

Ability 3 definition (what we want to test):
- The dialogue contains content that is NOT a real user fact (e.g., the user is translating text, writing for someone else, drafting a story/assignment, simulating a scenario, or otherwise producing statements that should not be stored as the user’s true personal profile).
- A “good” Ability-3 memory candidate is a memory item whose content LOOKS like a personal fact, but is actually only present inside that non-real context, so a model should NOT treat it as a true user attribute in future interactions.

Input you will receive:
- session_id: a string like "sess_..."
- memory_id: a string like "m1"
- dialogue: a list of turns
- memory_item: { label, value, evidence_text }  (already labeled; do not relabel)

Your task:
1) Decide whether this memory_item is a GOOD candidate for Ability 3 evaluation (KEEP) or should be removed (DROP).
2) If KEEP, write 1–2 non-leading user-perspective queries that can be asked later to test whether a model mistakenly treats the memory as a real user fact.

KEEP criteria (must satisfy ALL):
- The memory_item.value is a claim about the user (identity/background/possession/relationship/event/trait) that a model could incorrectly treat as a real personal fact.
- The dialogue/evidence suggests the claim is context-bound / non-real (e.g., translation of third-party text, prompted writing, hypothetical scenario, or content clearly not self-asserted as reality).
- The claim is concrete enough to form a clear yes/no judgment about whether it should be stored or treated as a fact later.

DROP if ANY is true:
- The memory is clearly a real user fact (self-reported personal background, stable preference, real relationship, real location).
- The memory is purely procedural or task meta (e.g., “user asked for translation,” “user asked for week-to-day conversion”).
- The memory is too vague to ask about (no specific entity/attribute).
- The memory is only about the assistant’s behavior rather than the user.

Non-leading query rules (very important):
- Your queries MUST NOT contain any of these words/phrases: role-play, fictional, pretend, imaginary, story, character, script, novel, magic, wizard.
- Do NOT hint that the claim is not real.
- Queries should be answerable as “Yes/No” (or “Should/Should not”) by a grader/model.
- Each query must be <= 25 English words, and must end with a question mark.

OUTPUT FORMAT (hard constraints):
- Output exactly ONE line of JSON (no extra lines).
- Either:
  {"decision":"DROP"}
- Or:
  {"decision":"KEEP","session_id":"...","memory_id":"...","queries":["...","..."]}

Now process the following:

session_id: {SESSION_ID}
memory_id: {MEMORY_ID}
memory_item.value: {MEMORY_VALUE}
memory_item.evidence_text: {EVIDENCE_TEXT}
dialogue: {DIALOGUE_TURNS}

Return ONE line following the output format.

"""

import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional, TextIO

try:
    from agents.shared import call_model_api
except ImportError:  # Allow direct script execution.
    import sys

    current_dir = os.path.dirname(__file__)
    src_dir = os.path.dirname(os.path.dirname(current_dir))
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    from agents.shared import call_model_api


INPUT_DIR = "data/wildchat/memories/selected/task4/ability3"
OUTPUT_DIR = "data/wildchat/memories/filtered/task4/ability3"
DEFAULT_MODEL_ID = "deepseek-reasoner"
DEBUG_LOG_LIMIT = 50


def iter_jsonl_paths(root_dir: str) -> List[str]:
    pattern = os.path.join(root_dir, "**", "*.jsonl")
    return sorted(glob.glob(pattern, recursive=True))


def iter_records(file_path: str) -> Iterable[dict]:
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_memory_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    mem_type = str(value).strip().lower()
    if mem_type in {"direct", "indirect"}:
        return mem_type
    return None


def _extract_session_id(record: dict) -> str:
    sessions = record.get("sessions") or []
    if sessions and sessions[0].get("session_id"):
        return str(sessions[0].get("session_id"))
    match = record.get("match") or {}
    if match.get("matched_session_id"):
        return str(match.get("matched_session_id"))
    if record.get("session_id"):
        return str(record.get("session_id"))
    return ""


def _extract_dialogue_text(record: dict) -> str:
    sessions = record.get("sessions") or []
    if not sessions:
        return ""
    turns = sessions[0].get("turns") or []
    lines = []
    for turn in turns:
        role = str(turn.get("role") or "").strip()
        text = str(turn.get("text") or "").strip()
        if not role and not text:
            continue
        lines.append(f"{role}: {text}".strip())
    return "\n".join(lines)


def _extract_evidence_text(memory_item: dict) -> str:
    if isinstance(memory_item.get("evidence"), dict):
        text = memory_item.get("evidence", {}).get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    evidence_text = memory_item.get("evidence_text")
    if isinstance(evidence_text, str):
        return evidence_text.strip()
    return ""


def _build_prompt(
    session_id: str,
    memory_id: str,
    label: str,
    memory_value: str,
    evidence_text: str,
    dialogue_turns: str,
) -> str:
    prompt = Filter_PROMPT
    prompt = prompt.replace("{SESSION_ID}", session_id)
    prompt = prompt.replace("{MEMORY_ID}", memory_id)
    prompt = prompt.replace("{LABEL}", label)
    prompt = prompt.replace("{MEMORY_VALUE}", memory_value)
    prompt = prompt.replace("{EVIDENCE_TEXT}", evidence_text)
    prompt = prompt.replace("{DIALOGUE_TURNS}", dialogue_turns)
    return prompt


def _split_queries(raw_text: str) -> List[str]:
    if not raw_text:
        return []
    parts = [part.strip() for part in raw_text.split("|") if part.strip()]
    queries = []
    for part in parts:
        cleaned = part
        if cleaned.lower().startswith("q1:"):
            cleaned = cleaned[3:].strip()
        if cleaned.lower().startswith("q2:"):
            cleaned = cleaned[3:].strip()
        if cleaned:
            queries.append(cleaned)
    return queries


def _parse_model_output(raw: str) -> Dict[str, Any] | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None

    upper = text.upper()
    if upper == "DROP":
        return {"decision": "DROP"}
    if upper.startswith("DROP"):
        return {"decision": "DROP"}

    if upper.startswith("KEEP"):
        text = text[4:].lstrip(" \t:-")

    if "\t" in text:
        parts = text.split("\t", 2)
        if len(parts) >= 3:
            queries = _split_queries(parts[2])
            return {
                "decision": "KEEP",
                "session_id": parts[0].strip(),
                "memory_id": parts[1].strip(),
                "queries": queries,
            }

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None

    if isinstance(obj, dict):
        decision = str(obj.get("decision", "")).upper()
        if not decision and obj.get("queries"):
            decision = "KEEP"
        if decision not in {"KEEP", "DROP"}:
            return None
        obj["decision"] = decision
        return obj
    return None


def _output_path(label: str, mem_type: str) -> str:
    parts = [part.strip() for part in label.split("/") if part.strip()]
    return os.path.join(OUTPUT_DIR, *parts, f"{mem_type}.jsonl")


def _get_output_handle(
    label: str,
    mem_type: str,
    handles: Dict[str, TextIO],
) -> TextIO:
    key = f"{label}::{mem_type}"
    if key in handles:
        return handles[key]
    path = _output_path(label, mem_type)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "w", encoding="utf-8")
    handles[key] = handle
    return handle


def _load_api_config(api_config_path: str) -> Dict[str, Any]:
    with open(api_config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def filter_ability3(
    input_dir: str = INPUT_DIR,
    output_dir: str = OUTPUT_DIR,
    api_config_path: str = "configs/api.json",
    api_config: Optional[Dict[str, Any]] = None,
    model_id: Optional[str] = None,
) -> None:
    jsonl_paths = iter_jsonl_paths(input_dir)
    if not jsonl_paths:
        raise RuntimeError(f"No jsonl files found under: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    local_api_config = api_config if api_config is not None else _load_api_config(api_config_path)
    chosen_model_id = model_id or os.environ.get("TASK4_FILTER_MODEL_ID") or DEFAULT_MODEL_ID
    error_path = os.path.join(output_dir, "errors.log")
    log_path = os.path.join(output_dir, "run.log")

    handles: Dict[str, TextIO] = {}
    try:
        with open(error_path, "w", encoding="utf-8") as error_handle, open(
            log_path, "w", encoding="utf-8"
        ) as log_handle:
            print(f"[ability3] input_dir={input_dir}")
            print(f"[ability3] output_dir={output_dir}")
            print(f"[ability3] model_id={chosen_model_id}")
            log_handle.write(f"input_dir={input_dir}\n")
            log_handle.write(f"output_dir={output_dir}\n")
            log_handle.write(f"model_id={chosen_model_id}\n")
            log_handle.write(f"files={len(jsonl_paths)}\n")

            for path in jsonl_paths:
                print(f"[ability3] file={path}")
                log_handle.write(f"file={path}\n")
                record_count = 0
                kept_count = 0
                drop_count = 0
                error_count = 0
                api_calls = 0
                for record in iter_records(path):
                    record_count += 1
                    session_id = _extract_session_id(record)
                    dialogue_turns = _extract_dialogue_text(record)
                    for item in record.get("memory_items") or []:
                        label = str(item.get("label") or "").strip()
                        if not label:
                            continue
                        mem_type = _normalize_memory_type(item.get("type"))
                        if not mem_type:
                            continue
                        memory_id = str(item.get("memory_id") or "")
                        memory_value = str(item.get("value") or "")
                        evidence_text = _extract_evidence_text(item)
                        prompt = _build_prompt(
                            session_id=session_id,
                            memory_id=memory_id,
                            label=label,
                            memory_value=memory_value,
                            evidence_text=evidence_text,
                            dialogue_turns=dialogue_turns,
                        )
                        try:
                            raw = call_model_api(
                                chosen_model_id,
                                "M_filter",
                                "You are a helpful assistant.",
                                prompt,
                                api_config=local_api_config,
                            )
                            api_calls += 1
                            print(
                                f"[ability3] ok session={session_id} memory={memory_id} "
                                f"label={label} type={mem_type}"
                            )
                        except Exception as exc:  # noqa: BLE001
                            error_handle.write(
                                f"{session_id}\t{memory_id}\tapi_error\t{exc}\n"
                            )
                            error_count += 1
                            print(
                                f"[ability3] error session={session_id} memory={memory_id} "
                                f"label={label} type={mem_type} error={exc}"
                            )
                            continue

                        parsed = _parse_model_output(raw or "")
                        if api_calls <= DEBUG_LOG_LIMIT:
                            log_handle.write(
                                f"resp\t{session_id}\t{memory_id}\t{label}\t{mem_type}\t{raw.strip()}\n"
                            )
                        if parsed is None:
                            error_handle.write(
                                f"{session_id}\t{memory_id}\tinvalid_output\t{(raw or '').strip()}\n"
                            )
                            error_count += 1
                            print(
                                f"[ability3] invalid session={session_id} memory={memory_id} "
                                f"label={label} type={mem_type}"
                            )
                            continue

                        if parsed.get("decision") == "DROP":
                            drop_count += 1
                            print(
                                f"[ability3] drop session={session_id} memory={memory_id} "
                                f"label={label} type={mem_type}"
                            )
                            continue

                        queries = parsed.get("queries") or []
                        if not isinstance(queries, list) or not queries:
                            error_handle.write(
                                f"{session_id}\t{memory_id}\tmissing_queries\t{(raw or '').strip()}\n"
                            )
                            error_count += 1
                            print(
                                f"[ability3] missing_queries session={session_id} memory={memory_id} "
                                f"label={label} type={mem_type}"
                            )
                            continue

                        output_entry = {
                            "session_id": parsed.get("session_id") or session_id,
                            "memory_id": parsed.get("memory_id") or memory_id,
                            "label": label,
                            "type": mem_type,
                            "queries": queries,
                        }
                        handle = _get_output_handle(label, mem_type, handles)
                        handle.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                        kept_count += 1
                        print(
                            f"[ability3] keep session={session_id} memory={memory_id} "
                            f"label={label} type={mem_type}"
                        )

                print(
                    f"[ability3] done file={path} records={record_count} kept={kept_count} "
                    f"dropped={drop_count} errors={error_count} api_calls={api_calls}"
                )
                log_handle.write(
                    f"done file={path} records={record_count} kept={kept_count} "
                    f"dropped={drop_count} errors={error_count} api_calls={api_calls}\n"
                )
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    filter_ability3()