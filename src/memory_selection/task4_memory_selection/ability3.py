"""Ability 3: Real-virtual distinguishing ability.
    [All the record has candidate memories and the value / reasoning of the candidate memories contains 'role-play' judgement from the 60000 dataset]
    role-play: 774，

"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterable, Set, TextIO, Tuple


INPUT_DIR = (
    "data/wildchat/memories/stage1_raw/120000 memories/"
    "raw memories {dialogue, intention, memory} categoried by turns"
)
OUTPUT_DIR = "data/wildchat/memories/selected/task4/ability3"


ROLE_PLAY_DIR = os.path.join(OUTPUT_DIR, "role-play")


def iter_records(file_path: str) -> Iterable[dict]:
    with open(file_path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    for record in payload.get("data", []):
        yield record


ROLE_PLAY_PATTERNS = ("role-play", "role play", "roleplay")


def _normalize_memory_type(value: str | None) -> str | None:
    if not value:
        return None
    mem_type = str(value).strip().lower()
    if mem_type in {"direct", "indirect"}:
        return mem_type
    return None


def _role_play_types(record: dict) -> Set[str]:
    candidates = record.get("memory_stage1_candidates") or []
    if not candidates:
        return set()
    matched_types: Set[str] = set()
    for item in candidates:
        value = str(item.get("value") or "").lower()
        reasoning = str(item.get("reasoning") or "").lower()
        if any(pat in value for pat in ROLE_PLAY_PATTERNS) or any(
            pat in reasoning for pat in ROLE_PLAY_PATTERNS
        ):
            mem_type = _normalize_memory_type(item.get("type"))
            if mem_type:
                matched_types.add(mem_type)
    return matched_types


def _type_output_path(mem_type: str) -> str:
    return os.path.join(ROLE_PLAY_DIR, f"{mem_type}.jsonl")


def extract_records() -> None:
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.merged.json")))
    if not input_files:
        raise RuntimeError(f"No input files found under: {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ROLE_PLAY_DIR, exist_ok=True)
    handles: dict[str, TextIO] = {}

    try:
        counts: Dict[str, int] = {}
        for file_path in input_files:
            for record in iter_records(file_path):
                matched_types = _role_play_types(record)
                if not matched_types:
                    continue
                for mem_type in matched_types:
                    label = "role-play"
                    counts[label] = counts.get(label, 0) + 1
                    if mem_type not in handles:
                        handles[mem_type] = open(
                            _type_output_path(mem_type), "w", encoding="utf-8"
                        )
                    handles[mem_type].write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
        
        print("\nExtraction summary (matches docstring format):")
        sorted_labels = sorted(counts.keys())
        for label in sorted_labels:
            print(f"    {label}: {counts[label]},")
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    extract_records()
