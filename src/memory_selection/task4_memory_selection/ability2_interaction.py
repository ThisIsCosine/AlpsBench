"""Ability 2: Interaction preferences following ability.

    60000 dataset:
    Preferences/Interaction_Preferences: 5936,
    interaction/style/preference: 1,

"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterable, Set, TextIO, Tuple


INPUT_DIR = (
    "data/wildchat/memories/stage1_raw/120000 memories/"
    "raw memories {dialogue, intention, memory} categoried by turns"
)
OUTPUT_DIR = "data/wildchat/memories/selected/task4/ability2_interaction"


def parse_labels_from_docstring() -> Set[str]:
    doc = __doc__ or ""
    labels: Set[str] = set()
    for raw_line in doc.splitlines():
        line = raw_line.strip().strip('"').strip("'")
        if "/" not in line:
            continue
        label = line.split(":", 1)[0].strip().rstrip(",")
        if "/" in label:
            labels.add(label)
    return labels


def iter_records(file_path: str) -> Iterable[dict]:
    with open(file_path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    for record in payload.get("data", []):
        yield record


def _normalize_memory_type(value: str | None) -> str | None:
    if not value:
        return None
    mem_type = str(value).strip().lower()
    if mem_type in {"direct", "indirect"}:
        return mem_type
    return None


def label_type_to_path(label: str, mem_type: str) -> str:
    parts = [part.strip() for part in label.split("/") if part.strip()]
    return os.path.join(OUTPUT_DIR, *parts, f"{mem_type}.jsonl")


def get_label_file(
    label: str,
    mem_type: str,
    open_files: Dict[str, TextIO],
) -> TextIO:
    key = f"{label}::{mem_type}"
    if key in open_files:
        return open_files[key]
    file_path = label_type_to_path(label, mem_type)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_handle = open(file_path, "w", encoding="utf-8")
    open_files[key] = file_handle
    return file_handle


def extract_records() -> None:
    target_labels = parse_labels_from_docstring()
    if not target_labels:
        raise RuntimeError("No labels found in docstring.")

    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.merged.json")))
    if not input_files:
        raise RuntimeError(f"No input files found under: {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    open_files: Dict[str, TextIO] = {}

    try:
        counts: Dict[str, int] = {}
        for file_path in input_files:
            for record in iter_records(file_path):
                memory_items = record.get("memory_items") or []
                record_targets: Set[Tuple[str, str]] = set()
                for item in memory_items:
                    label = item.get("label")
                    if label not in target_labels:
                        continue
                    mem_type = _normalize_memory_type(item.get("type"))
                    if not mem_type:
                        continue
                    record_targets.add((label, mem_type))

                for label, mem_type in record_targets:
                    counts[label] = counts.get(label, 0) + 1
                    file_handle = get_label_file(label, mem_type, open_files)
                    file_handle.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
        
        print("\nExtraction summary (matches docstring format):")
        sorted_labels = sorted(counts.keys())
        for label in sorted_labels:
            print(f"    {label}: {counts[label]},")
    finally:
        for file_handle in open_files.values():
            file_handle.close()


if __name__ == "__main__":
    extract_records()
