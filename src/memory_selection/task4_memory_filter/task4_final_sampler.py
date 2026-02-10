import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

INPUT_ROOT = os.path.join("data", "wildchat", "memories", "selected", "task4")
OUTPUT_ROOT = os.path.join("data", "wildchat", "memories", "selected", "task4_final")

# Per-label targets. None means keep all for that label.
TARGETS: Dict[str, Optional[Dict[str, Optional[int]]]] = {
    "ability1": {
        "Personal_Background/Education": 55,
        "Personal_Background/Family": 50,
        "Personal_Background/Identity": 55,
        "Personal_Background/Location": 55,
        "Personal_Background/Occupation": 55,
        "Social_Relationships/Colleagues": 30,
        "Social_Relationships/Family": 54,
        "Social_Relationships/Friends": 54,
        "Social_Relationships/Partners": 54,
        "States_Experiences/Past_Experience": 54,
        "Thoughts/Goals/Long_Term": 54,
    },
    # Keep all for ability2_general
    "ability2_general": None,
    "ability2_interaction": {
        "Preferences/Interaction_Preferences": 250,
        "interaction/style/preference": 250,
    },
    "ability3": {
        "role-play": 500,
    },
    "ability4": {
        "Constraints_and_Boundaries/Communication_Preferences": 100,
        "Constraints_and_Boundaries/Disliked_Topics": 100,
        "Constraints_and_Boundaries/Do_Not_Remember": 100,
        "Constraints_and_Boundaries/Interaction_Preferences": 100,
        "Constraints_and_Boundaries/Sensitive_Topics": 100,
    },
}


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _dialogue_turns_count(record: dict) -> int:
    sessions = record.get("sessions") or []
    total = 0
    for session in sessions:
        turns = session.get("turns") or []
        total += len(turns)
    return total


def _label_from_path(root_dir: str, file_path: str) -> str:
    rel_path = os.path.relpath(file_path, root_dir)
    parts = rel_path.split(os.sep)
    if len(parts) < 2:
        return ""
    # Drop the last segment (direct.jsonl/indirect.jsonl)
    label_parts = parts[:-1]
    return "/".join(label_parts)


def _collect_label_files(ability_dir: str) -> Dict[str, Dict[str, str]]:
    label_files: Dict[str, Dict[str, str]] = {}
    for root, _dirs, files in os.walk(ability_dir):
        for fname in files:
            if fname not in {"direct.jsonl", "indirect.jsonl"}:
                continue
            file_path = os.path.join(root, fname)
            label = _label_from_path(ability_dir, file_path)
            if not label:
                continue
            label_files.setdefault(label, {})[fname.replace(".jsonl", "")] = file_path
    return label_files


def _write_jsonl(path: str, records: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _sample_label(
    label: str,
    label_files: Dict[str, str],
    target: Optional[int],
    output_root: str,
) -> Tuple[int, int]:
    indirect_path = label_files.get("indirect")
    direct_path = label_files.get("direct")
    indirect_records: List[dict] = []
    direct_records: List[Tuple[int, dict]] = []

    if indirect_path and os.path.exists(indirect_path):
        indirect_records = list(_iter_jsonl(indirect_path))

    if target is None:
        remaining = None
    else:
        if len(indirect_records) > target:
            indirect_records = indirect_records[:target]
            remaining = 0
        else:
            remaining = max(target - len(indirect_records), 0)

    if direct_path and os.path.exists(direct_path):
        if remaining is None or remaining > 0:
            for record in _iter_jsonl(direct_path):
                if remaining is None:
                    direct_records.append((0, record))
                else:
                    direct_records.append((_dialogue_turns_count(record), record))

    if remaining is None:
        chosen_direct = [record for _score, record in direct_records]
    else:
        direct_records.sort(key=lambda item: item[0], reverse=True)
        chosen_direct = [record for _score, record in direct_records[:remaining]]

    output_label_root = os.path.join(output_root, *label.split("/"))
    indirect_out = os.path.join(output_label_root, "indirect.jsonl")
    direct_out = os.path.join(output_label_root, "direct.jsonl")

    kept_indirect = _write_jsonl(indirect_out, indirect_records) if indirect_records else 0
    kept_direct = _write_jsonl(direct_out, chosen_direct) if chosen_direct else 0
    return kept_indirect, kept_direct


def run() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for ability, target_map in TARGETS.items():
        ability_dir = os.path.join(INPUT_ROOT, ability)
        if not os.path.exists(ability_dir):
            print(f"[skip] ability={ability} missing input dir: {ability_dir}")
            continue

        label_files_map = _collect_label_files(ability_dir)
        if not label_files_map:
            print(f"[skip] ability={ability} no label files found.")
            continue

        print(f"[ability] {ability} labels={len(label_files_map)}")
        for label, files in sorted(label_files_map.items()):
            if target_map is None:
                target = None
            else:
                target = target_map.get(label)
                if target is None:
                    print(f"[warn] ability={ability} label={label} target missing, keep all")
            kept_indirect, kept_direct = _sample_label(
                label,
                files,
                target,
                os.path.join(OUTPUT_ROOT, ability),
            )
            total = kept_indirect + kept_direct
            target_text = "all" if target is None else str(target)
            print(
                f"[done] ability={ability} label={label} target={target_text} "
                f"kept={total} (indirect={kept_indirect}, direct={kept_direct})"
            )


if __name__ == "__main__":
    run()
