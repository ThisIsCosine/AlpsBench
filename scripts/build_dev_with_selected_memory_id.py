import json
from collections import defaultdict
from pathlib import Path

# I/O directory configuration (relative to repo root)
# Modify these if your layout changes.
CLEANED_DIR = Path("data/wildchat/cleaned")
SELECTED_DIR = Path("data/wildchat/memories/selected/task4")
OUTPUT_DIR = Path("benchmark/dev_with_selected_memory_id")


def build_selected_index(selected_path: Path):
    """Index a selected.jsonl file by (session_id, memory_id) -> parsed record."""
    index = {}
    duplicate_keys = defaultdict(int)
    with selected_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.rstrip("\n")
            if not raw:
                continue
            record = json.loads(raw)
            # A selected record can contain multiple sessions and memory items.
            session_ids = [
                session.get("session_id")
                for session in record.get("sessions", [])
                if session.get("session_id")
            ]
            memory_ids = set()
            for item in record.get("memory_items", []):
                memory_id = item.get("memory_id")
                if memory_id:
                    memory_ids.add(memory_id)
            for item in record.get("memory_stage1_candidates", []):
                memory_id = item.get("memory_id")
                if memory_id:
                    memory_ids.add(memory_id)
            if not session_ids or not memory_ids:
                continue
            for session_id in session_ids:
                for memory_id in memory_ids:
                    key = (session_id, memory_id)
                    if key in index:
                        duplicate_keys[key] += 1
                        continue
                    index[key] = record
    return index, duplicate_keys


def main():
    repo_root = Path(__file__).resolve().parents[1]
    cleaned_root = repo_root / CLEANED_DIR
    selected_root = repo_root / SELECTED_DIR
    output_root = repo_root / OUTPUT_DIR

    if not cleaned_root.exists():
        raise FileNotFoundError(f"Missing cleaned root: {cleaned_root}")
    if not selected_root.exists():
        raise FileNotFoundError(f"Missing selected root: {selected_root}")

    # Discover all cleaned jsonl files (each contains session_id and memory_id).
    cleaned_files = sorted(cleaned_root.rglob("*.jsonl"))
    if not cleaned_files:
        raise FileNotFoundError(f"No jsonl files under {cleaned_root}")

    # Tracking counters and anomalies.
    total_pairs = 0
    total_written = 0
    missing_pairs = []
    missing_selected_files = []
    duplicate_summary = defaultdict(int)

    for cleaned_path in cleaned_files:
        rel_path = cleaned_path.relative_to(cleaned_root)
        selected_path = selected_root / rel_path
        output_path = output_root / rel_path

        # Skip cleaned files that do not exist in selected.
        if not selected_path.exists():
            missing_selected_files.append(str(rel_path))
            continue

        # Build per-file index to map (session_id, memory_id) -> full record.
        selected_index, duplicate_keys = build_selected_index(selected_path)
        for key, count in duplicate_keys.items():
            duplicate_summary[key] += count

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with cleaned_path.open("r", encoding="utf-8") as cleaned_handle, output_path.open(
            "w", encoding="utf-8"
        ) as output_handle:
            for line in cleaned_handle:
                raw = line.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                session_id = record.get("session_id")
                memory_id = record.get("memory_id")
                total_pairs += 1
                if not session_id or not memory_id:
                    missing_pairs.append(
                        {"file": str(rel_path), "session_id": session_id, "memory_id": memory_id}
                    )
                    continue
                key = (session_id, memory_id)
                selected_record = selected_index.get(key)
                if not selected_record:
                    missing_pairs.append(
                        {"file": str(rel_path), "session_id": session_id, "memory_id": memory_id}
                    )
                    continue
                # Add selected memory id to the full record.
                selected_record = dict(selected_record)
                selected_record["selected_memory_id"] = memory_id
                output_handle.write(json.dumps(selected_record, ensure_ascii=False) + "\n")
                total_written += 1

    print("Done.")
    print(f"Total pairs in cleaned: {total_pairs}")
    print(f"Total records written: {total_written}")
    if missing_selected_files:
        print(f"Missing selected files: {len(missing_selected_files)}")
        for rel_path in missing_selected_files[:10]:
            print(f"  - {rel_path}")
        if len(missing_selected_files) > 10:
            print("  - ...")
    if missing_pairs:
        print(f"Missing pairs: {len(missing_pairs)} (showing first 10)")
        for item in missing_pairs[:10]:
            print(
                f"  - {item['file']} :: session_id={item['session_id']} memory_id={item['memory_id']}"
            )
    if duplicate_summary:
        print(f"Duplicate keys in selected: {len(duplicate_summary)}")


if __name__ == "__main__":
    main()
