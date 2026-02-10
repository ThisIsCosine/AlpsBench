import json
import os
import sys
from typing import Iterable, List, Set

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

INPUT_ROOT = os.path.join("data", "wildchat", "memories", "selected", "task4_final")
OUTPUT_ROOT = os.path.join("data", "wildchat", "memories", "selected", "task4_final_flatten")
OUTPUT_FILE = os.path.join(OUTPUT_ROOT, "task4_final_flatten.jsonl")


def _iter_jsonl_paths(root_dir: str) -> Iterable[str]:
    for root, _dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".jsonl"):
                yield os.path.join(root, fname)


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _split_into_chunks(items: List[dict], chunks: int) -> List[List[dict]]:
    total = len(items)
    base = total // chunks
    extra = total % chunks
    result: List[List[dict]] = []
    start = 0
    for idx in range(chunks):
        size = base + (1 if idx < extra else 0)
        result.append(items[start : start + size])
        start += size
    return result


def run() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    seen_sessions: Set[str] = set()
    kept_records: List[dict] = []
    total_records = 0

    jsonl_paths = sorted(_iter_jsonl_paths(INPUT_ROOT))
    print(f"[scan] files={len(jsonl_paths)} root={INPUT_ROOT}")
    for path in jsonl_paths:
        file_total = 0
        file_kept = 0
        for record in _iter_jsonl(path):
            total_records += 1
            file_total += 1
            session_id = record.get("sessions", [{}])[0].get("session_id")
            if not session_id:
                # If no session_id, treat as unique to avoid dropping data.
                kept_records.append(record)
                file_kept += 1
                continue
            if session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)
            kept_records.append(record)
            file_kept += 1
        print(f"[file] {path} total={file_total} kept={file_kept}")

    print(f"[summary] total_records={total_records} kept_unique={len(kept_records)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as handle:
        for record in kept_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[write] flatten={OUTPUT_FILE} rows={len(kept_records)}")

    chunks = _split_into_chunks(kept_records, 4)
    for idx, chunk in enumerate(chunks, start=1):
        out_path = os.path.join(OUTPUT_ROOT, f"task4_final_flatten_part{idx}.jsonl")
        with open(out_path, "w", encoding="utf-8") as handle:
            for record in chunk:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[write] part{idx} rows={len(chunk)} path={out_path}")


if __name__ == "__main__":
    run()
