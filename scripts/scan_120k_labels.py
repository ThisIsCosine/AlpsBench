import glob
import json
import os
from collections import Counter

INPUT_DIR = (
    "data/wildchat/memories/stage1_raw/120000 memories/"
    "raw memories {dialogue, intention, memory} categoried by turns"
)

def scan_labels():
    all_labels = Counter()
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.merged.json")))
    
    if not input_files:
        print(f"No files found in {INPUT_DIR}")
        return

    print(f"Scanning {len(input_files)} files...")
    for file_path in input_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                for record in payload.get("data", []):
                    for item in record.get("memory_items", []):
                        label = item.get("label")
                        if label:
                            all_labels[label] += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print("\n--- All Labels Found ---")
    for label, count in all_labels.most_common():
        print(f"{label}: {count}")

    print("\n--- Interaction-related Labels ---")
    for label, count in all_labels.most_common():
        if "interaction" in label.lower():
            print(f"{label}: {count}")

if __name__ == "__main__":
    scan_labels()
