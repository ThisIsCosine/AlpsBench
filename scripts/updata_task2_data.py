import json
import os
import re
import sys
from collections import defaultdict

# ================= Configuration =================
# 1. Target Task 2 Dataset (The file to be updated)
TARGET_DATASET_FILE = r"data/task2_dataset.jsonl"

# 2. Source Directory (Human Annotation files containing the correct 'memory')
SOURCE_DIR = r"data/human_annotation_with_selected_memory_id/task4_final_human_annotation_with_selected_memory_id"

# 3. Output File (The new dataset)
OUTPUT_FILE = r"data/task2_dataset_updated_memory.jsonl"

# ================= Logic =================

def normalize_text(text):
    """
    Standardize text for matching: remove non-alphanumeric chars, lower case.
    This helps match dialogues even if whitespace/punctuation slightly differs.
    """
    if not text:
        return ""
    # Only keep letters and numbers
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def extract_signature_from_source(item):
    """
    Extracts user input from Source (Human Annotation) format to build a key.
    Format: item['dialogue'] -> list of turns
    """
    dialogue = item.get("dialogue", [])
    # Strategy: Concatenate the first User turn, or the first few chars of the whole dialog
    # Let's try finding the first user turn.
    for turn in dialogue:
        if turn.get("role") == "user":
            return normalize_text(turn.get("text", ""))
    
    # Fallback: if no user turn found, use first turn text
    if dialogue:
        return normalize_text(dialogue[0].get("text", ""))
    return None

def extract_signature_from_target(item):
    """
    Extracts user input from Target (Task 2 Dataset) format to build a key.
    Format: item['record']['old_dialogue'] -> list of turns
    """
    record = item.get("record", {})
    old_dialogue = record.get("old_dialogue", [])
    
    # Same strategy as source
    for turn in old_dialogue:
        if turn.get("role") == "user":
            return normalize_text(turn.get("text", ""))
            
    if old_dialogue:
        return normalize_text(old_dialogue[0].get("text", ""))
    return None

def list_jsonl_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".jsonl"):
                files.append(os.path.join(root, filename))
    return files

def main():
    print(f"Loading Source Data from: {SOURCE_DIR}")
    
    source_map = defaultdict(list)
    source_files = list_jsonl_files(SOURCE_DIR)
    total_source = 0
    
    # 1. Build Lookup Map from Source
    for file_path in source_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    sig = extract_signature_from_source(data)
                    if sig:
                        # Store the memory items we want to inject later
                        # Check keys 'memory_items' or 'memories'
                        memories = data.get("memory_items") or data.get("memories") or []
                        source_map[sig].append({
                            "memories": memories,
                            "file": os.path.basename(file_path),
                            "id": data.get("session_id") or data.get("line_index")
                        })
                        total_source += 1
                except Exception as e:
                    print(f"Error parsing source line: {e}")

    print(f"Loaded {total_source} source records. Unique signatures: {len(source_map)}")

    # 2. Process Target File
    print(f"Processing Target File: {TARGET_DATASET_FILE}")
    
    updated_records = []
    match_count = 0
    missing_count = 0
    conflict_count = 0
    
    with open(TARGET_DATASET_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} target records. Updating...")

    report_conflicts = []

    for i, line in enumerate(lines):
        if not line.strip(): continue
        item = json.loads(line)
        
        sig = extract_signature_from_target(item)
        
        matches = source_map.get(sig)
        
        if not matches:
            # Case: No match found
            # print(f"Warning: No match found for record {i}") 
            missing_count += 1
            updated_records.append(item) # Keep original
            continue
        
        # Case: One-to-Many Conflict
        if len(matches) > 1:
            conflict_count += 1
            # Check if memories are actually different
            first_mem = json.dumps(matches[0]["memories"], sort_keys=True)
            is_real_conflict = False
            for m in matches[1:]:
                if json.dumps(m["memories"], sort_keys=True) != first_mem:
                    is_real_conflict = True
                    break
            
            if is_real_conflict:
                report_conflicts.append({
                    "target_index": i,
                    "signature_start": sig[:50],
                    "match_candidates": matches
                })
        
        # Action: Update Memory
        # We take the first match's memory
        new_memory = matches[0]["memories"]
        
        # Task 2 formatting: update 'memory' field at root AND 'old_memory_items' inside record if it exists
        # NOTE: Your task_evaluators.py reads: probe["old_memory_items"] = entry.get("memory", [])
        
        item["memory"] = new_memory
        
        # Optionally update internal record if structure requires it, mostly entry["memory"] is key
        if "record" in item:
            item["record"]["old_memory_items"] = new_memory

        updated_records.append(item)
        match_count += 1

    # 3. Write Output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in updated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 4. Report
    print("=" * 40)
    print(f"Processing Complete.")
    print(f"Total Target Records: {len(lines)}")
    print(f"Successfully Updated: {match_count}")
    print(f"Missing Source Matches: {missing_count} (Kept original)")
    print(f"Signatue Conflicts    : {conflict_count}")
    print(f"Output saved to       : {OUTPUT_FILE}")
    print("=" * 40)

    if report_conflicts:
        print("\n[CONFLICT REPORT - One Signature maps to Multiple Sources]")
        print("Note: If the content is identical, we just picked the first one. Below are potential mismatches.")
        for conflict in report_conflicts[:10]: # Print first 10
            print(f"\n--- Target Index {conflict['target_index']} ---")
            print(f"Sig: {conflict['signature_start']}...")
            for idx, cand in enumerate(conflict['match_candidates']):
                print(f"  Candidate {idx+1}: File={cand['file']}, ID={cand['id']}, Memory Count={len(cand['memories'])}")

if __name__ == "__main__":
    main()