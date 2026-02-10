# import json
# import os
# import sys
# import threading
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # Ensure src is in python path
# sys.path.append(os.getcwd())

# try:
#     from tqdm import tqdm
# except ImportError:
#     tqdm = lambda x, **kwargs: x

# from src.agents.shared import ensure_dir, append_jsonl, iter_jsonl
# from src.agents.task1_memory_extraction.curator import Task1Curator

# # ================= USER CONFIGURATION =================
# # 1. Input: Path to the existing evaluation report containing probes and model outputs
# INPUT_FILE = r"runs/task1_eval_human_annotation_deepseek-reasoner/evaluation_reports.jsonl"

# # 2. Output: Directory to save the new scored reports
# OUTPUT_DIR = r"runs/task1_eval_human_annotation_deepseek-reasoner"

# # 3. Concurrency
# MAX_WORKERS = 100 
# # ======================================================

# def process_recurate(record, curator, api_config):
#     """
#     Re-scores a record using existing probe and model report.
#     """
#     try:
#         probe = record.get("probe")
#         report = record.get("report")
        
#         if not probe or not report:
#             print(f"Skipping record {record.get('record_id')}: Missing probe or report.")
#             return None

#         # Call curator with use_llm=True
#         new_score = curator.score_report(
#             probe,
#             report,
#             use_llm=True,  # IMPORTANT: Enable LLM judge
#             matcher="greedy",
#             api_config=api_config
#         )

#         # Update the record with the new score
#         record["score"] = new_score
#         return record

#     except Exception as e:
#         print(f"Error curating record {record.get('record_id', 'unknown')}: {e}")
#         return None

# def main():
#     ensure_dir(OUTPUT_DIR)
#     config_path = "configs/api.json"
    
#     print(f"Loading config from {config_path}...")
#     try:
#         with open(config_path, "r", encoding="utf-8") as f:
#             api_config = json.load(f)
#     except FileNotFoundError:
#         print("Error: configs/api.json not found.")
#         return

#     print(f"Re-Curating (Scoring) Task 1 from: {INPUT_FILE}")
#     print(f"Saving to: {OUTPUT_DIR}")
    
#     curator = Task1Curator()
    
#     # Load records
#     records = list(iter_jsonl(INPUT_FILE))
#     print(f"Loaded {len(records)} records. Processing with {MAX_WORKERS} workers...")

#     output_path = os.path.join(OUTPUT_DIR, "evaluation_reports_recurated.jsonl")
#     if os.path.exists(output_path):
#         os.remove(output_path)

#     write_lock = threading.Lock()
    
#     def safe_append(res):
#         with write_lock:
#             append_jsonl(output_path, res)

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {
#             executor.submit(process_recurate, rec, curator, api_config): rec
#             for rec in records
#         }

#         for future in tqdm(as_completed(futures), total=len(records), desc="Recurating"):
#             res = future.result()
#             if res:
#                 safe_append(res)

#     print("-" * 30)
#     print("Recuration finished!")
#     print(f"New results saved to: {output_path}")

# if __name__ == "__main__":
#     main()


import json
import os
import sys
import threading
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure src is in python path
sys.path.append(os.getcwd())

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from src.agents.shared import ensure_dir, append_jsonl, iter_jsonl
from src.agents.task1_memory_extraction.curator import Task1Curator
input_str = "lightmem"
# ================= USER CONFIGURATION =================
# 1. Input: Path to the existing evaluation report containing probes and model outputs
INPUT_FILE = r"memory_system_mem/human/" + input_str + "_human.json"

# 2. Output: Directory to save the new scored reports
OUTPUT_DIR = r"memory_system_mem/human"

# 3. Concurrency
MAX_WORKERS = 20  # 稍微调高并发
# ======================================================

def process_recurate(record, curator, api_config):
    """
    针对 evermemos_mem.json 的特定结构进行评分。
    结构假设: 
      - record["ground_truth_memories"] -> 标准答案 list
      - record["memory_items"] -> 模型预测 list
    """

    # === 1. 构建 Probe (包含 Ground Truth) ===
    # Task1Curator.score_report 需要 probe["memory_items"] 作为 Ground Truth
    # 如果您的数据中 ground_truth_memories 是标准答案:
    gt_memories = record.get("ground_truth_memories")
    
    if gt_memories is None:
        # Fallback check
        print(f"Skipping record: Missing 'ground_truth_memories'. Keys found: {list(record.keys())}")
        return None

    # 构建符合 Curator 预期的 probe 结构
    # Curator 内部逻辑: ground_truth = probe.get("memory_items", [])
    probe = {
        "ground_truth_memories": gt_memories

    }

    # === 2. 提取 Report (模型输出) ===
    # 假设 memory_items 是模型提取出来的结果
    report = record.get("memory_items")
    fake_report_structure = {
            "runs": [
                {
                    "memory_items": report 
                }
            ]
        }
    if report is None:
        print(f"Skipping record: Missing 'memory_items' (Model Output).")
        return None

    # === 3. Call Curator ===
    new_score = curator.score_report(
        probe,
        fake_report_structure, # 模型输出的列表
        use_llm=False,
        matcher="greedy", 
        api_config=api_config
    )

    # Update the record with the new score
    new_record = copy.deepcopy(record)
    new_record["score"] = new_score
    
    return new_record

    
def main():
    ensure_dir(OUTPUT_DIR)
    config_path = "configs/api.json"
    
    print(f"Loading config from {config_path}...")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            api_config = json.load(f)
    except FileNotFoundError:
        print("Error: configs/api.json not found.")
        return

    print(f"Re-Curating (Scoring) Task 1 from: {INPUT_FILE}")
    print(f"Saving to: {OUTPUT_DIR}")
    
    curator = Task1Curator()
    
    # Load records (JSON array support)
    records = []
    if INPUT_FILE.endswith(".json"):
        try:
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                else:
                    # 如果不是列表，可能是单个对象
                    records = [data]
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file (might be JSONL?): {e}")
            # Fallback to iter_jsonl if simple load fails
            records = list(iter_jsonl(INPUT_FILE))
    else:
        # 默认作为 JSONL 处理
        records = list(iter_jsonl(INPUT_FILE))

    print(f"Loaded {len(records)} records. Processing with {MAX_WORKERS} workers...")

    output_path = os.path.join(OUTPUT_DIR, input_str + "_curated.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)

    write_lock = threading.Lock()
    
    def safe_append(res):
        with write_lock:
            append_jsonl(output_path, res)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_recurate, rec, curator, api_config): rec
            for rec in records
        }

        for future in tqdm(as_completed(futures), total=len(records), desc="Recurating"):
            res = future.result()
            if res:
                safe_append(res)

    print("-" * 30)
    print("Recuration finished!")
    print(f"New results saved to: {output_path}")

if __name__ == "__main__":
    main()