import json
import os
import sys
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# 确保能导入 src
sys.path.append(os.getcwd())

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from src.agents.shared import call_model_api, iter_jsonl, list_jsonl_files, ensure_dir, append_jsonl
from src.agents.task2_memory_update.evaluator import Task2Evaluator
from src.agents.task2_memory_update.curator import Task2Curator

# ================= 默认配置 =================
DEFAULT_INPUT_DIR = r"data/task2/"
DEFAULT_OUTPUT_DIR = r"runs/task2_results"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_WORKERS = 50

DEFAULT_TASK2_QUERY = (
    "You are a Memory Update System.\n"
    "Your task is to merge new information from the recent dialogue into the existing memory list.\n"
    "Rules:\n"
    "1. ADD: If new facts appear, add them.\n"
    "2. UPDATE: If a fact changes, update the existing entry.\n"
    "3. DELETE: If a fact is explicitly negated, remove it.\n"
    "4. IGNORE: Trivial chat needs no change.\n\n"
    "Return the FINAL complete list of memories as a JSON object:\n"
    "{ \"memory_items\": [ ... ] }"
)
# ===============================================

def process_pipeline(entry, evaluator, curator, model_list, api_config):
    record = entry.get("record", {}) or {}
    
    query = entry.get("query", "") or ""
    if not query:
        query = DEFAULT_TASK2_QUERY

    gt_memories = entry.get("answer", []) or []

    probe = {
        "task": "memory_update",
        "old_dialogue": record.get("old_dialogue", []) or [],
        "new_dialogue": record.get("new_dialogue", []) or [],
        "old_memory_items": entry.get("memory", []) or [],
        "expected_updated_memory_items": gt_memories,
        "query": query,
        "metadata": entry.get("metadata", {}) or {},
    }

    report = evaluator.evaluate_models(
        probe, model_list, call_model=call_model_api, api_config=api_config
    )
    
    model_runs = report.get("runs", [])
    score_struct = {"runs": []}
    
    for run in model_runs:
        model_name = run.get("model", "unknown")
        pred_memories = run.get("memory_items", [])
        
        pred_record = {"memory_items": pred_memories}
        gold_record = {"memory_items": gt_memories}
        
        score_data = curator.score(
            gold_record, pred_record, use_llm=True, matcher="greedy", api_config=api_config
        )
        
        score_struct["runs"].append({
            "model": model_name,
            "score": score_data,
            "_meta": {"pred_count": len(pred_memories), "gold_count": len(gt_memories)}
        })

    return {"entry": entry, "report": report, "score": score_struct}

def print_summary(results):
    model_stats = {} 
    for res in results:
        for run in res.get("score", {}).get("runs", []):
            name = run["model"]
            metrics = run.get("score", {}).get("metrics", {}).get("coverage", {})
            f1 = metrics.get("f1", 0.0)
            if name not in model_stats: model_stats[name] = []
            model_stats[name].append(f1)
            
    print("\nPreliminary F1 Summary:")
    print("-" * 40)
    for model, scores in model_stats.items():
        avg_f1 = sum(scores) / len(scores) if scores else 0
        print(f"Model: {model:<20} | Avg F1: {avg_f1:.4f} | Count: {len(scores)}")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Run Task 2 Inference and Scoring Pipeline")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to evaluate")
    parser.add_argument("--data_path", type=str, default=DEFAULT_INPUT_DIR, help="Path to input dataset file or folder")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save results")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help="Max parallel workers")
    args = parser.parse_args()

    model_list = [args.model]
    safe_model_name = args.model.replace("/", "-")
    final_output_dir = f"{args.output_dir}_{safe_model_name}"
    
    ensure_dir(final_output_dir)
    config_path = "configs/api.json"
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            api_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} Not found.")
        return

    print("="*60)
    print("Starting Task 2 Full Pipeline (Evaluate + Curate)")
    print(f"Input Data: {args.data_path}")
    print(f"Evaluating Model: {model_list}")
    print("="*60)

    evaluator = Task2Evaluator()
    curator = Task2Curator()

    entries = []
    if os.path.isfile(args.data_path):
        paths = [args.data_path]
    else:
        paths = list_jsonl_files(args.data_path)
        
    for path in paths:
        for ent in iter_jsonl(path):
            entries.append(ent)
            
    print(f"Loaded {len(entries)} entries. Processing with {args.workers} workers...")
    
    output_file = os.path.join(final_output_dir, "scored.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)
        
    lock = threading.Lock()
    results = [] 
    
    def safe_handle_result(res):
        with lock:
            append_jsonl(output_file, res)
            results.append(res)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_pipeline, ent, evaluator, curator, model_list, api_config): ent 
            for ent in entries
        }
        for future in tqdm(as_completed(futures), total=len(entries), desc="Processing"):
            try:
                res = future.result()
                safe_handle_result(res)
            except Exception as e:
                print(f"Error processing entry: {e}")

    print("-" * 30)
    print(f"Pipeline Finished! Results saved to: {output_file}")
    if results:
        print_summary(results)

if __name__ == "__main__":
    main()