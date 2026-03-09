import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# 确保能导入 src
sys.path.append(os.getcwd())

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from src.agents.shared import call_model_api, iter_jsonl, list_jsonl_files, ensure_dir, append_jsonl
from src.agents.task3_memory_retrieval.evaluator import Task3Evaluator
from src.agents.task3_memory_retrieval.curator import Task3Grader

# ================= 用户配置区域 =================
# 1. 输入数据文件 (由 run_task3_distractor 生成的最终数据集)
DEFAULT_DATA_PATH = r"data/task3/"
DEFAULT_DISTRACTORS = 100
DEFAULT_MODEL = "gpt-4o"
DEFAULT_OUTPUT_DIR = r"runs/task3_results"
DEFAULT_MAX_WORKERS = 100
USE_LLM_JUDGE = True
# ===============================================

def process_pipeline(entry, evaluator, grader, model_list, api_config):
    """
    单条处理流水线：
    1. Parse Entry -> Probe (Task 3 数据集中的 record 本身就是 probe 结构)
    2. Evaluator -> Model Output (Selection)
    3. Grader -> Score (Accuracy)
    """
    probe = entry.get("record", {})
    if not probe:
        # 兼容旧格式或直接 probe 列表
        probe = entry 
        
    # 简单校验
    if "query" not in probe or "candidate_memories" not in probe:
        print("no candidite")
        return {"entry": entry, "error": "invalid_format_missing_fields"}

    try:
        report = evaluator.evaluate_models(
            probe,
            model_list,
            call_model=call_model_api,
            use_judge=True,
            api_config=api_config,
            get_usage = True
        )
    except Exception as e:
        return {"entry": entry, "error": f"evaluator_failed: {str(e)}"}
    
    try:
        score = grader.score_report(
            probe,
            report,
            use_llm=USE_LLM_JUDGE,
            api_config=api_config
        )
    except Exception as e:
        return {"entry": entry, "report": report, "error": f"grader_failed: {str(e)}"}
    
    # 返回完整结果
    return {
        "entry": entry,
        "report": report,
        "score": score
    }

def main():


    parser = argparse.ArgumentParser(description="Run Task 3 Full Pipeline (Evaluate + Grade)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to evaluate")
    parser.add_argument("--distractors", type=int, default=DEFAULT_DISTRACTORS, help="Number of distractors (e.g., 100, 300, 1000)")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Base directory for task 3 data")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Base output directory")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help="Max parallel workers")
    args = parser.parse_args()

    model_list = [args.model]

    input_file = os.path.join(args.data_path, f"task3_dataset_d{args.distractors}.jsonl")
    final_output_dir = f"{args.output_dir}_d{args.distractors}_{model_list[0]}"

    ensure_dir(final_output_dir)
    config_path = "configs/api.json"
    
    print(f"Loading config from {config_path}...")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            api_config = json.load(f)
    except FileNotFoundError:
        print("Error: configs/api.json Not found.")
        return

    print("="*60)
    print("Starting Task 3 Full Pipeline (Evaluate + Grade)")
    print(f"Input: {input_file}")
    print(f"Models: {model_list}")
    print(f"Distractors: {args.distractors}")
    print("="*60)

    evaluator = Task3Evaluator()
    grader = Task3Grader()

    # 加载所有数据
    entries = []
    if os.path.isfile(input_file):
        paths = [input_file]
    elif os.path.isdir(input_file):
        paths = list_jsonl_files(input_file)
    else:
        print(f"Error: Input path {input_file} not found. Ensure the dataset for d{args.distractors} exists.")
        return
        
    for path in paths:
        for ent in iter_jsonl(path):
            entries.append(ent)
            
    print(f"Loaded {len(entries)} entries. Processing with {args.workers} workers...")
    
    output_filename = f"scored.jsonl"
    output_file = os.path.join(final_output_dir, output_filename)
    
    if os.path.exists(output_file):
        print("exist files")
        return
        
    lock = threading.Lock()
    results = [] 
    
    def safe_handle_result(res):
        with lock:
            append_jsonl(output_file, res)
            if "error" not in res:
                results.append(res)
            else:
                pass

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_pipeline, ent, evaluator, grader, model_list, api_config): ent 
            for ent in entries
        }
        
        for future in tqdm(as_completed(futures), total=len(entries), desc="Evaluating"):
            try:
                res = future.result()
                safe_handle_result(res)
            except Exception as e:
                print(f"Unhandled exception in thread: {e}")

    print("-" * 30)
    print(f"Pipeline Finished!")
    print(f"All results saved to: {output_file}")
    
    if results:
        print_summary(results)
    else:
        print("No successful results to summarize.")

def print_summary(results):
    model_stats = {} 
    
    for res in results:
        score_data = res.get("score", {})
        runs = score_data.get("runs", [])
        
        for run in runs:
            model_name = run.get("model", "unknown")
            judge_info = run.get("judge", {})
            

            score_val = judge_info.get("score", 0.0)
            
            if model_name not in model_stats: 
                model_stats[model_name] = {"total_score": 0.0, "count": 0, "hits": 0}
            
            model_stats[model_name]["total_score"] += score_val
            model_stats[model_name]["count"] += 1
            if score_val > 0.5: 
                model_stats[model_name]["hits"] += 1
            
    print("\n=== Evaluation Summary ===")
    print(f"{'Model':<25} | {'Avg Score':<10} | {'Accuracy':<10} | {'Count':<8}")
    print("-" * 65)
    for model, stats in model_stats.items():
        count = stats["count"]
        if count == 0: continue
        avg_score = stats["total_score"] / count
        accuracy = stats["hits"] / count
        print(f"{model:<25} | {avg_score:.4f}     | {accuracy:.2%}     | {count:<8}")
    print("-" * 65)

if __name__ == "__main__":
    main()