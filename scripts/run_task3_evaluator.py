import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
INPUT_FILE = r"data/task3_dataset_d1000.jsonl"

# 2. 设置评测模型
MODEL_LIST = ["meta-llama/llama-4-maverick"] 

# 3. 结果输出目录
OUTPUT_DIR = r"runs/task3_d1000_results_llama-4-mavericktoken_consume"

# 4. 并发线程数
MAX_WORKERS = 100

# 5. Grader 配置 (参照 curator 逻辑，通常使用规则匹配快速打分，use_llm=False)
USE_LLM_JUDGE = True
# ===============================================

def process_pipeline(entry, evaluator, grader, model_list, api_config):
    """
    单条处理流水线：
    1. Parse Entry -> Probe (Task 3 数据集中的 record 本身就是 probe 结构)
    2. Evaluator -> Model Output (Selection)
    3. Grader -> Score (Accuracy)
    """
    # 1. 提取 Probe
    # Task 3 数据集的结构通常是 {"seed_id": "...", "record": { "query":..., "candidate_memories":... }}
    probe = entry.get("record", {})
    if not probe:
        # 兼容旧格式或直接 probe 列表
        probe = entry 
        
    # 简单校验
    if "query" not in probe or "candidate_memories" not in probe:
        print("no candidite")
        return {"entry": entry, "error": "invalid_format_missing_fields"}

    # 2. 运行模型 (Evaluator)
    # evaluate_models 返回 report
    # 参照 curator.py 中的调用方式:
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
    
    # 3. 运行评分 (Grader)
    # 参照 curator.py 中的调用方式:
    # score = grader.score_report(probe, report, use_llm=False, api_config=local_api_config)
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
    ensure_dir(OUTPUT_DIR)
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
    print(f"Input: {INPUT_FILE}")
    print(f"Models: {MODEL_LIST}")
    print("="*60)

    evaluator = Task3Evaluator()
    grader = Task3Grader()

    # 加载所有数据
    entries = []
    if os.path.isfile(INPUT_FILE):
        paths = [INPUT_FILE]
    elif os.path.isdir(INPUT_FILE):
        paths = list_jsonl_files(INPUT_FILE)
    else:
        print(f"Error: Input path {INPUT_FILE} not found.")
        return
        
    for path in paths:
        for ent in iter_jsonl(path):
            entries.append(ent)
            
    print(f"Loaded {len(entries)} entries. Processing with {MAX_WORKERS} workers...")
    
    output_filename = f"scored.jsonl"
    output_file = os.path.join(OUTPUT_DIR, output_filename)
    
    if os.path.exists(output_file):
        print("exist files")
        return
        # os.remove(output_file)
        
    lock = threading.Lock()
    results = [] 
    
    def safe_handle_result(res):
        with lock:
            append_jsonl(output_file, res)
            if "error" not in res:
                results.append(res)
            else:
                # 可以选择记录错误日志
                pass

    # 并行执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_pipeline, ent, evaluator, grader, MODEL_LIST, api_config): ent 
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
            
            # Task 3 Grader 返回的 judge 结构: {"used_memory": bool, "score": float, ...}
            # score 通常是 0.0 或 1.0 (overlap score) 或 [0,1]
            score_val = judge_info.get("score", 0.0)
            
            if model_name not in model_stats: 
                model_stats[model_name] = {"total_score": 0.0, "count": 0, "hits": 0}
            
            model_stats[model_name]["total_score"] += score_val
            model_stats[model_name]["count"] += 1
            if score_val > 0.5: # 简单的 Accuracy 阈值
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