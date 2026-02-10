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

from src.agents.shared import call_model_api, iter_jsonl, list_jsonl_files, ensure_dir, append_jsonl, resolve_session_id
from src.agents.task2_memory_update.evaluator import Task2Evaluator
from src.agents.task2_memory_update.curator import Task2Curator

# ================= 用户配置区域 =================
# 1. 输入数据文件或目录
INPUT_DIR = r"data/task2_dataset_human.jsonl" 
# 2. 设置评测模型

MODEL_LIST = ["qwen3-max"] 

# 3. 结果输出目录
OUTPUT_DIR = r"runs/task2_results_qwen3_max_ali"


# 4. 并发线程数
MAX_WORKERS = 200

# 5. Fallback Prompt (防止原始数据没有 query)
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
    """
    单条处理流水线：
    1. Parse Entry -> Probe
    2. Evaluator -> Model Output
    3. Curator -> Score
    """
    record = entry.get("record", {}) or {}
    
    # 1. 构建 Probe（严格遵循 run_task2_dataset 结构）
    query = entry.get("query", "") or ""
    if not query:
        print("Warn: No query found in record, using default.")
        query = DEFAULT_TASK2_QUERY

    # 获取 Ground Truth
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

    # 2. 运行模型 (Evaluator)
    # evaluate_models 返回 report: {"runs": [{"model": "...", "memory_items": [...], ...}]}
    report = evaluator.evaluate_models(
        probe,
        model_list,
        call_model=call_model_api,
        api_config=api_config
    )
    
    # 3. 运行评分 (Curator)
    # 我们遍历 report 中的每个 run 进行打分，然后把分数塞回 report 或者新的 score 字段
    model_runs = report.get("runs", [])
    
    # 构建 standard score 结构
    score_struct = {
        "runs": []
    }
    
    for run in model_runs:
        model_name = run.get("model", "unknown")
        pred_memories = run.get("memory_items", [])
        
        # 准备打分输入
        pred_record = {"memory_items": pred_memories}
        gold_record = {"memory_items": gt_memories}
        
        # 计算分数
        score_data = curator.score(
            gold_record, 
            pred_record,
            use_llm=True, 
            matcher="greedy",
            api_config=api_config
        )
        
        # 将分数记录下来
        run_score_info = {
            "model": model_name,
            "score": score_data,
            # 可以根据需要保留一些基本的计数
            "_meta": {
                "pred_count": len(pred_memories),
                "gold_count": len(gt_memories)
            }
        }
        score_struct["runs"].append(run_score_info)

    # 返回完整结果：包含原始 entry, 模型输出 report, 和 评分 score
    return {
        "entry": entry,
        "report": report,
        "score": score_struct
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
    print("Starting Task 2 Full Pipeline (Evaluate + Curate)")
    print(f"Input: {INPUT_DIR}")
    print(f"Models: {MODEL_LIST}")
    print("="*60)

    evaluator = Task2Evaluator()
    curator = Task2Curator()

    # 加载所有数据
    entries = []
    if os.path.isfile(INPUT_DIR):
        paths = [INPUT_DIR]
    else:
        paths = list_jsonl_files(INPUT_DIR)
        
    for path in paths:
        for ent in iter_jsonl(path):
            entries.append(ent)
            
    print(f"Loaded {len(entries)} entries. Processing with {MAX_WORKERS} workers...")
    
    # 输出文件：包含 scores 和 model outputs 
    # 这里文件名加上了 _scored 后缀以示区别
    output_filename = "scored.jsonl"
    output_file = os.path.join(OUTPUT_DIR, output_filename)
    
    if os.path.exists(output_file):
        print("exist")
        return
        os.remove(output_file)
        
    lock = threading.Lock()
    results = [] # In-memory storage for summary stats
    
    def safe_handle_result(res):
        with lock:
            append_jsonl(output_file, res)
            results.append(res)

    # 并行执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_pipeline, ent, evaluator, curator, MODEL_LIST, api_config): ent 
            for ent in entries
        }
        
        for future in tqdm(as_completed(futures), total=len(entries), desc="Processing"):
            try:
                res = future.result()
                safe_handle_result(res)
            except Exception as e:
                ent = futures[future]
                meta = ent.get("metadata", {})
                sid = meta.get("session_id", "unknown")
                print(f"Error processing session {sid}: {e}")

    print("-" * 30)
    print(f"Pipeline Finished!")
    print(f"All results saved to: {output_file}")
    
    if results:
        print_summary(results)

def print_summary(results):
    model_stats = {} 
    
    for res in results:
        for run in res.get("score", {}).get("runs", []):
            name = run["model"]
            # score -> metrics -> coverage -> f1
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

if __name__ == "__main__":
    main()