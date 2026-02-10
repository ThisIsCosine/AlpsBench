import json
import os
import sys

sys.path.append(os.getcwd())

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from src.agents.shared import iter_jsonl, ensure_dir, append_jsonl
from src.agents.task2_memory_update.curator import Task2Curator

# ================= 配置区域 =================
# 1. 对应 Step 1 的输出文件
INPUT_FILE = r"runs/task2_split_run/step1_model_outputs.jsonl"

# 2. 结果保存目录
OUTPUT_DIR = r"runs/task2_split_run"
# ============================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: 找不到输入文件 {INPUT_FILE}，请先运行 Step 1。")
        return

    print("Step 2: Running Curator (Scoring)...")
    curator = Task2Curator()
    ensure_dir(OUTPUT_DIR)
    
    output_file = os.path.join(OUTPUT_DIR, "step2_scores.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)
    
    records = list(iter_jsonl(INPUT_FILE))
    print(f"Loaded {len(records)} evaluated records to score.")

    results = []
    
    for record in tqdm(records, desc="Scoring"):
        # 从 Step 1 的输出结构中提取信息
        # record 结构: {"record_id": ..., "probe": {...}, "report": {...}}
        
        probe = record.get("probe", {})
        report = record.get("report", {})
        
        # 获取 Ground Truth
        # 在 Step 1 中我们将它放到了 probe["ground_truth_memories"]
        gt_memories = probe.get("ground_truth_memories", [])
        gold_record = {"memory_items": gt_memories}

        # 遍历所有模型的运行结果
        # report 结构通常是: {"runs": [{"model": "gpt4", "memory_items": [...]}, ...]}
        model_runs = report.get("runs", [])
        
        scored_runs = []
        
        for run in model_runs:
            model_name = run.get("model", "unknown")
            pred_memories = run.get("memory_items", [])
            pred_record = {"memory_items": pred_memories}
            
            # 调用评分逻辑
            score_data = curator.score(gold_record, pred_record)
            
            scored_runs.append({
                "model": model_name,
                "score": score_data,
                "pred_count": len(pred_memories),
                "gold_count": len(gt_memories)
            })
            
        # 保存这个 session 的所有评分结果
        output_entry = {
            "record_id": record.get("record_id"),
            "session_id": probe.get("metadata", {}).get("session_id"),
            "scored_runs": scored_runs
        }
        
        append_jsonl(output_file, output_entry)
        results.append(output_entry)

    print("-" * 30)
    print(f"Step 2 Finished. Scores saved to: {output_file}")
    
    # 简单打印一下平均分 (Optional)
    if results:
        print_summary(results)

def print_summary(results):
    model_stats = {} # { model_name: [f1_scores...] }
    
    for res in results:
        for run in res["scored_runs"]:
            name = run["model"]
            # 假设 score_data 结构来自 standard shared score_records
            # 通常是: score -> metrics -> coverage -> f1
            metrics = run.get("score", {}).get("metrics", {}).get("coverage", {})
            f1 = metrics.get("f1", 0.0)
            
            if name not in model_stats: model_stats[name] = []
            model_stats[name].append(f1)
            
    print("\nPreliminary F1 Summary:")
    for model, scores in model_stats.items():
        avg_f1 = sum(scores) / len(scores) if scores else 0
        print(f"  Model: {model:<20} | Avg F1: {avg_f1:.4f} | Records: {len(scores)}")

if __name__ == "__main__":
    main()