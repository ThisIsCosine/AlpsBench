import json
import os
import sys
import threading
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util

# 确保能导入 src
sys.path.append(os.getcwd())

from src.agents.shared import iter_jsonl, list_jsonl_files, ensure_dir, append_jsonl
from src.agents.task3_memory_retrieval.curator import Task3Grader

# ================= 配置区域 =================
INPUT_FILE = r"data/task3_dataset_d1000.jsonl"
OUTPUT_DIR = r"runs/task3_d1000_vector_results"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 
MAX_WORKERS = 12  
# ===============================================

def run_vector_retrieval(probe, model, model_name="vector_sim"):
    query = probe.get("query", "")
    candidates = probe.get("candidate_memories", [])
    
    if not candidates: return {"error": "no candidates"}

    # 1. 准备文本
    candidate_texts = []
    valid_candidates = []
    
    # 提取文本，处理可能的 None
    for cand in candidates:
        txt = cand.get("value") or cand.get("content") or cand.get("text") or ""
        if txt:
            candidate_texts.append(str(txt)) # 确保是字符串
            valid_candidates.append(cand)
            
    if not candidate_texts: return {"error": "no valid text"}

    try:
        # 2. 编码 (Embedding)
        # 批量编码
        query_emb = model.encode(query, convert_to_tensor=True)
        cand_embs = model.encode(candidate_texts, convert_to_tensor=True)

        # 3. 计算余弦相似度
        cosine_scores = util.cos_sim(query_emb, cand_embs)[0]

        # 4. 获取最佳匹配
        best_idx_tensor = np.argmax(cosine_scores.cpu().numpy())
        best_idx = int(best_idx_tensor)
        best_score = float(cosine_scores[best_idx])
        
        best_memory = valid_candidates[best_idx]

        run_result = {
            "model": model_name,
            "raw_output": f"Cosine Similarity: {best_score:.4f}",
            "parsed_response": best_memory, 
        }
        
        return {
            "probe_id": probe.get("metadata", {}).get("dialog_id", ""),
            "runs": [run_result]
        }
    except Exception as e:
        return {"error": f"encoding_failed: {str(e)}"}

def process_pipeline(entry, model, grader, api_config):
    """单条处理流"""
    probe = entry.get("record", {})
    if not probe: probe = entry

    # === 检索 ===
    try:
        report = run_vector_retrieval(probe, model, model_name=EMBEDDING_MODEL)
        if "error" in report:
            return {"entry": entry, "error": report["error"]}
    except Exception as e:
        return {"entry": entry, "error": f"retrieval_exception: {e}"}
    
    # === 评分 ===
    try:
        score = grader.score_report(
            probe, 
            report, 
            use_llm=False, 
            api_config=api_config
        )
    except Exception as e:
         return {"entry": entry, "error": f"grading_exception: {e}"}
    
    return {
        "entry": entry,
        "report": report,
        "score": score
    }

def main():
    ensure_dir(OUTPUT_DIR)
    config_path = "configs/api.json"
    api_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            api_config = json.load(f)

    print(f"Loading Embedding Model: {EMBEDDING_MODEL} ...")
    # 加载模型 (在主进程加载，传入线程使用)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.")

    grader = Task3Grader()
    
    output_file = os.path.join(OUTPUT_DIR, "scored_vector.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)

    entries = []
    # 收集所有数据文件
    file_list = list_jsonl_files(INPUT_FILE) if os.path.isdir(INPUT_FILE) else [INPUT_FILE]
    for path in file_list:
        for ent in iter_jsonl(path):
            entries.append(ent)
    
    print(f"Loaded {len(entries)} entries. Processing with vectors in parallel (Workers={MAX_WORKERS})...")
    
    results = []
    lock = threading.Lock()

    # 使用 ThreadPoolExecutor 进行并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        futures = {
            executor.submit(process_pipeline, ent, embed_model, grader, api_config): ent 
            for ent in entries
        }
        
        for future in tqdm(as_completed(futures), total=len(entries), desc="Vector Search"):
            res = future.result()
            
            # 写入结果 (加锁)
            with lock:
                append_jsonl(output_file, res)
                if "error" not in res: 
                    results.append(res)

    # 统计准确率
    hits = 0
    for r in results:
        runs = r.get("score", {}).get("runs", [])
        if runs:
            # Task3Grader 结构通常: runs[0]["judge"]["score"]
            judge_res = runs[0].get("judge", {})
            if judge_res.get("score") == 1.0:
                hits += 1

    count = len(results)
    if count > 0:
        print(f"\nVector Model ({EMBEDDING_MODEL}) Accuracy: {hits / count:.2%} ({hits}/{count})")
    else:
        print("\nNo valid results.")
        
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()