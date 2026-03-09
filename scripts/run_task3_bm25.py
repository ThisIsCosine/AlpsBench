import json
import os
import sys
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from rank_bm25 import BM25Okapi

import argparse
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize



# === NLTK 资源初始化 (自动下载) ===
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab') # 新版 nltk 需要
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# 初始化 Stemmer 和 Stopwords
_STEMMER = PorterStemmer()
_STOP_WORDS = set(stopwords.words('english'))
# ========================

# 确保能导入 src
sys.path.append(os.getcwd())

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from src.agents.shared import iter_jsonl, list_jsonl_files, ensure_dir, append_jsonl
from src.agents.task3_memory_retrieval.curator import Task3Grader

# ================= 配置区域 =================
DEFAULT_DATA_PATH = r"data/task3"
DEFAULT_DISTRACTORS = 700
DEFAULT_OUTPUT_DIR = r"runs/task3_bm25_results"
DEFAULT_MAX_WORKERS = 50
# ===============================================

def nltk_tokenize(text):
    """
    使用 NLTK 进行高级分词：
    1. 转小写
    2. word_tokenize 分词
    3. 过滤非字母数字字符 (isalnum)
    4. 过滤停用词
    5. Porter Stemming (词干提取)
    """
    text = str(text).lower()
    
    # 分词
    try:
        tokens = word_tokenize(text)
    except Exception:
        # fallback if nltk fails on some weird encoding
        tokens = text.split()

    processed_tokens = []
    for t in tokens:
        # 只保留字母数字且不在停用词表中
        if t.isalnum() and t not in _STOP_WORDS:
            # 词干提取 (例如 processing -> process)
            stemmed = _STEMMER.stem(t)
            processed_tokens.append(stemmed)
            
    return processed_tokens

def run_bm25_retrieval(probe, model_name="bm25"):
    query = probe.get("query", "")
    candidates = probe.get("candidate_memories", [])
    
    if not candidates:
        return {"error": "no candidates"}

    # 1. 准备语料
    corpus_texts = []
    valid_candidates = []
    
    for cand in candidates:
        txt = cand.get("value") or cand.get("content") or cand.get("text") or ""
        if txt:
            corpus_texts.append(txt)
            valid_candidates.append(cand)
            
    if not corpus_texts:
        return {"error": "no valid text in candidates"}

    # 2. 分词 (使用新的 NLTK tokenizer)
    tokenized_corpus = [nltk_tokenize(doc) for doc in corpus_texts]
    tokenized_query = nltk_tokenize(query)

    if not tokenized_query:
        tokenized_query = str(query).lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_memory = valid_candidates[best_idx]

    run_result = {
        "model": model_name,
        "raw_output": f"BM25 Score: {best_score:.4f}",
        "parsed_response": best_memory, 
    }
    
    return {
        "probe_id": probe.get("metadata", {}).get("dialog_id", ""),
        "runs": [run_result]
    }

def process_pipeline(entry, grader, api_config):
    probe = entry.get("record", {})
    if not probe: probe = entry 

    if "query" not in probe or "candidate_memories" not in probe:
        return {"entry": entry, "error": "invalid_format"}

    try:
        report = run_bm25_retrieval(probe, model_name="bm25_nltk")
        # print(report)
    except Exception as e:
        return {"entry": entry, "error": f"bm25_failed: {str(e)}"}
    
    # === 评分 ===
    try:
        score = grader.score_report(
            probe,
            report,
            use_llm=False, 
            api_config=api_config
        )
    except Exception as e:
        return {"entry": entry, "report": report, "error": f"grader_failed: {str(e)}"}
    
    return {"entry": entry, "report": report, "score": score}

def main():
    parser = argparse.ArgumentParser(description="Run Task 3 BM25 Retrieval Baseline")
    parser.add_argument("--distractors", type=int, default=DEFAULT_DISTRACTORS, help="Number of distractors (e.g., 100, 300, 1000)")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Base directory for task 3 data")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Base directory to save results")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help="Max parallel workers")
    args = parser.parse_args()

    INPUT_FILE = os.path.join(args.data_path, f"task3_dataset_d{args.distractors}.jsonl")
    OUTPUT_DIR = f"{args.output_dir}_d{args.distractors}"
    MAX_WORKERS = args.workers
    ensure_dir(OUTPUT_DIR)
    config_path = "configs/api.json"
    api_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            api_config = json.load(f)

    print(f"Running BM25 Retrieval Baseline (NLTK Enhanced)...")
    grader = Task3Grader()

    entries = []
    for path in list_jsonl_files(INPUT_FILE) if os.path.isdir(INPUT_FILE) else [INPUT_FILE]:
        for ent in iter_jsonl(path):
            entries.append(ent)
            
    print(f"Loaded {len(entries)} entries.")
    output_file = os.path.join(OUTPUT_DIR, "scored_bm25.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)
        
    lock = threading.Lock()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pipeline, ent, grader, api_config): ent for ent in entries}
        for future in tqdm(as_completed(futures), total=len(entries), desc="BM25 Search"):
            res = future.result()
            with lock:
                append_jsonl(output_file, res)
                if "error" not in res: results.append(res)

    hits = sum(1 for r in results if r["score"]["runs"][0]["judge"]["score"] == 1.0)
    
    if len(results) > 0:
        print(f"\nBM25 Accuracy: {hits / len(results):.2%} ({hits}/{len(results)})")
    else:
        print("\nNo results generated.")
        
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()