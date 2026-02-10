import json
import os
import sys
import threading
import copy
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 将项目根目录加入路径
sys.path.append(os.getcwd())

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from src.agents.shared import ensure_dir, append_jsonl, list_jsonl_files, iter_jsonl, resolve_session_id, call_model_api
from src.agents.task3_memory_retrieval.generator import Task3Generator
from src.agents.task3_memory_retrieval.curator import Task3Judger

# ================= 用户配置区域 =================
# 1. 输入 Seed 数据 (包含 dialogue 和 memories)
INPUT_PATH = r"data/human_annotation_with_selected_memory_id/task4_final_human_annotation_with_selected_memory_id" 

# 2. 输出目录 (生成的 Probes 将保存在这里)
OUTPUT_DIR = r"runs/task3_generation_only_1"

# 3. 模型配置
GENERATOR_MODEL = "gpt-5.2" # 用于生成 Query
JUDGE_MODEL = "deepseek-chat"   # 用于检查 Query 质量 (Judger)

# 4. 运行参数
MAX_ATTEMPTS = 3         # 每个 Seed 最多重试几次
MAX_WORKERS = 200        # 并发线程数
SKIP_ON_DISCARD = False  # 如果 Judger 判定为 DISCARD，是否直接跳过

# 5. 调试限制 (设置为 None 则运行全部)
SAMPLE_LIMIT = None
# ===============================================

def call_model_api_wrapper(model_id, prompt_id, sys_prompt, user_prompt, api_config=None):
    return call_model_api(model_id, prompt_id, sys_prompt, user_prompt, api_config=api_config)

def process_single_seed_pipeline(
    seed: Dict[str, Any],
    generator: Task3Generator,
    judger: Task3Judger,
    api_config: Dict[str, Any]
):
    """
    完全遵循 curator.run_task3_pipeline 的逻辑进行生成和判决。
    """
    # 1. Resolve ID and Memory (Logic from pipeline)
    seed_id = resolve_session_id(seed) or seed.get("seed_id")
    
    memories = seed.get("memories") or seed.get("memory_items") or seed.get("memory") or []
    selected_memory_id = seed.get("selected_memory_id")
    selected_memory = seed.get("selected_memory")
    
    if not selected_memory and selected_memory_id:
        for memory in memories:
            m_id = memory.get("memory_id") or memory.get("id")
            if str(m_id) == str(selected_memory_id):
                selected_memory = memory
                break
                
    if not selected_memory:
         if len(memories) == 1 and not selected_memory_id:
             selected_memory = memories[0]

    if not selected_memory:
        return {"status": "error", "reason": "missing_selected_memory", "seed_id": seed_id or "unknown"}

    current_seed = dict(seed)
    current_seed["selected_memory"] = selected_memory
    current_seed["selected_memory_id"] = selected_memory.get("memory_id")
    
    def bound_call_model(sys_prompt, user_prompt):
        return call_model_api_wrapper(
            GENERATOR_MODEL, 
            "task3_gen",
            sys_prompt, 
            user_prompt, 
            api_config=api_config
        )

    # 2. Pipeline Loop
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # A. Generate Probe
            probe = generator.generate_probe(
                current_seed,
                bound_call_model,
                model_id=GENERATOR_MODEL,
                api_config=api_config,
                log_fn=None
            )
            
            if current_seed.get("dialogue") is not None:
                probe["dialogue"] = current_seed.get("dialogue")
            if probe.get("selected_memory") and not probe.get("selected_memory_id"):
                probe["selected_memory_id"] = probe.get("selected_memory", {}).get("memory_id")
            
            # B. Judge Decision
            decision = judger.decide_generation_only(
                probe,
                curator_model_id=JUDGE_MODEL,
                api_config=api_config,
            )
            
            # === [Curator Logic] Force Accept on Max Attempts ===
            # 如果是最后一次尝试，强制接受，忽略 Judger 的 REWRITE 建议
            if attempt == MAX_ATTEMPTS:
                if decision["verdict"] != "ACCEPT":
                    print(f"    [Max attempts reached] Seed {seed_id}: Forcing verdict to ACCEPT (was {decision['verdict']}).")
                    decision["verdict"] = "ACCEPT"
                    # 标记为强制通过
                    decision["reasons"] = decision.get("reasons", []) + ["forced_accept_max_attempts"]
            # ====================================================

            # C. Handle Verdicts
            if decision["verdict"] == "ACCEPT":
                # Check for forced flag
                is_forced = "forced_accept_max_attempts" in decision.get("reasons", [])
                
                return {
                    "status": "success",
                    "final_probe": probe,
                    "attempts": attempt,
                    "seed_id": seed_id,
                    "verdict": "FORCED_ACCEPT" if is_forced else "ACCEPT"
                }
            
            if decision["verdict"] == "REWRITE" or decision["verdict"] == "DISCARD":
                feedback = decision.get("feedback_for_generator") or ""
                if feedback:
                    current_seed["generator_feedback"] = feedback
                
                # Continue loop to next attempt
                continue
            
            # if decision["verdict"] == "DISCARD":
            #     # Curator logic breaks on DISCARD
            #     if SKIP_ON_DISCARD:
            #         return {"status": "skipped", "reason": "discarded_by_judge", "seed_id": seed_id}
                # else:
                #     return {"status": "failed", "reason": "discarded_by_judge", "seed_id": seed_id}

        except Exception as e:
            error_msg = str(e)
            print(error_msg)
            if "generator_missing_query" in error_msg:
                 current_seed["generator_feedback"] = (
                     "System: You FAILED to generate a JSON object with a 'query' field. "
                     "You MUST output valid JSON containing {'query': '...'}."
                 )
                 # 如果是最后一次尝试且发生 Crash，我们就无法挽救了，只能算失败
                 continue

            # print(f"[Error] Seed {seed_id} - Attempt {attempt} crashed: {e}")
            continue
            
    # 3. Post-Loop Fallback
    # 如果代码运行到这里，说明循环跑满了 MAX_ATTEMPTS 且全部 Exception Crashed（因为如果有 Verdict 都会在循环内返回）
    return {"status": "failed", "reason": "all_attempts_crashed", "seed_id": seed_id}

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

    print("=== Starting Task 3 Generator (Pipeline Mirror with Force Accept) ===")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max Workers: {MAX_WORKERS}")
    
    generator = Task3Generator()
    judger = Task3Judger()

    seeds = []
    if os.path.isdir(INPUT_PATH):
        files = list_jsonl_files(INPUT_PATH)
    else:
        files = [INPUT_PATH]
    
    for f in files:
        for line in iter_jsonl(f):
            seeds.append(line)
            
    print(f"Total seeds found: {len(seeds)}")
    
    # === Limit Samples ===
    if SAMPLE_LIMIT and len(seeds) > SAMPLE_LIMIT:
        seeds = seeds[:SAMPLE_LIMIT]
        print(f"--> Limited run to first {SAMPLE_LIMIT} seeds.")
    
    output_path = os.path.join(OUTPUT_DIR, "generated_probes.jsonl")
    audit_path = os.path.join(OUTPUT_DIR, "generation_audit.jsonl")
    
    if os.path.exists(output_path): os.remove(output_path)
    if os.path.exists(audit_path): os.remove(audit_path)

    lock = threading.Lock()
    stats = {"success": 0, "failed": 0, "skipped": 0, "forced": 0}

    def save_result(res):
        with lock:
            if res["status"] == "success":
                append_jsonl(output_path, res["final_probe"])
                stats["success"] += 1
                if res.get("verdict") == "FORCED_ACCEPT":
                    stats["forced"] += 1
            elif res["status"] == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
            append_jsonl(audit_path, res)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_seed_pipeline, seed, generator, judger, api_config): seed 
            for seed in seeds
        }
        
        for future in tqdm(as_completed(futures), total=len(seeds), desc="Generating"):
            try:
                result = future.result()
                save_result(result)
            except Exception as e:
                print(f"Unhandled exception: {e}")

    print("-" * 30)
    print("Generation Complete.")
    print(f"Total Seeds Processed: {len(seeds)}")
    print(f"Generated: {stats['success']} (Forced Accept: {stats['forced']})")
    print(f"Skipped/Discarded: {stats['skipped']}")
    print(f"Errors: {stats['failed']}")
    print(f"Probes saved to: {output_path}")

if __name__ == "__main__":
    main()