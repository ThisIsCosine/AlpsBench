import json
import os
import sys
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 将项目根目录加入路径
sys.path.append(os.getcwd())

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from src.agents.shared import ensure_dir, append_jsonl, iter_jsonl, resolve_session_id
from src.agents.task3_memory_retrieval.curator import _sanitize_memory_item as _curator_sanitize

# ================= 用户配置区域 =================
# 1. 前一步生成的 Probes 文件 (必须包含 source_file 字段)
INPUT_PROBES_FILE = r"runs/task3_generation_only_1/probes_with_source.jsonl"

# 2. 原始数据根目录 (用于构建全局池子)
SEED_DATA_DIR = r"data/human_annotation_with_selected_memory_id"

# 3. 输出目录
OUTPUT_DIR = r"runs/task3_final_dataset"

# 4. Distractor 设置
DISTRACTOR_COUNT = 200  # 最终选项数 (1个正确 + N个干扰)

# 5. 是否打乱选项顺序
SHUFFLE_CANDIDATES = True

# 6. 是否保留标签
INCLUDE_LABEL = True 

# 7. 并发设置
MAX_WORKERS = 20
# ===============================================

def _sanitize_memory_wrapper(m, include_label):
    if not m: return {}
    return _curator_sanitize(m, include_label)

def load_file_based_memory_pool(source_dir):
    """
    加载所有 memory，并按来源文件路径分组。
    返回: { file_abs_path: [memory_item, memory_item, ...] }
    """
    file_memory_map = {}
    source_dir = os.path.abspath(source_dir)
    print(f"Scanning memories in: {source_dir}")
    
    total_mems = 0
    file_count = 0
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".jsonl") and not file.startswith("_"):
                full_path = os.path.abspath(os.path.join(root, file))
                memories_in_file = []
                
                try:
                    for record in iter_jsonl(full_path):
                        # 尝试多种字段名
                        mems = record.get("memories") or record.get("memory_items") or record.get("memory") or []
                        
                        # 兼容单条格式
                        if not mems and ("value" in record or "content" in record):
                            mems = [record]
                            
                        for m in mems:
                            # 必须有 ID 和 内容
                            if (m.get("memory_id") or m.get("id")) and (m.get("value") or m.get("content")):
                                # 简单清理一下 ID，统一转字符串
                                m["memory_id"] = str(m.get("memory_id") or m.get("id"))
                                memories_in_file.append(m)
                                
                    if memories_in_file:
                        file_memory_map[full_path] = memories_in_file
                        total_mems += len(memories_in_file)
                        file_count += 1
                        
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
    
    print(f"  Loaded {total_mems} memories from {file_count} files.")
    return file_memory_map

def process_single_probe_dynamic(probe, file_memory_map, distractor_count, include_label):
    """
    针对每条 Probe 生成数据：
    1. 识别 source_file
    2. 从 global map 中排除该 source_file
    3. 从剩余的文件中采样 distractors
    """
    seed_id = probe.get("seed_id") or resolve_session_id(probe) or "unknown"
    selected_memory = probe.get("selected_memory")
    source_file = probe.get("source_file")
    
    if not selected_memory: 
        return None

    # 标准化 source_file 路径方便比较 (处理 Win/Linux 路径符差异)
    excluded_path = None
    if source_file and source_file != "NOT_FOUND":
        excluded_path = Path(source_file).resolve()
    # excluded_path = str(excluded_path).replace('\\', '\\\\')
    # === 构建候选 Distractor 池 ===
    # 逻辑：遍历 Map，如果 key != excluded_path，则该 list 里的所有 memory 都是合法的 distractor
    # 为了性能，我们不一定要把所有 valid items 扁平化成一个巨型 list (那太慢了)。
    # 我们可以随机选文件，再从文件里随机选 memory。
    
    all_files = list(file_memory_map.keys())
    valid_files = [f for f in all_files if f != str(excluded_path)]
    # print(len(valid_files), len(all_files))
    
    candidate_memories = [selected_memory] # 先放入正确答案
    needed = distractor_count
    
    target_id = str(selected_memory.get("memory_id"))
    chosen_ids = {target_id} # 用于去重
    
    attempts = 0
    max_attempts = needed * 200 # 防止死循环
    
    if len(valid_files) > 0:
        while len(candidate_memories) < (needed + 1) and attempts < max_attempts:
            attempts += 1
            
            # 1. 随机选一个非源文件
            rand_file = random.choice(valid_files)
            mems_in_file = file_memory_map[rand_file]

            if not mems_in_file: continue
            
            # 2. 从该文件随机选一条记忆
            rand_mem = random.choice(mems_in_file)

            m_id = str(rand_mem.get("memory_id"))
            
            # 3. 校验：不能是重复 ID，且必须有内容
            if m_id != target_id:
                candidate_memories.append(rand_mem)
                chosen_ids.add(m_id)
    else:
        # 极端情况：只有一个文件，或者没找到 valid files
        # 此时只能从 source_file 内部找 (虽然原则是排除，但为了不报错只能 fallback)
        pass 

    # 截断（如果采样过多）或补全（如果实在找不到够数的）
    # 这里我们只保留 needed + 1 个
    candidate_memories = candidate_memories[:(needed + 1)]

    # 打乱顺序
    if SHUFFLE_CANDIDATES:
        random.shuffle(candidate_memories)

    # Sanitize 清理字段
    final_candidates = [_sanitize_memory_wrapper(m, include_label) for m in candidate_memories]
    final_selected = _sanitize_memory_wrapper(selected_memory, include_label)

    dataset_entry = {
        "seed_id": seed_id,
        "record": {
            "dialogue": probe.get("dialogue", []),
            "selected_memory_id": selected_memory.get("memory_id"),
            "selected_memory": final_selected,
            "candidate_memories": final_candidates,
            "query": probe.get("query", ""),
            "metadata": probe.get("metadata", {}),
            # 可选：保留调试信息
            # "debug_source_excluded": excluded_path
        },
    }
    return dataset_entry

def main():
    ensure_dir(OUTPUT_DIR)
    
    # 1. 加载 Global Map (按文件分组)
    # 这里必须加载整个 SEED_DATA_DIR，因为我们不知道哪些文件是 distractors
    global_file_map = load_file_based_memory_pool(SEED_DATA_DIR)
    
    if not global_file_map:
        print("[Error] No memories found. Please check SEED_DATA_DIR.")
        return
    
    # 2. 加载 Probes
    print(f"Loading probes from: {INPUT_PROBES_FILE}")
    if not os.path.exists(INPUT_PROBES_FILE):
        print(f"Error: Probes file not found: {INPUT_PROBES_FILE}")
        return
        
    probes = list(iter_jsonl(INPUT_PROBES_FILE))
    print(f"Loaded {len(probes)} probes.")

    output_path = os.path.join(OUTPUT_DIR, f"task3_dataset_d{DISTRACTOR_COUNT}.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Building Dataset with Strict File Exclusion (N={DISTRACTOR_COUNT})...")
    
    lock = threading.Lock()
    stats = {"count": 0, "skipped": 0}

    def run_job(p):
        return process_single_probe_dynamic(p, global_file_map, DISTRACTOR_COUNT, INCLUDE_LABEL)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_job, p): p for p in probes}
        
        for future in tqdm(as_completed(futures), total=len(probes), desc="Building"):
            try:
                res = future.result()
                if res:
                    with lock:
                        append_jsonl(output_path, res)
                        stats["count"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                print(f"Error building entry: {e}")

    print("-" * 30)
    print(f"Done. Entries: {stats['count']} | Skipped: {stats['skipped']}")
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()