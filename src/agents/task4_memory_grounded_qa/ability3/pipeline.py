"""
Task 4 Ability 3 Pipeline.
"""

import os
import json
import argparse
from .generator import Ability3Generator
from .evaluator import Ability3Evaluator
from .curator import Ability3Curator
from ...shared import iter_jsonl, append_jsonl, ensure_dir, make_run_dir, list_jsonl_files

def run_ability3_pipeline(
    model_under_test: str,
    input_dir: str = "benchmark/dev_with_selected_memory_id/ability3",
    max_samples: int = 10,
    gen_model_id: str = "gpt-5.2",
    judge_model_id: str = "gpt-5.2"
):
    # Handle the potential missing directory for ability3
    if not os.path.exists(input_dir):
        # Try data dir as fallback
        fallback = "data/wildchat/memories/selected/task4/ability3"
        if os.path.exists(fallback): input_dir = fallback
        else:
            print(f"Error: input_dir {input_dir} not found."); return

    output_dir = make_run_dir("runs", "task4_ability3")
    ensure_dir(output_dir)
    results_file = os.path.join(output_dir, "ability3_results.jsonl")

    gen = Ability3Generator(model_id=gen_model_id)
    evaluator = Ability3Evaluator()
    curator = Ability3Curator(judge_model_id=judge_model_id)
    
    jsonl_files = list_jsonl_files(input_dir)
    count = 0

    print(f"\n=== TASK 4 ABILITY 3: ROLE-PLAY GROUNDING PIPELINE ===")
    
    for file_path in jsonl_files:
        for record in iter_jsonl(file_path):
            if count >= max_samples: return
            try:
                query = gen.generate_query(record)
                if not query: continue
                print(f"--- [Sample {count}] Query: {query}")

                dialogue = (record.get("sessions") or [{}])[0].get("turns") or record.get("dialogue") or []
                model_output = evaluator.evaluate_sample(model_under_test, dialogue, query)
                
                selected_id = record.get("selected_memory_id")
                memories = record.get("memory_items") or []
                selected_memory = next((m for m in memories if m.get("memory_id") == selected_id), None)
                
                scores = curator.score(selected_memory, model_output)
                print(f"  [Result] Grounding: {scores.get('context_grounding', 0)}")

                append_jsonl(results_file, {
                    "sample_id": count,
                    "query": query,
                    "model_output": model_output,
                    "ground_truth_memory": selected_memory,
                    "scores": scores
                })
                count += 1
            except Exception as e:
                print(f"  [Error] {e}"); continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    run_ability3_pipeline(model_under_test=args.model, max_samples=args.samples)
