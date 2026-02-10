"""
Task 4 Ability 5 Pipeline.
"""

import os
import json
import argparse
from .generator import Ability5Generator
from .evaluator import Ability5Evaluator
from .curator import Ability5Curator
from ...shared import iter_jsonl, append_jsonl, ensure_dir, make_run_dir, list_jsonl_files

def run_ability5_pipeline(
    model_under_test: str,
    input_dir: str = "benchmark/dev_with_selected_memory_id/ability5",
    max_samples: int = 10,
    gen_model_id: str = "gpt-5.2",
    judge_model_id: str = "gpt-5.2"
):
    if not os.path.exists(input_dir):
        fallback = "data/wildchat/memories/selected/task4/ability5"
        if os.path.exists(fallback): input_dir = fallback
        else:
            print(f"Error: input_dir {input_dir} not found."); return

    output_dir = make_run_dir("runs", "task4_ability5")
    ensure_dir(output_dir)
    results_file = os.path.join(output_dir, "ability5_results.jsonl")

    gen = Ability5Generator(model_id=gen_model_id)
    evaluator = Ability5Evaluator()
    curator = Ability5Curator(judge_model_id=judge_model_id)
    
    jsonl_files = list_jsonl_files(input_dir)
    count = 0

    print(f"\n=== TASK 4 ABILITY 5: EQ GROUNDING PIPELINE ===")
    
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
                print(f"  [Result] EQ Grounding: {scores.get('emotional_grounding', 0)}")

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
    run_ability5_pipeline(model_under_test=args.model, max_samples=args.samples)
