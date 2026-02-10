"""
Task 4 Ability 2 Interaction Pipeline.
"""

import os
import json
import argparse
from .generator import Ability2InteractionGenerator
from .evaluator import Ability2InteractionEvaluator
from .curator import Ability2InteractionCurator
from ...shared import iter_jsonl, append_jsonl, ensure_dir, make_run_dir, list_jsonl_files

def run_ability2_interaction_pipeline(
    model_under_test: str,
    input_dir: str = "benchmark/dev_with_selected_memory_id/ability2_interaction",
    max_samples: int = 10,
    gen_model_id: str = "gpt-5.2",
    judge_model_id: str = "gpt-5.2"
):
    output_dir = make_run_dir("runs", "task4_ability2_interaction")
    ensure_dir(output_dir)
    results_file = os.path.join(output_dir, "ability2_interaction_results.jsonl")

    gen = Ability2InteractionGenerator(model_id=gen_model_id)
    evaluator = Ability2InteractionEvaluator()
    curator = Ability2InteractionCurator(judge_model_id=judge_model_id)
    
    jsonl_files = list_jsonl_files(input_dir)
    count = 0

    print(f"\n=== TASK 4 ABILITY 2 INTERACTION: PREFERENCE GROUNDING PIPELINE ===")
    
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
                print(f"  [Result] Following: {scores.get('preference_following', 0)}")

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
    run_ability2_interaction_pipeline(model_under_test=args.model, max_samples=args.samples)
