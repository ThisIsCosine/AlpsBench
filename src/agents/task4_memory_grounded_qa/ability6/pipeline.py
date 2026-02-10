"""
Task 4 Ability 6 Pipeline: Memory Misuse Test Runner.
"""

import os
import json
import argparse
from .generator import Ability6Generator
from .evaluator import Ability6Evaluator
from .curator import Ability6Curator
from ...shared import iter_jsonl, append_jsonl, ensure_dir, make_run_dir, list_jsonl_files

def run_ability6_pipeline(
    model_under_test: str,
    input_dir: str = "benchmark/dev_with_selected_memory_id/ability1", # Can use any ability as base
    benchmark_root: str = "benchmark/dev_with_selected_memory_id",
    max_samples: int = 10,
    gen_model_id: str = "gpt-5.2",
    judge_model_id: str = "gpt-5.2",
    distractor_count: int = 5
):
    output_dir = make_run_dir("runs", "task4_ability6")
    ensure_dir(output_dir)
    results_file = os.path.join(output_dir, "ability6_results.jsonl")

    gen = Ability6Generator(model_id=gen_model_id, distractor_count=distractor_count)
    evaluator = Ability6Evaluator()
    curator = Ability6Curator(judge_model_id=judge_model_id)
    
    jsonl_files = list_jsonl_files(input_dir)
    count = 0

    print(f"\n=== TASK 4 ABILITY 6: MEMORY MISUSE / HALLUCINATION TEST ===")
    print(f"Target Model: {model_under_test} | Generator: {gen_model_id} | Judge: {judge_model_id}\n")

    for file_path in jsonl_files:
        for record in iter_jsonl(file_path):
            if count >= max_samples: return

            print(f"--- [Sample {count}] ---")
            
            try:
                # 1. Generate General Query + Distractor Memories
                gen_data = gen.generate_query_and_distractors(record, benchmark_root)
                query = gen_data["query"]
                distractors = gen_data["distractor_memories"]
                
                if not query:
                    print("  [Skip] No query generated.")
                    continue
                print(f"  [Gen] Query: {query}")
                print(f"  [Gen] Injected {len(distractors)} irrelevant memories.")

                # 2. Evaluate Target Model
                dialogue = (record.get("sessions") or [{}])[0].get("turns") or record.get("dialogue") or []
                model_output = evaluator.evaluate_sample(model_under_test, dialogue, query, distractors)
                print(f"  [Eval] Used Fact: {model_output.get('used_memory_fact')}")

                # 3. Score
                scores = curator.score(distractors, model_output)
                print(f"  [Result] No Hallucination: {scores.get('no_hallucination_score', 0)} | Honesty: {scores.get('honesty_score', 0)}")
                if scores.get("reasoning"):
                    print(f"  [Reason] {scores.get('reasoning')}")

                # 4. Save
                append_jsonl(results_file, {
                    "sample_id": count,
                    "query": query,
                    "distractors": distractors,
                    "model_output": model_output,
                    "scores": scores
                })
                count += 1
                print("-" * 50)

            except Exception as e:
                print(f"  [Error] {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    run_ability6_pipeline(
        model_under_test=args.model, 
        max_samples=args.samples
    )
