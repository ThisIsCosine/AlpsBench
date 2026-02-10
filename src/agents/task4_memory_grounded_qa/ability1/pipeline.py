"""
Task 4 Ability 1 Pipeline: Persona-Grounded QA Runner.
Processes records from data/wildchat/memories/selected/task4_final_with_selected_memory_id/ability1.
"""

import os
import json
import argparse
from typing import Any, Dict, List, Optional
from .generator import Ability1Generator
from .evaluator import Ability1Evaluator
from .curator import Ability1Curator
from ...shared import iter_jsonl, append_jsonl, ensure_dir, make_run_dir, list_jsonl_files

def run_ability1_pipeline(
    model_under_test: str,
    input_dir: str = "data/wildchat/memories/selected/task4_final_with_selected_memory_id/ability1",
    gen_model_id: str = "gpt-5.2",
    judge_model_id: str = "gpt-5.2"
):
    output_dir = make_run_dir("runs", "task4_ability1")
    ensure_dir(output_dir)
    results_file = os.path.join(output_dir, "ability1_results.jsonl")

    gen = Ability1Generator(model_id=gen_model_id)
    evaluator = Ability1Evaluator()
    curator = Ability1Curator(judge_model_id=judge_model_id)
    
    jsonl_files = list_jsonl_files(input_dir)
    count = 0

    print(f"\n=== TASK 4 ABILITY 1: PERSONA GROUNDING PIPELINE ===")
    print(f"Target Model: {model_under_test} | Generator: {gen_model_id} | Judge: {judge_model_id}\n")

    for file_path in jsonl_files:
        print(f"Processing file: {os.path.basename(file_path)}")
        for record in iter_jsonl(file_path):
            print(f"--- [Sample {count}] ---")
            
            try:
                # 1. Generate Persona-Based Query
                query = gen.generate_query(record)
                if not query:
                    print("  [Skip] No query generated.")
                    continue
                print(f"  [Gen] Query: {query}")

                # 2. Evaluate Target Model
                selected_id = record.get("selected_memory_id")
                memories = record.get("memory_items") or record.get("memories") or []
                selected_memory = record.get("selected_memory") or next(
                    (m for m in memories if m.get("memory_id") == selected_id),
                    None,
                )
                if not selected_memory:
                    print("  [Skip] No selected_memory found.")
                    continue
                model_output = evaluator.evaluate_sample(model_under_test, selected_memory, query)
                print(f"  [Eval] Answer: {model_output.get('answer', '')[:100]}...")
                print(f"  [Eval] Used Fact: {model_output.get('used_memory_fact', '')}")

                # 3. Score
                scores = curator.score(selected_memory, model_output)
                print(f"  [Result] Match: {scores.get('fact_match', 0)} | Correctness: {scores.get('memory_correctness', 0)} | Quality: {scores.get('answer_quality', 0)}")
                if scores.get("reasoning"):
                    print(f"  [Reason] {scores.get('reasoning')}")

                # 4. Save
                append_jsonl(results_file, {
                    "sample_id": count,
                    "source_file": file_path,
                    "query": query,
                    "model_output": model_output,
                    "ground_truth_memory": selected_memory,
                    "scores": scores
                })
                count += 1
                print("-" * 50)

            except Exception as e:
                print(f"  [Error] {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model under test")
    parser.add_argument("--gen_model", default="gpt-5.2", help="Model for generation")
    parser.add_argument("--judge_model", default="gpt-5.2", help="Model for judging")
    args = parser.parse_args()

    run_ability1_pipeline(
        model_under_test=args.model, 
        gen_model_id=args.gen_model,
        judge_model_id=args.judge_model
    )
