"""
Task 4 Batch Runner: Orchestrates all Memory-Grounded QA abilities.
Produces output in the format of dataset.jsonl, decisions.jsonl, events.jsonl, probes.jsonl, and reports.jsonl.
"""

import os
import json
import argparse
import random
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from ..shared import (
    iter_jsonl, append_jsonl, ensure_dir, make_run_dir, 
    list_jsonl_files, log_event, resolve_session_id
)

# Import all abilities
from .ability1 import Ability1Generator, Ability1Evaluator, Ability1Curator
from .ability2_general import Ability2GeneralGenerator, Ability2GeneralEvaluator, Ability2GeneralCurator
from .ability2_interaction import Ability2InteractionGenerator, Ability2InteractionEvaluator, Ability2InteractionCurator
from .ability3 import Ability3Generator, Ability3Evaluator, Ability3Curator
from .ability4 import Ability4Generator, Ability4Evaluator, Ability4Curator
from .ability6 import Ability6Generator, Ability6Evaluator, Ability6Curator

ABILITIES = {
    "ability1": {
        "gen": Ability1Generator,
        "eval": Ability1Evaluator,
        "cur": Ability1Curator,
        "input": "data/wildchat/memories/selected/task4_final_with_selected_memory_id/ability1"
    },
    "ability2_general": {
        "gen": Ability2GeneralGenerator,
        "eval": Ability2GeneralEvaluator,
        "cur": Ability2GeneralCurator,
        "input": "benchmark/dev_with_selected_memory_id/ability2_general"
    },
    "ability2_interaction": {
        "gen": Ability2InteractionGenerator,
        "eval": Ability2InteractionEvaluator,
        "cur": Ability2InteractionCurator,
        "input": "benchmark/dev_with_selected_memory_id/ability2_interaction"
    },
    "ability3": {
        "gen": Ability3Generator,
        "eval": Ability3Evaluator,
        "cur": Ability3Curator,
        "input": "data/wildchat/memories/selected/task4/ability3"
    },
    "ability4": {
        "gen": Ability4Generator,
        "eval": Ability4Evaluator,
        "cur": Ability4Curator,
        "input": "benchmark/dev_with_selected_memory_id/ability4"
    },
    "ability6": {
        "gen": Ability6Generator,
        "eval": Ability6Evaluator,
        "cur": Ability6Curator,
        "input": "data/wildchat/memories/selected/task4_final_with_selected_memory_id/ability1" # uses ability1 as base
    }
}

def run_task4_batch(
    model_under_test: str,
    samples_per_ability: int = 5,
    gen_model_id: str = "gpt-5.2",
    judge_model_id: str = "gpt-5.2"
):
    output_dir = make_run_dir("runs", "task4_batch")
    ensure_dir(output_dir)
    
    log_file_path = os.path.join(output_dir, "run.log")
    events_path = os.path.join(output_dir, "events.jsonl")
    probes_path = os.path.join(output_dir, "probes.jsonl")
    decisions_path = os.path.join(output_dir, "decisions.jsonl")
    reports_path = os.path.join(output_dir, "reports.jsonl")
    dataset_path = os.path.join(output_dir, "dataset.jsonl")
    errors_path = os.path.join(output_dir, "errors.jsonl")

    def log_msg(msg: str, terminal: bool = True):
        """Helper to log to file and optionally print to terminal."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {msg}"
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
        if terminal:
            print(formatted)

    log_fn = lambda event, payload: log_event(events_path, event, payload)
    
    log_msg(f"=== TASK 4 BATCH RUNNER ===")
    log_msg(f"Target Model: {model_under_test}")
    log_msg(f"Output Directory: {output_dir}\n")

    for ability_id, components in ABILITIES.items():
        log_msg(f"--- Running {ability_id} ---")
        
        # Instantiate components
        gen = components["gen"](model_id=gen_model_id)
        evaluator = components["eval"]()
        curator = components["cur"](judge_model_id=judge_model_id)
        
        input_dir = components["input"]
        if not os.path.exists(input_dir):
            log_msg(f"  [Warning] Input directory {input_dir} not found. Skipping {ability_id}.", terminal=True)
            continue
            
        jsonl_files = list_jsonl_files(input_dir)
        random.shuffle(jsonl_files) 
        
        ability_count = 0
        
        for file_path in jsonl_files:
            if ability_count >= samples_per_ability:
                break
                
            records = list(iter_jsonl(file_path))
            random.shuffle(records)
            
            for record in records:
                if ability_count >= samples_per_ability:
                    break
                
                seed_id = resolve_session_id(record) or f"{ability_id}_{ability_count}"
                log_msg(f"  [Sample {ability_count}] Seed: {seed_id}", terminal=True)
                
                try:
                    # 1. Generate Probe
                    log_msg(f"    [1/3] Generating natural probe query...", terminal=False)
                    if ability_id == "ability6":
                        gen_data = gen.generate_query_and_distractors(
                            record,
                            "data/wildchat/memories/selected/task4_final_with_selected_memory_id",
                        )
                        query = gen_data["query"]
                        memories_to_provide = gen_data["distractor_memories"]
                        selected_memory = None 
                    elif ability_id == "ability1":
                        query = gen.generate_query(record)
                        memories_to_provide = record.get("memory_items") or record.get("memories") or []
                        selected_id = record.get("selected_memory_id")
                        selected_memory = record.get("selected_memory") or next(
                            (m for m in memories_to_provide if m.get("memory_id") == selected_id),
                            None,
                        )
                    else:
                        query = gen.generate_query(record)
                        memories_to_provide = record.get("memory_items") or []
                        selected_id = record.get("selected_memory_id")
                        selected_memory = next((m for m in memories_to_provide if m.get("memory_id") == selected_id), None)
                    
                    if not query:
                        log_msg("    [Skip] No query generated.", terminal=True)
                        continue
                    
                    log_msg(f"    => Gen Query: \"{query}\"", terminal=False)
                    if selected_memory:
                        log_msg(f"    => Target Fact: {selected_memory.get('value')}", terminal=False)

                    # 2. Evaluate
                    log_msg(f"    [2/3] Calling target model: {model_under_test}...", terminal=False)
                    dialogue = (record.get("sessions") or [{}])[0].get("turns") or record.get("dialogue") or []
                    
                    if ability_id == "ability6":
                        model_output = evaluator.evaluate_sample(model_under_test, dialogue, query, memories_to_provide)
                    elif ability_id == "ability1":
                        if not selected_memory:
                            log_msg("    [Skip] No selected_memory found.", terminal=True)
                            continue
                        model_output = evaluator.evaluate_sample(model_under_test, selected_memory, query)
                    else:
                        model_output = evaluator.evaluate_sample(model_under_test, dialogue, query)
                        
                    log_msg(f"    => Answer Snippet: {str(model_output.get('answer', ''))[:100]}...", terminal=False)
                    log_msg(f"    => Model Claimed Fact: {model_output.get('used_memory_fact')}", terminal=False)

                    # 3. Score
                    log_msg(f"    [3/3] Judge (gpt-5.2) scoring...", terminal=False)
                    if ability_id == "ability6":
                        scores = curator.score(memories_to_provide, model_output)
                    else:
                        scores = curator.score(selected_memory, model_output)
                        
                    log_msg(f"    => Score Results: {json.dumps({k: v for k, v in scores.items() if isinstance(v, (int, float))})}", terminal=True)
                    if scores.get("reasoning"):
                        log_msg(f"    => Judge Reasoning: {scores.get('reasoning')}", terminal=False)
                    
                    # --- RE-ADDED MISSING SAVE LOGIC ---
                    probe = {
                        "ability": ability_id,
                        "query": query,
                        "selected_memory": selected_memory,
                        "distractor_memories": gen_data["distractor_memories"] if ability_id == "ability6" else []
                    }
                    append_jsonl(probes_path, {"seed_id": seed_id, "ability": ability_id, "probe": probe})
                    append_jsonl(reports_path, {"seed_id": seed_id, "ability": ability_id, "output": model_output})
                    append_jsonl(decisions_path, {"seed_id": seed_id, "ability": ability_id, "scores": scores})
                    
                    # 4. Dataset Entry
                    dataset_entry = {
                        "seed_id": seed_id,
                        "ability": ability_id,
                        "record": {
                            "dialogue": dialogue,
                            "query": query,
                            "selected_memory": selected_memory,
                            "distractor_memories": gen_data["distractor_memories"] if ability_id == "ability6" else []
                        }
                    }
                    append_jsonl(dataset_path, dataset_entry)
                    
                    ability_count += 1
                    log_fn("task4_sample_done", {"seed_id": seed_id, "ability": ability_id})

                except Exception as e:
                    
                    # 4. Dataset Entry
                    dataset_entry = {
                        "seed_id": seed_id,
                        "ability": ability_id,
                        "record": {
                            "dialogue": dialogue,
                            "query": query,
                            "selected_memory": selected_memory,
                            "distractor_memories": gen_data["distractor_memories"] if ability_id == "ability6" else []
                        }
                    }
                    append_jsonl(dataset_path, dataset_entry)
                    
                    ability_count += 1
                    print(f"    [Done] Score: {json.dumps(scores)}")

                except Exception as e:
                    print(f"    [Error] {e}")
                    append_jsonl(errors_path, {"seed_id": seed_id, "ability": ability_id, "error": str(e)})
                    continue

    print(f"\n=== Finished Task 4 Batch Run ===")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model under test")
    parser.add_argument("--samples", type=int, default=5, help="Samples per ability")
    args = parser.parse_args()
    
    run_task4_batch(
        model_under_test=args.model,
        samples_per_ability=args.samples
    )
