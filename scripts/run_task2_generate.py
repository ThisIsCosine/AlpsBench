from src.agents.task_runner import run_task2_from_jsonl_dir
import json

cfg = json.load(open('configs/api.json','r',encoding='utf-8'))
cfg['task2_controls'] = {
    'no_personal_turns':8,
    'fake_memory':3,
    'new_memory':1,
    'conflict_memory':1
}

print("before run")
run_task2_from_jsonl_dir(
    jsonl_dir='data/human_annotation_with_selected_memory_id/task4_final_human_annotation_with_selected_memory_id',
    model_list=[],
    gen_call_model=None,
    eval_call_model=None,
    max_samples=10000,
    max_attempts_per_seed=10,
    generator_model_id='gpt-5.21111',
    api_config=cfg
)
print("after run")
