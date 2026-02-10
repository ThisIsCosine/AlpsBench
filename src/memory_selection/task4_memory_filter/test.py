import os
import sys
import multiprocessing as mp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.memory_selection.task4_memory_filter.filters_runner import RunnerConfig, run_filters


def _run_ability(ability_name: str) -> None:
    output_root = os.path.join("data", "wildchat", "memories", "filtered", "task4", ability_name)
    config = RunnerConfig(
        abilities=[ability_name],
        default_target=200,
        max_passes=2,
        output_root=output_root,
    )
    run_filters(config)


if __name__ == "__main__":
    abilities = [
        "ability1",
        "ability2_general",
        "ability2_interaction",
        "ability3",
        "ability4",
    ]
    processes = [mp.Process(target=_run_ability, args=(ability,)) for ability in abilities]
    for process in processes:
        process.start()
    for process in processes:
        process.join()