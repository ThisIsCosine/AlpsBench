"""
Agent package for benchmark tasks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "run_task_pipeline",
    "run_selected_task",
    "run_task1_from_jsonl_dir",
    "run_task2_from_jsonl_dir",
    "run_task3_from_jsonl_dir",
    "run_task4_from_jsonl_dir",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import task_runner

        return getattr(task_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
