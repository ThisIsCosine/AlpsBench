from __future__ import annotations


def describe_task() -> dict[str, object]:
    return {
        "task": "task4",
        "public_abilities": ["ability1", "ability2", "ability3", "ability4", "ability5"],
        "purpose": "Memory-grounded QA benchmark track.",
        "status": "public scaffold landed; public evaluation wiring pending.",
    }
