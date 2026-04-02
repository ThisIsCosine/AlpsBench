from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

from .reports import read_json, read_jsonl, write_json, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DATA_ROOT = REPO_ROOT / "benchmark_data"
DATASET_VERSION = "v1"
PUBLIC_SPLITS = ("examples", "dev", "validation", "test")
REFERENCE_SPLITS = ("examples", "dev", "validation")
LEAKAGE_CHECK_SPLITS = ("dev", "validation", "test")
TASK3_DISTRACTOR_LEVELS = (100, 300, 500, 700, 1000)
TASK3_PUBLIC_TRACK_NAMES = tuple(f"task3_d{level}" for level in TASK3_DISTRACTOR_LEVELS)
TASK4_ABILITIES = ("ability1", "ability2", "ability3", "ability4", "ability5")
TASK4_PUBLIC_TRACK_NAMES = tuple(f"task4_{ability}" for ability in TASK4_ABILITIES)
PUBLIC_TRACK_NAMES = (
    "task1",
    "task2",
    *TASK3_PUBLIC_TRACK_NAMES,
    *TASK4_PUBLIC_TRACK_NAMES,
)
PUBLIC_SPLIT_ORDER = ("dev", "validation", "test")
PUBLIC_SPLIT_WEIGHTS = {"dev": 1, "validation": 1, "test": 3}
PUBLIC_HASH_SALT = "alpsbench-public-split-v1"
TASK1_EXAMPLE_COUNT = 2
SOURCE_DATASET_ROOT = Path("huggingface") / "Alpsbench" / "dataset"
TASK2_TRACK_NAME = "task2"
TASK2_EXAMPLE_COUNT = 2
TASK4_TRACK_SOURCE_PATHS = {
    "task4_ability1": ("task4", "ability1.json"),
    "task4_ability2": ("task4", "ability2.json"),
    "task4_ability3": ("task4", "ability3.json"),
    "task4_ability4": ("task4", "ability4.json"),
}
TASK4_ABILITY5_SOURCE_FILES = ("final_data_English.json", "final_data_Chinese.json")


def _candidate_alps_data_roots() -> list[Path]:
    candidates: list[Path] = []
    env_value = os.environ.get("ALPS_DATA_ROOT")
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(REPO_ROOT.parent.parent / "Alps_data")
    return candidates


def resolve_alps_data_root() -> Path | None:
    for candidate in _candidate_alps_data_roots():
        if candidate.exists():
            return candidate
    return None


def _source_dataset_root(alps_data_root: Path) -> Path:
    return alps_data_root / SOURCE_DATASET_ROOT


def _task2_source_path(alps_data_root: Path) -> Path:
    results_root = alps_data_root / "task2_general_model_results" / "task2_results"
    if results_root.exists():
        scored_paths = sorted(results_root.rglob("*scored.jsonl"))
        if scored_paths:
            return scored_paths[0]
    return _source_dataset_root(alps_data_root) / "task2" / "task2_dataset.jsonl"


def _task3_source_path(alps_data_root: Path, *, distractors: int) -> Path:
    return _source_dataset_root(alps_data_root) / "task3" / f"task3_dataset_d{distractors}.jsonl"


def _task4_source_paths(alps_data_root: Path, *, track_name: str) -> list[Path]:
    if track_name == "task4_ability5":
        return [
            _source_dataset_root(alps_data_root) / "task4" / "ability5_ei" / filename
            for filename in TASK4_ABILITY5_SOURCE_FILES
        ]
    rel_parts = TASK4_TRACK_SOURCE_PATHS.get(track_name)
    if rel_parts is None:
        raise ValueError(f"Unsupported Task 4 track: {track_name}")
    return [_source_dataset_root(alps_data_root).joinpath(*rel_parts)]


def _normalize_distractor_level(distractors: int | str | None) -> int:
    if distractors is None:
        return TASK3_DISTRACTOR_LEVELS[0]
    try:
        level = int(distractors)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported Task 3 distractor level: {distractors}") from exc
    if level not in TASK3_DISTRACTOR_LEVELS:
        raise ValueError(f"Unsupported Task 3 distractor level: {level}")
    return level


def _track_name_to_task4_ability(track_name: str) -> str:
    if track_name not in TASK4_PUBLIC_TRACK_NAMES:
        raise ValueError(f"Unsupported Task 4 track: {track_name}")
    return track_name.split("_", 1)[1]


def _track_relpath(track_name: str) -> Path:
    if track_name.startswith("task3_d"):
        return Path("task3") / track_name.split("_", 1)[1]
    return Path(track_name)


def _track_data_dir(*, split: str, track_name: str) -> Path:
    return BENCHMARK_DATA_ROOT / split / _track_relpath(track_name)


def _track_hidden_gold_dir(track_name: str) -> Path:
    return REPO_ROOT / "hidden" / "private_gold" / "test" / _track_relpath(track_name)


def _track_name_from_dir(track_dir: Path) -> str:
    if track_dir.parent.name == "task3" and track_dir.name.startswith("d"):
        return f"task3_{track_dir.name}"
    return track_dir.name


def _extract_top_level_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "session_id": row.get("session_id"),
        "canonical_id": row.get("canonical_id"),
        "stratum": row.get("stratum", "UNKNOWN"),
        "ability": row.get("ability"),
    }


def _load_release_track_metadata(track_name: str) -> dict[str, dict[str, Any]]:
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for split in PUBLIC_SPLITS:
        model_input_path = _track_data_dir(split=split, track_name=track_name) / "model_input.jsonl"
        if not model_input_path.exists():
            continue
        for row in read_jsonl(model_input_path):
            benchmark_id = row.get("benchmark_id")
            if benchmark_id:
                metadata_by_id[str(benchmark_id)] = _extract_top_level_metadata(row)
    return metadata_by_id


def _load_release_split_ids(track_name: str) -> dict[str, list[str]] | None:
    split_ids: dict[str, list[str]] = {}
    for split in PUBLIC_SPLITS:
        model_input_path = _track_data_dir(split=split, track_name=track_name) / "model_input.jsonl"
        if not model_input_path.exists():
            return None
        split_ids[split] = [
            str(row["benchmark_id"])
            for row in read_jsonl(model_input_path)
            if row.get("benchmark_id")
        ]
    return split_ids


def _metadata_value(
    benchmark_id: str,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    key: str,
    default: Any,
) -> Any:
    value = metadata_by_id.get(benchmark_id, {}).get(key)
    return default if value is None else value


def _compact_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _task1_manifest_path(alps_data_root: Path) -> Path:
    return (
        alps_data_root
        / "manual_check_full_usable_dataset"
        / "benchmark_construction_check_data"
        / "task1"
        / "manifest.json"
    )


def _task1_output_split_dir(alps_data_root: Path) -> Path:
    return alps_data_root / "output_split" / "output_split"


def _task1_benchmark_id(*, session_id: str, canonical_id: str) -> str:
    return f"{session_id}__{canonical_id}"


def _task1_dialogue_from_sessions(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not sessions:
        return []
    turns = sessions[0].get("turns") or []
    dialogue: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        dialogue.append(
            {
                "role": turn.get("role"),
                "text": turn.get("text"),
            }
        )
    return dialogue


def load_task1_authoritative_pairs() -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
    alps_data_root = resolve_alps_data_root()
    if alps_data_root is None:
        return None

    manifest_path = _task1_manifest_path(alps_data_root)
    output_split_dir = _task1_output_split_dir(alps_data_root)
    if not manifest_path.exists() or not output_split_dir.exists():
        return None

    manifest_rows = read_json(manifest_path)
    if not isinstance(manifest_rows, list):
        raise ValueError(f"Unexpected task1 manifest format: {manifest_path}")

    rich_rows_by_session_id: dict[str, dict[str, Any]] = {}
    for split_path in sorted(output_split_dir.glob("split_part*.json")):
        payload = read_json(split_path)
        for row in payload.get("data", []):
            sessions = row.get("sessions") or []
            if not sessions:
                continue
            session_id = sessions[0].get("session_id")
            if not session_id:
                continue
            rich_rows_by_session_id[str(session_id)] = row

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for manifest_row in manifest_rows:
        session_id = str(manifest_row["session_id"])
        canonical_id = str(manifest_row["canonical_id"])
        stratum = manifest_row.get("stratum", "UNMAPPED")
        rich_row = rich_rows_by_session_id.get(session_id)
        if rich_row is None:
            raise ValueError(f"Missing task1 authoritative row for session_id={session_id}")

        benchmark_id = _task1_benchmark_id(session_id=session_id, canonical_id=canonical_id)
        sessions = rich_row.get("sessions") or []
        input_row = {
            "benchmark_id": benchmark_id,
            "task": "task1",
            "session_id": session_id,
            "canonical_id": canonical_id,
            "stratum": stratum,
            "input": {
                "user_id": rich_row.get("user_id"),
                "line_index": rich_row.get("line_index"),
                "sessions": sessions,
                "dialogue": _task1_dialogue_from_sessions(sessions),
                "metadata": {
                    "match": rich_row.get("match"),
                    "source_dataset_file": None,
                },
            },
        }
        reference_row = {
            "benchmark_id": benchmark_id,
            "task": "task1",
            "session_id": session_id,
            "canonical_id": canonical_id,
            "gold": {
                "memory_items": rich_row.get("memory_items") or [],
                "memory_stage1_candidates": rich_row.get("memory_stage1_candidates") or [],
                "selected_memory_id": rich_row.get("selected_memory_id"),
            },
        }
        pairs.append((input_row, reference_row))

    return pairs


def _hash_rank(benchmark_id: str) -> str:
    payload = f"{PUBLIC_HASH_SALT}:{benchmark_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def partition_public_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    ranked_pairs = [
        (index, _hash_rank(input_row["benchmark_id"]), input_row, reference_row)
        for index, (input_row, reference_row) in enumerate(pairs)
    ]
    ranked_pairs.sort(key=lambda item: (item[1], item[0]))

    total = len(ranked_pairs)
    unit = total // sum(PUBLIC_SPLIT_WEIGHTS.values())
    split_counts = {
        "dev": unit * PUBLIC_SPLIT_WEIGHTS["dev"],
        "validation": unit * PUBLIC_SPLIT_WEIGHTS["validation"],
    }
    split_counts["test"] = total - split_counts["dev"] - split_counts["validation"]

    partitioned: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    offset = 0
    for split in PUBLIC_SPLIT_ORDER:
        next_offset = offset + split_counts[split]
        rows = ranked_pairs[offset:next_offset]
        rows.sort(key=lambda item: item[0])
        partitioned[split] = [(input_row, reference_row) for _, _, input_row, reference_row in rows]
        offset = next_offset

    if offset != total:
        raise ValueError("Task1 partitioning did not consume all rows.")

    return partitioned


def sync_authoritative_task1_release() -> dict[str, Any] | None:
    pairs = load_task1_authoritative_pairs()
    if pairs is None:
        return None

    examples = pairs[:TASK1_EXAMPLE_COUNT]
    partitioned = partition_public_pairs(pairs)
    hidden_gold_dir = REPO_ROOT / "hidden" / "private_gold" / "test" / "task1"

    examples_dir = BENCHMARK_DATA_ROOT / "examples" / "task1"
    examples_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(examples_dir / "model_input.jsonl", [row for row, _ in examples])
    write_jsonl(examples_dir / "reference_output.jsonl", [row for _, row in examples])

    for split in PUBLIC_SPLIT_ORDER:
        split_dir = BENCHMARK_DATA_ROOT / split / "task1"
        split_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(split_dir / "model_input.jsonl", [row for row, _ in partitioned[split]])
        if split == "test":
            reference_path = split_dir / "reference_output.jsonl"
            if reference_path.exists():
                reference_path.unlink()
        else:
            write_jsonl(split_dir / "reference_output.jsonl", [row for _, row in partitioned[split]])

    hidden_gold_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(hidden_gold_dir / "reference_output.jsonl", [row for _, row in partitioned["test"]])

    return {
        "status": "synced",
        "source": "authoritative_task1_pool",
        "examples_rows": len(examples),
        "dev_rows": len(partitioned["dev"]),
        "validation_rows": len(partitioned["validation"]),
        "test_rows": len(partitioned["test"]),
    }


def _write_release_pairs(
    *,
    track_name: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    template_track_name: str | None = None,
    example_count: int = TASK1_EXAMPLE_COUNT,
) -> dict[str, int]:
    split_ids = _load_release_split_ids(template_track_name or track_name)
    if split_ids is None:
        examples = pairs[:example_count]
        partitioned = partition_public_pairs(pairs)
    else:
        pair_by_id = {input_row["benchmark_id"]: (input_row, reference_row) for input_row, reference_row in pairs}
        used_ids: set[str] = set()

        def _pairs_for_ids(requested_ids: list[str], *, split_name: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
            rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
            missing_count = 0
            for benchmark_id in requested_ids:
                pair = pair_by_id.get(benchmark_id)
                if pair is None:
                    missing_count += 1
                    continue
                rows.append(pair)
                used_ids.add(benchmark_id)

            if missing_count:
                replacements = [
                    pair
                    for benchmark_id, pair in pair_by_id.items()
                    if benchmark_id not in used_ids
                ][:missing_count]
                rows.extend(replacements)
                used_ids.update(pair[0]["benchmark_id"] for pair in replacements)

            if len(rows) != len(requested_ids):
                raise ValueError(
                    f"Unable to reconstruct {track_name} {split_name} split: "
                    f"requested={len(requested_ids)} resolved={len(rows)}"
                )
            return rows

        examples = _pairs_for_ids(split_ids["examples"], split_name="examples")
        partitioned = {
            split: _pairs_for_ids(split_ids[split], split_name=split)
            for split in PUBLIC_SPLIT_ORDER
        }

    examples_dir = _track_data_dir(split="examples", track_name=track_name)
    examples_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(examples_dir / "model_input.jsonl", [row for row, _ in examples])
    write_jsonl(examples_dir / "reference_output.jsonl", [row for _, row in examples])

    for split in PUBLIC_SPLIT_ORDER:
        split_dir = _track_data_dir(split=split, track_name=track_name)
        split_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(split_dir / "model_input.jsonl", [row for row, _ in partitioned[split]])
        if split == "test":
            reference_path = split_dir / "reference_output.jsonl"
            if reference_path.exists():
                reference_path.unlink()
        else:
            write_jsonl(split_dir / "reference_output.jsonl", [row for _, row in partitioned[split]])

    hidden_gold_dir = _track_hidden_gold_dir(track_name)
    hidden_gold_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(hidden_gold_dir / "reference_output.jsonl", [row for _, row in partitioned["test"]])

    return {
        "examples_rows": len(examples),
        "dev_rows": len(partitioned["dev"]),
        "validation_rows": len(partitioned["validation"]),
        "test_rows": len(partitioned["test"]),
    }


def load_task2_authoritative_pairs() -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
    alps_data_root = resolve_alps_data_root()
    if alps_data_root is None:
        return None

    source_path = _task2_source_path(alps_data_root)
    if not source_path.exists():
        return None

    metadata_by_id = _load_release_track_metadata(TASK2_TRACK_NAME)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw_row in read_jsonl(source_path):
        row = raw_row.get("entry") if isinstance(raw_row.get("entry"), Mapping) else raw_row
        if not isinstance(row, Mapping):
            continue
        record = row.get("record") or {}
        metadata = row.get("metadata") or {}
        session_id = str(metadata.get("session_id") or "")
        if not session_id:
            continue
        benchmark_id = session_id
        input_row = {
            "benchmark_id": benchmark_id,
            "task": "task2",
            "session_id": _metadata_value(benchmark_id, metadata_by_id, "session_id", session_id),
            "canonical_id": _metadata_value(benchmark_id, metadata_by_id, "canonical_id", session_id),
            "stratum": _metadata_value(benchmark_id, metadata_by_id, "stratum", "UNKNOWN"),
            "input": {
                "record_id": row.get("record_id"),
                "old_dialogue": record.get("old_dialogue") or [],
                "new_dialogue": record.get("new_dialogue") or [],
                "memory": row.get("memory") or [],
                "query": row.get("query"),
                "metadata": metadata,
            },
        }
        reference_row = {
            "benchmark_id": benchmark_id,
            "task": "task2",
            "session_id": input_row["session_id"],
            "canonical_id": input_row["canonical_id"],
            "gold": {
                "answer": row.get("answer") or [],
            },
        }
        pairs.append((input_row, reference_row))
    return pairs


def load_task3_authoritative_pairs(*, distractors: int) -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
    alps_data_root = resolve_alps_data_root()
    if alps_data_root is None:
        return None

    level = _normalize_distractor_level(distractors)
    source_path = _task3_source_path(alps_data_root, distractors=level)
    if not source_path.exists():
        return None

    metadata_by_id = _load_release_track_metadata("task3_d100")
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_benchmark_ids: set[str] = set()
    for row in read_jsonl(source_path):
        record = row.get("record") or {}
        metadata = record.get("metadata") or {}
        benchmark_id = str(metadata.get("session_id") or row.get("seed_id") or "")
        if not benchmark_id:
            continue
        if benchmark_id in seen_benchmark_ids:
            continue
        seen_benchmark_ids.add(benchmark_id)
        input_row = {
            "benchmark_id": benchmark_id,
            "task": "task3",
            "session_id": _metadata_value(benchmark_id, metadata_by_id, "session_id", benchmark_id),
            "canonical_id": _metadata_value(benchmark_id, metadata_by_id, "canonical_id", benchmark_id),
            "stratum": _metadata_value(benchmark_id, metadata_by_id, "stratum", "UNKNOWN"),
            "input": {
                "dialogue": record.get("dialogue") or [],
                "candidate_memories": record.get("candidate_memories") or [],
                "query": record.get("query"),
                "metadata": metadata,
            },
        }
        reference_row = {
            "benchmark_id": benchmark_id,
            "task": "task3",
            "session_id": input_row["session_id"],
            "canonical_id": input_row["canonical_id"],
            "gold": {
                "selected_memory_id": record.get("selected_memory_id"),
                "selected_memory": record.get("selected_memory") or {},
            },
        }
        pairs.append((input_row, reference_row))
    return pairs


def _task4_selected_memory(source_row: Mapping[str, Any]) -> dict[str, Any]:
    extracted_memory = source_row.get("extracted_memory") or []
    memory_key = source_row.get("memory_key")
    if memory_key:
        for item in extracted_memory:
            if isinstance(item, Mapping) and item.get("memory_id") == memory_key:
                return _compact_dict(
                    {
                        "memory_id": item.get("memory_id"),
                        "label": item.get("label"),
                        "value": item.get("value"),
                        "evidence": item.get("evidence"),
                    }
                )
        return {"memory_id": memory_key}
    for item in extracted_memory:
        if isinstance(item, Mapping):
            return _compact_dict(
                {
                    "memory_id": item.get("memory_id"),
                    "label": item.get("label"),
                    "value": item.get("value"),
                    "evidence": item.get("evidence"),
                }
            )
    return {}


def load_task4_authoritative_pairs(*, track_name: str) -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
    alps_data_root = resolve_alps_data_root()
    if alps_data_root is None:
        return None

    source_paths = _task4_source_paths(alps_data_root, track_name=track_name)
    if any(not path.exists() for path in source_paths):
        return None

    metadata_by_id = _load_release_track_metadata(track_name)
    ability = _track_name_to_task4_ability(track_name)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for source_path in source_paths:
        source_rows = read_json(source_path)
        if not isinstance(source_rows, list):
            raise ValueError(f"Unexpected Task 4 source format: {source_path}")
        for row in source_rows:
            benchmark_id = str(row.get("session_id") or "")
            if not benchmark_id:
                continue
            queries = row.get("queries") or []
            selected_memory = _task4_selected_memory(row)
            primary_query = ""
            if queries and isinstance(queries[0], Mapping):
                primary_query = str(queries[0].get("query") or "")
            model_input: dict[str, Any] = {"query": primary_query}
            if ability == "ability1":
                model_input["selected_memory"] = selected_memory
            else:
                model_input["dialogue_history"] = row.get("conversation") or []
            input_row = {
                "benchmark_id": benchmark_id,
                "task": "task4",
                "ability": _metadata_value(benchmark_id, metadata_by_id, "ability", ability),
                "session_id": _metadata_value(benchmark_id, metadata_by_id, "session_id", benchmark_id),
                "canonical_id": _metadata_value(benchmark_id, metadata_by_id, "canonical_id", benchmark_id),
                "stratum": _metadata_value(benchmark_id, metadata_by_id, "stratum", "UNKNOWN"),
                "input": {
                    "query": primary_query,
                    "queries": queries,
                    "model_input": model_input,
                    "audit_context": {
                        "conversation": row.get("conversation") or [],
                        "source_dataset_file": str(source_path),
                    },
                },
            }
            reference_row = {
                "benchmark_id": benchmark_id,
                "task": "task4",
                "ability": input_row["ability"],
                "session_id": input_row["session_id"],
                "canonical_id": input_row["canonical_id"],
                "gold": {
                    "memory_key": row.get("memory_key"),
                    "selected_memory": selected_memory,
                    "extracted_memory": row.get("extracted_memory") or [],
                    "conversation": row.get("conversation") or [],
                },
            }
            pairs.append((input_row, reference_row))
    return pairs


def load_authoritative_public_pairs(track_name: str) -> list[tuple[dict[str, Any], dict[str, Any]]] | None:
    if track_name == "task1":
        return load_task1_authoritative_pairs()
    if track_name == "task2":
        return load_task2_authoritative_pairs()
    if track_name.startswith("task3_d"):
        return load_task3_authoritative_pairs(distractors=int(track_name.split("_d", 1)[1]))
    if track_name in TASK4_PUBLIC_TRACK_NAMES:
        return load_task4_authoritative_pairs(track_name=track_name)
    raise ValueError(f"Unsupported public track name: {track_name}")


def sync_authoritative_task2_release() -> dict[str, Any] | None:
    pairs = load_task2_authoritative_pairs()
    if pairs is None:
        return None
    source_label = "authoritative_task2_dataset"
    alps_data_root = resolve_alps_data_root()
    if alps_data_root is not None:
        source_path = _task2_source_path(alps_data_root)
        if source_path.name.endswith("scored.jsonl"):
            source_label = "authoritative_task2_scored_entries"
    counts = _write_release_pairs(track_name="task2", pairs=pairs, example_count=TASK2_EXAMPLE_COUNT)
    return {
        "status": "synced",
        "source": source_label,
        **counts,
    }


def sync_authoritative_task3_release() -> dict[str, Any] | None:
    summary: dict[str, Any] = {
        "status": "synced",
        "source": "authoritative_task3_datasets",
        "tracks": {},
    }
    any_synced = False
    for level in TASK3_DISTRACTOR_LEVELS:
        track_name = f"task3_d{level}"
        pairs = load_task3_authoritative_pairs(distractors=level)
        if pairs is None:
            continue
        counts = _write_release_pairs(
            track_name=track_name,
            pairs=pairs,
            template_track_name="task3_d100",
            example_count=TASK2_EXAMPLE_COUNT,
        )
        summary["tracks"][track_name] = counts
        any_synced = True
    return summary if any_synced else None


def sync_authoritative_task4_release() -> dict[str, Any] | None:
    summary: dict[str, Any] = {
        "status": "synced",
        "source": "authoritative_task4_datasets",
        "tracks": {},
    }
    any_synced = False
    for track_name in TASK4_PUBLIC_TRACK_NAMES:
        pairs = load_task4_authoritative_pairs(track_name=track_name)
        if pairs is None:
            continue
        counts = _write_release_pairs(
            track_name=track_name,
            pairs=pairs,
            example_count=TASK2_EXAMPLE_COUNT,
        )
        summary["tracks"][track_name] = counts
        any_synced = True
    return summary if any_synced else None

def resolve_public_track_name(
    task: str,
    ability: str | None = None,
    distractors: int | str | None = None,
) -> str:
    task = task.strip().lower()
    if task == "task1":
        return "task1"
    if task == "task2":
        return "task2"
    if task == "task3":
        return f"task3_d{_normalize_distractor_level(distractors)}"
    if task.startswith("task3_d"):
        _normalize_distractor_level(task.split("_d", 1)[1])
        return task
    if task == "task4":
        if not ability:
            raise ValueError("Task 4 requires an ability value from ability1..ability5.")
        ability = ability.strip().lower()
        if ability not in TASK4_ABILITIES:
            raise ValueError(f"Unsupported public Task 4 ability: {ability}")
        return f"task4_{ability}"
    if task in PUBLIC_TRACK_NAMES:
        return task
    raise ValueError(f"Unknown public task name: {task}")


def resolve_public_track_dir(
    split: str,
    task: str,
    ability: str | None = None,
    distractors: int | str | None = None,
) -> Path:
    if split not in PUBLIC_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    track_name = resolve_public_track_name(task, ability=ability, distractors=distractors)
    return _track_data_dir(split=split, track_name=track_name)


def iter_public_track_dirs(splits: Sequence[str] = PUBLIC_SPLITS) -> Iterator[tuple[str, str, Path]]:
    for split in splits:
        if split not in PUBLIC_SPLITS:
            raise ValueError(f"Unknown split: {split}")
        for track_name in PUBLIC_TRACK_NAMES:
            yield split, track_name, _track_data_dir(split=split, track_name=track_name)


def _touch_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def _require_existing_file(path: Path) -> None:
    _require(path.exists(), f"Missing required public data file: {path}")


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_jsonl(path))


def _remove_legacy_task3_layout() -> None:
    for split in PUBLIC_SPLITS:
        task3_root = BENCHMARK_DATA_ROOT / split / "task3"
        if task3_root.exists():
            shutil.rmtree(task3_root)
        for track_name in TASK3_PUBLIC_TRACK_NAMES:
            legacy_dir = BENCHMARK_DATA_ROOT / split / track_name
            if legacy_dir.exists():
                shutil.rmtree(legacy_dir)
    hidden_root = REPO_ROOT / "hidden" / "private_gold" / "test"
    task3_hidden_root = hidden_root / "task3"
    if task3_hidden_root.exists():
        shutil.rmtree(task3_hidden_root)
    for track_name in TASK3_PUBLIC_TRACK_NAMES:
        legacy_dir = hidden_root / track_name
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)


def _build_split_summary(split: str) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for _, track_name, track_dir in iter_public_track_dirs((split,)):
        row_summary = {"model_input_rows": _count_rows(track_dir / "model_input.jsonl")}
        reference_path = track_dir / "reference_output.jsonl"
        if reference_path.exists():
            row_summary["reference_output_rows"] = _count_rows(reference_path)
        summary[track_name] = row_summary
    return summary


def _build_split_manifest() -> Dict[str, Any]:
    tracks: dict[str, dict[str, int]] = {}
    for track_name in PUBLIC_TRACK_NAMES:
        dev_rows = _count_rows(_track_data_dir(split="dev", track_name=track_name) / "model_input.jsonl")
        validation_rows = _count_rows(_track_data_dir(split="validation", track_name=track_name) / "model_input.jsonl")
        test_rows = _count_rows(_track_data_dir(split="test", track_name=track_name) / "model_input.jsonl")
        tracks[track_name] = {
            "source_rows": dev_rows + validation_rows + test_rows,
            "dev_rows": dev_rows,
            "validation_rows": validation_rows,
            "test_rows": test_rows,
        }

    return {
        "status": "ok",
        "source_split": "dev",
        "hash_salt": PUBLIC_HASH_SALT,
        "weights": PUBLIC_SPLIT_WEIGHTS,
        "tracks": tracks,
    }


def _collect_track_benchmark_ids(split: str, track_name: str) -> set[str]:
    track_dir = _track_data_dir(split=split, track_name=track_name)
    rows = read_jsonl(track_dir / "model_input.jsonl")
    return {str(row.get("benchmark_id")) for row in rows if row.get("benchmark_id")}


def _validate_split_leakage(splits: Sequence[str]) -> Dict[str, Any]:
    selected_splits = [split for split in LEAKAGE_CHECK_SPLITS if split in splits]
    overlaps: list[dict[str, Any]] = []

    for track_name in PUBLIC_TRACK_NAMES:
        ids_by_split = {
            split: _collect_track_benchmark_ids(split, track_name)
            for split in selected_splits
        }
        for left_index, left_split in enumerate(selected_splits):
            for right_split in selected_splits[left_index + 1 :]:
                overlap_ids = sorted(ids_by_split[left_split] & ids_by_split[right_split])
                if overlap_ids:
                    overlaps.append(
                        {
                            "track": track_name,
                            "left_split": left_split,
                            "right_split": right_split,
                            "count": len(overlap_ids),
                            "sample_ids": overlap_ids[:5],
                        }
                    )

    if overlaps:
        raise ValueError(f"Split leakage detected: {overlaps[0]}")
    return {
        "status": "ok",
        "checked_splits": selected_splits,
        "checked_pairs": (len(selected_splits) * (len(selected_splits) - 1)) // 2,
    }


def build_public_data_layout(*, overwrite: bool = False) -> Dict[str, Any]:
    created: list[str] = []
    _remove_legacy_task3_layout()
    task1_sync = sync_authoritative_task1_release()
    task2_sync = sync_authoritative_task2_release()
    task3_sync = sync_authoritative_task3_release()
    task4_sync = sync_authoritative_task4_release()

    for split in ("examples", "dev", "validation"):
        for _, track_name, target_dir in iter_public_track_dirs((split,)):
            target_dir.mkdir(parents=True, exist_ok=True)
            created.append(str(target_dir.relative_to(REPO_ROOT)).replace("\\", "/"))
            _require_existing_file(target_dir / "model_input.jsonl")
            _require_existing_file(target_dir / "reference_output.jsonl")

    for _, track_name, target_dir in iter_public_track_dirs(("test",)):
        target_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(target_dir.relative_to(REPO_ROOT)).replace("\\", "/"))
        _touch_file(target_dir / "model_input.jsonl")

    artifacts_dir = BENCHMARK_DATA_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    created.append(str(artifacts_dir.relative_to(REPO_ROOT)).replace("\\", "/"))

    raw_export_index_payload = {
        "status": "public-layout",
        "dataset_version": DATASET_VERSION,
        "note": "The public repository no longer rebuilds from legacy intermediate directories. Canonical raw source remains under Alps_data_v1/.",
    }
    write_json(artifacts_dir / "raw_export_index.json", raw_export_index_payload)

    build_summary = {
        "status": "ok",
        "dataset_version": DATASET_VERSION,
        "source": "public_layout",
        "splits": {
            "examples": _build_split_summary("examples"),
            "dev": _build_split_summary("dev"),
            "validation": _build_split_summary("validation"),
            "test": _build_split_summary("test"),
        },
    }
    write_json(artifacts_dir / "build_summary.json", build_summary)

    example_smoke_report = {
        "status": "ok",
        "dataset_version": DATASET_VERSION,
        "source": "public_layout",
        "examples": _build_split_summary("examples"),
    }
    write_json(artifacts_dir / "example_smoke_report.json", example_smoke_report)
    split_manifest = _build_split_manifest()
    write_json(artifacts_dir / "split_manifest.json", split_manifest)
    hidden_gold_root = REPO_ROOT / "hidden" / "private_gold"
    hidden_gold_root.mkdir(parents=True, exist_ok=True)
    write_json(hidden_gold_root / "split_manifest.json", split_manifest)

    manifest = {
        "status": "ok",
        "dataset_version": DATASET_VERSION,
        "created_dirs": sorted(set(created)),
        "public_data_is_canonical": True,
        "overwrite_requested": overwrite,
    }
    if task1_sync is not None:
        manifest["task1_authoritative_sync"] = task1_sync
    if task2_sync is not None:
        manifest["task2_authoritative_sync"] = task2_sync
    if task3_sync is not None:
        manifest["task3_authoritative_sync"] = task3_sync
    if task4_sync is not None:
        manifest["task4_authoritative_sync"] = task4_sync
    write_json(artifacts_dir / "public_layout_manifest.json", manifest)
    return manifest


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_reference_pair(track_dir: Path) -> Dict[str, int]:
    track_name = _track_name_from_dir(track_dir)
    model_input_path = track_dir / "model_input.jsonl"
    reference_output_path = track_dir / "reference_output.jsonl"
    _require(model_input_path.exists(), f"Missing model_input.jsonl in {track_dir}")
    _require(reference_output_path.exists(), f"Missing reference_output.jsonl in {track_dir}")

    input_rows = read_jsonl(model_input_path)
    reference_rows = read_jsonl(reference_output_path)

    _require(bool(input_rows), f"No rows found in {model_input_path}")
    _require(bool(reference_rows), f"No rows found in {reference_output_path}")

    input_ids = [row.get("benchmark_id") for row in input_rows]
    reference_ids = [row.get("benchmark_id") for row in reference_rows]

    _require(len(set(input_ids)) == len(input_ids), f"Duplicate benchmark_id values in {model_input_path}")
    _require(
        len(set(reference_ids)) == len(reference_ids),
        f"Duplicate benchmark_id values in {reference_output_path}",
    )
    _require(
        set(input_ids) == set(reference_ids),
        f"Input/reference benchmark_id mismatch in {track_dir}",
    )
    if track_name == "task1":
        has_dialogue = any((row.get("input", {}).get("dialogue") or []) for row in input_rows)
        has_gold = any((row.get("gold", {}).get("memory_items") or []) for row in reference_rows)
        _require(has_dialogue, f"Task1 input rows are all empty dialogue in {track_dir}")
        _require(has_gold, f"Task1 reference rows are all empty memory_items in {track_dir}")
    if track_name == "task2":
        has_new_dialogue = any((row.get("input", {}).get("new_dialogue") or []) for row in input_rows)
        has_gold = any((row.get("gold", {}).get("answer") or []) for row in reference_rows)
        _require(has_new_dialogue, f"Task2 input rows are all empty new_dialogue in {track_dir}")
        _require(has_gold, f"Task2 reference rows are all empty answer in {track_dir}")
    if track_name.startswith("task3_d"):
        has_query = any(bool(row.get("input", {}).get("query")) for row in input_rows)
        has_candidates = any((row.get("input", {}).get("candidate_memories") or []) for row in input_rows)
        _require(has_query, f"Task3 input rows are all missing query in {track_dir}")
        _require(has_candidates, f"Task3 input rows are all empty candidate_memories in {track_dir}")
    if track_name.startswith("task4_"):
        has_query = any(bool(row.get("input", {}).get("query")) for row in input_rows)
        has_selected_memory = any((row.get("gold", {}).get("selected_memory") or {}) for row in reference_rows)
        _require(has_query, f"Task4 input rows are all missing query in {track_dir}")
        _require(has_selected_memory, f"Task4 reference rows are all missing selected_memory in {track_dir}")
    return {"input_rows": len(input_rows), "reference_rows": len(reference_rows)}


def _validate_test_track(track_dir: Path) -> Dict[str, int]:
    track_name = _track_name_from_dir(track_dir)
    model_input_path = track_dir / "model_input.jsonl"
    reference_output_path = track_dir / "reference_output.jsonl"
    _require(model_input_path.exists(), f"Missing model_input.jsonl in {track_dir}")
    _require(not reference_output_path.exists(), f"Public test track must not expose reference_output.jsonl in {track_dir}")
    input_rows = read_jsonl(model_input_path)
    if track_name == "task1":
        has_dialogue = any((row.get("input", {}).get("dialogue") or []) for row in input_rows)
        _require(has_dialogue, f"Task1 test input rows are all empty dialogue in {track_dir}")
    if track_name == "task2":
        has_new_dialogue = any((row.get("input", {}).get("new_dialogue") or []) for row in input_rows)
        _require(has_new_dialogue, f"Task2 test input rows are all empty new_dialogue in {track_dir}")
    if track_name.startswith("task3_d"):
        has_candidates = any((row.get("input", {}).get("candidate_memories") or []) for row in input_rows)
        _require(has_candidates, f"Task3 test input rows are all empty candidate_memories in {track_dir}")
    if track_name.startswith("task4_"):
        has_query = any(bool(row.get("input", {}).get("query")) for row in input_rows)
        _require(has_query, f"Task4 test input rows are all missing query in {track_dir}")
    return {"input_rows": len(input_rows)}


def validate_public_data_layout(*, splits: Sequence[str] = PUBLIC_SPLITS) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "status": "ok",
        "dataset_version": DATASET_VERSION,
        "splits": {},
    }

    for split in splits:
        split_summary: Dict[str, Any] = {}
        for _, track_name, track_dir in iter_public_track_dirs((split,)):
            if split in REFERENCE_SPLITS:
                split_summary[track_name] = _validate_reference_pair(track_dir)
            else:
                split_summary[track_name] = _validate_test_track(track_dir)
        summary["splits"][split] = split_summary

    summary["split_leakage"] = _validate_split_leakage(splits)
    return summary
