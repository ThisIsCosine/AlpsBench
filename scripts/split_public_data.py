from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.data import PUBLIC_TRACK_NAMES, load_authoritative_public_pairs  # noqa: E402
from benchmark.reports import read_jsonl, write_json, write_jsonl  # noqa: E402


SOURCE_SPLIT = "dev"
PUBLIC_BENCHMARK_ROOT = REPO_ROOT / "benchmark_data"
HIDDEN_GOLD_ROOT = REPO_ROOT / "hidden" / "private_gold" / "test"
HASH_SALT = "alpsbench-public-split-v1"
SPLIT_ORDER = ("dev", "validation", "test")
SPLIT_WEIGHTS = {"dev": 1, "validation": 1, "test": 3}


def _hash_rank(benchmark_id: str) -> str:
    payload = f"{HASH_SALT}:{benchmark_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _partition_pairs(
    inputs: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    reference_by_id = {row["benchmark_id"]: row for row in references}
    paired_rows: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []

    for index, input_row in enumerate(inputs):
        benchmark_id = input_row["benchmark_id"]
        reference_row = reference_by_id.get(benchmark_id)
        if reference_row is None:
            raise ValueError(f"Missing reference row for {benchmark_id}.")
        paired_rows.append((index, _hash_rank(benchmark_id), input_row, reference_row))

    if len(paired_rows) != len(reference_by_id):
        raise ValueError("Input/reference row counts do not match.")

    paired_rows.sort(key=lambda item: (item[1], item[0]))
    total = len(paired_rows)
    unit = total // sum(SPLIT_WEIGHTS.values())
    split_counts = {
        "dev": unit * SPLIT_WEIGHTS["dev"],
        "validation": unit * SPLIT_WEIGHTS["validation"],
    }
    split_counts["test"] = total - split_counts["dev"] - split_counts["validation"]

    partitioned: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    offset = 0
    for split in SPLIT_ORDER:
        next_offset = offset + split_counts[split]
        rows = paired_rows[offset:next_offset]
        rows.sort(key=lambda item: item[0])
        partitioned[split] = [(input_row, reference_row) for _, _, input_row, reference_row in rows]
        offset = next_offset

    if offset != total:
        raise ValueError("Split allocation did not consume all rows.")

    return partitioned


def _reset_track_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_source_rows(track_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    authoritative_pairs = load_authoritative_public_pairs(track_name)
    if authoritative_pairs is not None:
        return (
            [input_row for input_row, _ in authoritative_pairs],
            [reference_row for _, reference_row in authoritative_pairs],
        )

    dev_track_dir = PUBLIC_BENCHMARK_ROOT / "dev" / track_name
    validation_track_dir = PUBLIC_BENCHMARK_ROOT / "validation" / track_name
    test_track_dir = PUBLIC_BENCHMARK_ROOT / "test" / track_name
    hidden_gold_track_dir = HIDDEN_GOLD_ROOT / track_name

    if (
        (validation_track_dir / "model_input.jsonl").exists()
        and (validation_track_dir / "reference_output.jsonl").exists()
        and (test_track_dir / "model_input.jsonl").exists()
        and (hidden_gold_track_dir / "reference_output.jsonl").exists()
    ):
        return (
            [
                *read_jsonl(dev_track_dir / "model_input.jsonl"),
                *read_jsonl(validation_track_dir / "model_input.jsonl"),
                *read_jsonl(test_track_dir / "model_input.jsonl"),
            ],
            [
                *read_jsonl(dev_track_dir / "reference_output.jsonl"),
                *read_jsonl(validation_track_dir / "reference_output.jsonl"),
                *read_jsonl(hidden_gold_track_dir / "reference_output.jsonl"),
            ],
        )

    return (
        read_jsonl(dev_track_dir / "model_input.jsonl"),
        read_jsonl(dev_track_dir / "reference_output.jsonl"),
    )


def split_public_data(*, force: bool = False, track_names: list[str] | None = None) -> dict[str, Any]:
    source_root = PUBLIC_BENCHMARK_ROOT / SOURCE_SPLIT
    validation_root = PUBLIC_BENCHMARK_ROOT / "validation"
    test_root = PUBLIC_BENCHMARK_ROOT / "test"
    selected_tracks = track_names or list(PUBLIC_TRACK_NAMES)

    if validation_root.exists() and any(validation_root.iterdir()) and not force:
        raise ValueError("benchmark_data/validation already exists and is not empty. Use --force to overwrite.")
    if test_root.exists() and any(test_root.iterdir()) and not force:
        raise ValueError("benchmark_data/test already exists and is not empty. Use --force to overwrite.")

    summary: dict[str, Any] = {
        "status": "ok",
        "source_split": SOURCE_SPLIT,
        "hash_salt": HASH_SALT,
        "weights": SPLIT_WEIGHTS,
        "tracks": {},
    }

    for track_name in selected_tracks:
        source_track_dir = source_root / track_name
        inputs, references = _load_source_rows(track_name)
        partitions = _partition_pairs(inputs, references)

        dev_track_dir = source_root / track_name
        validation_track_dir = validation_root / track_name
        test_track_dir = test_root / track_name
        hidden_gold_track_dir = HIDDEN_GOLD_ROOT / track_name

        _reset_track_dir(dev_track_dir)
        _reset_track_dir(validation_track_dir)
        _reset_track_dir(test_track_dir)
        _reset_track_dir(hidden_gold_track_dir)

        write_jsonl(dev_track_dir / "model_input.jsonl", [row for row, _ in partitions["dev"]])
        write_jsonl(dev_track_dir / "reference_output.jsonl", [row for _, row in partitions["dev"]])

        write_jsonl(validation_track_dir / "model_input.jsonl", [row for row, _ in partitions["validation"]])
        write_jsonl(validation_track_dir / "reference_output.jsonl", [row for _, row in partitions["validation"]])

        write_jsonl(test_track_dir / "model_input.jsonl", [row for row, _ in partitions["test"]])
        write_jsonl(hidden_gold_track_dir / "reference_output.jsonl", [row for _, row in partitions["test"]])

        summary["tracks"][track_name] = {
            "source_rows": len(inputs),
            "dev_rows": len(partitions["dev"]),
            "validation_rows": len(partitions["validation"]),
            "test_rows": len(partitions["test"]),
        }

    artifacts_dir = PUBLIC_BENCHMARK_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifacts_dir / "split_manifest.json", summary)
    write_json(HIDDEN_GOLD_ROOT.parent / "split_manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the public dev data into dev/validation/test partitions.")
    parser.add_argument("--force", action="store_true", help="Overwrite non-empty validation/test directories.")
    parser.add_argument(
        "--track",
        action="append",
        choices=list(PUBLIC_TRACK_NAMES),
        help="Split only the selected track. Repeatable.",
    )
    args = parser.parse_args()

    summary = split_public_data(force=args.force, track_names=args.track)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
