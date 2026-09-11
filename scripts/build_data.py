from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.runner import build_data  # noqa: E402


V4_REQUIRED_FILES = {
    "task1_dataset.jsonl",
    "task2_dataset.jsonl",
    "task3_dataset_d100.jsonl",
    "ability1.json",
    "ability2.json",
    "ability3.json",
    "ability4.json",
    "ability5.json",
    "annotations_selected_clean_final_v4.json",
}


def _locate_v4_root(path: Path) -> Path:
    candidates = (path, path / "Alps_data_final_v4")
    for candidate in candidates:
        if candidate.is_dir() and V4_REQUIRED_FILES.issubset(
            item.name for item in candidate.iterdir()
        ):
            return candidate
    missing = sorted(V4_REQUIRED_FILES - {item.name for item in path.iterdir()}) if path.is_dir() else []
    raise ValueError(f"Could not find a complete Alps_data_final_v4 bundle under {path}; missing: {missing}")


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename}")
        bundle.extractall(destination)


def _build_from_source(source: Path, *, overwrite: bool) -> dict[str, object]:
    if not overwrite:
        raise ValueError(
            "Importing a source bundle rewrites benchmark_data; pass --overwrite explicitly."
        )
    if source.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="alpsbench-v4-") as tmp:
            extracted = Path(tmp)
            _safe_extract(source, extracted)
            source_root = _locate_v4_root(extracted)
            os.environ["ALPS_DATA_ROOT"] = str(source_root)
            return build_data(overwrite=True)
    source_root = _locate_v4_root(source)
    os.environ["ALPS_DATA_ROOT"] = str(source_root)
    return build_data(overwrite=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public benchmark data layout.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing public-layout files.")
    parser.add_argument(
        "--source",
        type=Path,
        help="Path to Alps_data_final_v4.zip or its extracted directory.",
    )
    args = parser.parse_args()

    summary = (
        _build_from_source(args.source.expanduser().resolve(), overwrite=args.overwrite)
        if args.source
        else build_data(overwrite=args.overwrite)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
