from src.benchmark.data import (
    DATASET_VERSION,
    PUBLIC_TRACK_NAMES,
    build_public_data_layout,
    resolve_public_track_name,
    validate_public_data_layout,
)


def test_public_track_name_resolution() -> None:
    assert resolve_public_track_name("task1") == "task1"
    assert resolve_public_track_name("task3") == "task3_d100"
    assert resolve_public_track_name("task4", ability="ability3") == "task4_ability3"


def test_public_track_count() -> None:
    assert len(PUBLIC_TRACK_NAMES) == 8


def test_examples_dev_and_validation_runs() -> None:
    summary = validate_public_data_layout(splits=("examples", "dev", "validation"))
    assert summary["status"] == "ok"
    assert summary["dataset_version"] == DATASET_VERSION
    assert summary["split_leakage"]["status"] == "ok"


def test_build_public_data_layout_runs_without_legacy_sources() -> None:
    summary = build_public_data_layout()
    assert summary["status"] == "ok"
    assert summary["public_data_is_canonical"] is True
    assert summary["dataset_version"] == DATASET_VERSION
