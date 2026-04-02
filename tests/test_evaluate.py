import sys
from pathlib import Path

from src.benchmark.reports import read_jsonl, write_jsonl
from src.benchmark.runner import run_public_evaluation


def test_oracle_evaluation_scores_task3_examples(tmp_path: Path) -> None:
    summary = run_public_evaluation(
        task="task3",
        split="examples",
        oracle=True,
        output_dir=str(tmp_path / "task3_oracle"),
    )
    assert summary["status"] == "ok"
    assert summary["main_metric"] == "accuracy"
    assert summary["main_score"] == 1.0


def test_prediction_file_evaluation_scores_task1_examples(tmp_path: Path) -> None:
    references = read_jsonl(Path("benchmark_data/examples/task1/reference_output.jsonl"))
    predictions = [
        {
            "benchmark_id": row["benchmark_id"],
            "memory_items": row["gold"]["memory_items"],
        }
        for row in references
    ]
    prediction_path = tmp_path / "task1_predictions.jsonl"
    write_jsonl(prediction_path, predictions)

    summary = run_public_evaluation(
        task="task1",
        split="examples",
        predictions_path=str(prediction_path),
        output_dir=str(tmp_path / "task1_eval"),
    )
    assert summary["status"] == "ok"
    assert summary["main_metric"] == "mean_f1"
    assert summary["main_score"] == 1.0


def test_oracle_evaluation_scores_task1_validation(tmp_path: Path) -> None:
    summary = run_public_evaluation(
        task="task1",
        split="validation",
        oracle=True,
        limit=4,
        output_dir=str(tmp_path / "task1_validation_oracle"),
    )
    assert summary["status"] == "ok"
    assert summary["main_metric"] == "mean_f1"
    assert summary["main_score"] == 1.0


def test_test_split_allows_predictions_without_scoring(tmp_path: Path) -> None:
    inputs = read_jsonl(Path("benchmark_data/test/task1/model_input.jsonl"))[:3]
    predictions = [
        {
            "benchmark_id": row["benchmark_id"],
            "memory_items": [],
        }
        for row in inputs
    ]
    prediction_path = tmp_path / "test_predictions.jsonl"
    write_jsonl(prediction_path, predictions)

    summary = run_public_evaluation(
        task="task1",
        split="test",
        predictions_path=str(prediction_path),
        limit=len(inputs),
        output_dir=str(tmp_path / "task1_test_eval"),
    )
    assert summary["status"] == "ok"
    assert summary["local_scoring_available"] is False
    assert summary["num_rows"] == len(inputs)


def test_prediction_file_requires_exact_public_contract_shape(tmp_path: Path) -> None:
    references = read_jsonl(Path("benchmark_data/examples/task1/reference_output.jsonl"))
    invalid_predictions = [
        {
            "benchmark_id": row["benchmark_id"],
            "task": "task1",
            "memory_items": row["gold"]["memory_items"],
        }
        for row in references
    ]
    prediction_path = tmp_path / "invalid_task1_predictions.jsonl"
    write_jsonl(prediction_path, invalid_predictions)

    try:
        run_public_evaluation(
            task="task1",
            split="examples",
            predictions_path=str(prediction_path),
            output_dir=str(tmp_path / "invalid_task1_eval"),
        )
    except ValueError as exc:
        assert "must exactly match" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-contract prediction shape.")


def test_predict_program_evaluation_runs_on_task1_examples(tmp_path: Path) -> None:
    summary = run_public_evaluation(
        task="task1",
        split="examples",
        predict_argv=[sys.executable, "meta_script/mock_empty_memory_adapter.py"],
        output_dir=str(tmp_path / "task1_command_eval"),
    )
    assert summary["status"] == "ok"
    assert summary["main_metric"] == "mean_f1"
    assert summary["main_score"] == 1.0
