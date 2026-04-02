from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .data import build_public_data_layout, resolve_public_track_dir, resolve_public_track_name, validate_public_data_layout
from .metrics import score_prediction_row, summarize_score_rows, supports_local_scoring
from .reports import ensure_dir, read_jsonl, write_json, write_jsonl


def build_data(*, overwrite: bool = False) -> Dict[str, Any]:
    return build_public_data_layout(overwrite=overwrite)


def validate_data() -> Dict[str, Any]:
    return validate_public_data_layout()


def smoke_examples() -> Dict[str, Any]:
    return validate_public_data_layout(splits=("examples",))


def prepare_evaluation_run(
    *,
    task: str,
    split: str,
    ability: str | None = None,
    distractors: int | None = None,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    track_dir = resolve_public_track_dir(split, task, ability=ability, distractors=distractors)
    track_name = resolve_public_track_name(task, ability=ability, distractors=distractors)
    run_dir = Path(output_dir) if output_dir else Path("runs") / "public" / split / track_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "prepared",
        "task": task,
        "split": split,
        "ability": ability,
        "distractors": distractors,
        "track_dir": str(track_dir),
        "model_input_path": str(track_dir / "model_input.jsonl"),
        "reference_output_path": str(track_dir / "reference_output.jsonl"),
        "local_scoring_expected": supports_local_scoring(split),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _expected_prediction_keys(*, task: str) -> set[str]:
    if task in {"task1", "task2"}:
        return {"benchmark_id", "memory_items"}
    if task == "task3":
        return {"benchmark_id", "answer", "reason", "selected_memory_id"}
    if task == "task4":
        return {"benchmark_id", "answer", "used_memory_fact"}
    raise ValueError(f"Unsupported task: {task}")


def _validate_exact_prediction_row(
    *,
    task: str,
    input_row: Mapping[str, Any],
    raw_prediction: Mapping[str, Any],
    ability: str | None,
) -> None:
    del ability
    expected_top_level = _expected_prediction_keys(task=task)
    actual_top_level = set(raw_prediction.keys())
    if actual_top_level != expected_top_level:
        raise ValueError(
            f"Prediction keys for {input_row.get('benchmark_id')} must exactly match "
            f"{sorted(expected_top_level)}; got {sorted(actual_top_level)}."
        )

    if raw_prediction.get("benchmark_id") != input_row.get("benchmark_id"):
        raise ValueError(
            f"Prediction field 'benchmark_id' for {input_row.get('benchmark_id')} must equal "
            f"the input benchmark_id {input_row.get('benchmark_id')!r}; got {raw_prediction.get('benchmark_id')!r}."
        )

    if task in {"task1", "task2"} and not isinstance(raw_prediction.get("memory_items"), list):
        raise ValueError(f"Prediction memory_items must be a JSON array for {input_row.get('benchmark_id')}.")
    if task == "task3":
        if not isinstance(raw_prediction.get("answer"), str):
            raise ValueError(f"Prediction answer must be a string for {input_row.get('benchmark_id')}.")
        if not isinstance(raw_prediction.get("reason"), str):
            raise ValueError(f"Prediction reason must be a string for {input_row.get('benchmark_id')}.")
        if not isinstance(raw_prediction.get("selected_memory_id"), str):
            raise ValueError(f"Prediction selected_memory_id must be a string for {input_row.get('benchmark_id')}.")
    if task == "task4":
        if not isinstance(raw_prediction.get("answer"), str):
            raise ValueError(f"Prediction answer must be a string for {input_row.get('benchmark_id')}.")
        if not isinstance(raw_prediction.get("used_memory_fact"), str):
            raise ValueError(f"Prediction used_memory_fact must be a string for {input_row.get('benchmark_id')}.")


def _extract_prediction_for_scoring(
    *,
    task: str,
    raw_prediction: Mapping[str, Any],
    ability: str | None,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "benchmark_id": raw_prediction["benchmark_id"],
        "task": task,
    }

    if task == "task1":
        normalized["memory_items"] = raw_prediction.get("memory_items") or []
    elif task == "task2":
        normalized["memory_items"] = raw_prediction.get("memory_items") or []
    elif task == "task3":
        normalized["answer"] = raw_prediction.get("answer", "")
        normalized["reason"] = raw_prediction.get("reason", "")
        normalized["selected_memory_id"] = raw_prediction.get("selected_memory_id")
    elif task == "task4":
        if ability:
            normalized["ability"] = ability
        normalized["answer"] = raw_prediction.get("answer", "")
        normalized["used_memory_fact"] = raw_prediction.get("used_memory_fact", "")
    else:
        raise ValueError(f"Unsupported task: {task}")

    return normalized


def _build_oracle_prediction_row(*, task: str, reference: Mapping[str, Any]) -> dict[str, Any]:
    gold = reference.get("gold") or {}
    benchmark_id = reference["benchmark_id"]
    if task == "task1":
        return {
            "benchmark_id": benchmark_id,
            "memory_items": gold.get("memory_items") or [],
        }
    if task == "task2":
        return {
            "benchmark_id": benchmark_id,
            "memory_items": gold.get("answer") or [],
        }
    if task == "task3":
        selected_memory = gold.get("selected_memory") or {}
        return {
            "benchmark_id": benchmark_id,
            "answer": str(selected_memory.get("value") or ""),
            "reason": "oracle",
            "selected_memory_id": str(gold.get("selected_memory_id") or ""),
        }
    if task == "task4":
        selected_memory = gold.get("selected_memory") or {}
        selected_value = str(selected_memory.get("value") or "")
        return {
            "benchmark_id": benchmark_id,
            "answer": selected_value,
            "used_memory_fact": selected_value,
        }
    raise ValueError(f"Unsupported task: {task}")


def _load_track_rows(track_dir: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = read_jsonl(track_dir / "model_input.jsonl")
    references = read_jsonl(track_dir / "reference_output.jsonl")
    if limit is not None:
        inputs = inputs[:limit]
        if references:
            reference_ids = {row.get("benchmark_id") for row in inputs}
            references = [row for row in references if row.get("benchmark_id") in reference_ids]
    return inputs, references


def _load_prediction_file(
    *,
    task: str,
    inputs: list[dict[str, Any]],
    path: str,
    ability: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_by_id = {row["benchmark_id"]: row for row in inputs}
    raw_predictions = read_jsonl(Path(path))
    seen_ids: set[str] = set()
    exact_predictions: list[dict[str, Any]] = []
    scoring_predictions: list[dict[str, Any]] = []

    for raw_prediction in raw_predictions:
        benchmark_id = raw_prediction.get("benchmark_id")
        if benchmark_id not in input_by_id:
            raise ValueError(f"Prediction benchmark_id not found in input split: {benchmark_id}")
        if benchmark_id in seen_ids:
            raise ValueError(f"Duplicate prediction benchmark_id: {benchmark_id}")
        seen_ids.add(benchmark_id)
        _validate_exact_prediction_row(
            task=task,
            input_row=input_by_id[benchmark_id],
            raw_prediction=raw_prediction,
            ability=ability,
        )
        exact_predictions.append(dict(raw_prediction))
        scoring_predictions.append(
            _extract_prediction_for_scoring(
                task=task,
                raw_prediction=raw_prediction,
                ability=ability,
            )
        )

    missing = [benchmark_id for benchmark_id in input_by_id if benchmark_id not in seen_ids]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} benchmark rows.")

    return exact_predictions, scoring_predictions


def _run_predict_command(
    *,
    task: str,
    inputs: list[dict[str, Any]],
    command: str,
    ability: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact_predictions: list[dict[str, Any]] = []
    scoring_predictions: list[dict[str, Any]] = []
    for input_row in inputs:
        process = subprocess.run(
            command,
            input=json.dumps(input_row, ensure_ascii=True),
            text=True,
            encoding="utf-8",
            capture_output=True,
            shell=True,
            check=False,
        )
        if process.returncode != 0:
            stderr = process.stderr.strip()
            raise RuntimeError(
                f"Prediction command failed for {input_row.get('benchmark_id')}: "
                f"{stderr or f'exit code {process.returncode}'}"
            )
        stdout = process.stdout.strip()
        if not stdout:
            raise RuntimeError(f"Prediction command returned empty stdout for {input_row.get('benchmark_id')}.")
        try:
            raw_prediction = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Prediction command returned invalid JSON for {input_row.get('benchmark_id')}: {stdout}"
            ) from exc
        _validate_exact_prediction_row(
            task=task,
            input_row=input_row,
            raw_prediction=raw_prediction,
            ability=ability,
        )
        exact_predictions.append(dict(raw_prediction))
        scoring_predictions.append(
            _extract_prediction_for_scoring(
                task=task,
                raw_prediction=raw_prediction,
                ability=ability,
            )
        )
    return exact_predictions, scoring_predictions


def _run_predict_argv(
    *,
    task: str,
    inputs: list[dict[str, Any]],
    argv: list[str],
    ability: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact_predictions: list[dict[str, Any]] = []
    scoring_predictions: list[dict[str, Any]] = []
    for input_row in inputs:
        process = subprocess.run(
            argv,
            input=json.dumps(input_row, ensure_ascii=True),
            text=True,
            encoding="utf-8",
            capture_output=True,
            shell=False,
            check=False,
        )
        if process.returncode != 0:
            stderr = process.stderr.strip()
            raise RuntimeError(
                f"Prediction program failed for {input_row.get('benchmark_id')}: "
                f"{stderr or f'exit code {process.returncode}'}"
            )
        stdout = process.stdout.strip()
        if not stdout:
            raise RuntimeError(f"Prediction program returned empty stdout for {input_row.get('benchmark_id')}.")
        try:
            raw_prediction = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Prediction program returned invalid JSON for {input_row.get('benchmark_id')}: {stdout}"
            ) from exc
        _validate_exact_prediction_row(
            task=task,
            input_row=input_row,
            raw_prediction=raw_prediction,
            ability=ability,
        )
        exact_predictions.append(dict(raw_prediction))
        scoring_predictions.append(
            _extract_prediction_for_scoring(
                task=task,
                raw_prediction=raw_prediction,
                ability=ability,
            )
        )
    return exact_predictions, scoring_predictions


def _build_oracle_predictions(
    *,
    task: str,
    inputs: list[dict[str, Any]],
    references: list[dict[str, Any]],
    ability: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference_by_id = {row["benchmark_id"]: row for row in references}
    exact_predictions: list[dict[str, Any]] = []
    scoring_predictions: list[dict[str, Any]] = []
    for input_row in inputs:
        reference = reference_by_id.get(input_row["benchmark_id"])
        if reference is None:
            raise ValueError(f"Oracle mode requires reference output for {input_row['benchmark_id']}.")
        oracle_prediction = _build_oracle_prediction_row(task=task, reference=reference)
        _validate_exact_prediction_row(
            task=task,
            input_row=input_row,
            raw_prediction=oracle_prediction,
            ability=ability,
        )
        exact_predictions.append(dict(oracle_prediction))
        scoring_predictions.append(
            _extract_prediction_for_scoring(
                task=task,
                raw_prediction=oracle_prediction,
                ability=ability,
            )
        )
    return exact_predictions, scoring_predictions


def run_public_evaluation(
    *,
    task: str,
    split: str,
    ability: str | None = None,
    distractors: int | None = None,
    output_dir: str | None = None,
    predictions_path: str | None = None,
    predict_command: str | None = None,
    predict_argv: list[str] | None = None,
    oracle: bool = False,
    limit: int | None = None,
) -> Dict[str, Any]:
    if oracle and split == "test":
        raise ValueError("Oracle mode is not available on the public test split.")

    selected_mode_count = sum(bool(value) for value in (predictions_path, predict_command, predict_argv, oracle))
    if selected_mode_count != 1:
        raise ValueError(
            "Choose exactly one evaluation source: --predictions, --predict-program/--predict-arg, "
            "--predict-command, or --oracle."
        )

    track_dir = resolve_public_track_dir(split, task, ability=ability, distractors=distractors)
    track_name = resolve_public_track_name(task, ability=ability, distractors=distractors)
    run_dir = Path(output_dir) if output_dir else Path("runs") / "public" / split / track_name
    ensure_dir(run_dir)

    inputs, references = _load_track_rows(track_dir, limit=limit)
    if predict_argv:
        exact_predictions, scoring_predictions = _run_predict_argv(
            task=task,
            inputs=inputs,
            argv=predict_argv,
            ability=ability,
        )
        prediction_source = "predict_program"
    elif predict_command:
        exact_predictions, scoring_predictions = _run_predict_command(
            task=task,
            inputs=inputs,
            command=predict_command,
            ability=ability,
        )
        prediction_source = "predict_command"
    elif predictions_path:
        exact_predictions, scoring_predictions = _load_prediction_file(
            task=task,
            inputs=inputs,
            path=predictions_path,
            ability=ability,
        )
        prediction_source = "predictions_file"
    else:
        exact_predictions, scoring_predictions = _build_oracle_predictions(
            task=task,
            inputs=inputs,
            references=references,
            ability=ability,
        )
        prediction_source = "oracle"

    write_jsonl(run_dir / "predictions.jsonl", exact_predictions)

    manifest = {
        "status": "completed",
        "task": task,
        "split": split,
        "ability": ability,
        "distractors": distractors,
        "track_dir": str(track_dir),
        "prediction_source": prediction_source,
        "predictions_path": str(run_dir / "predictions.jsonl"),
        "local_scoring_expected": supports_local_scoring(split),
        "num_inputs": len(inputs),
        "num_predictions": len(exact_predictions),
    }

    summary: Dict[str, Any]
    if references:
        reference_by_id = {row["benchmark_id"]: row for row in references}
        score_rows: list[dict[str, Any]] = []
        for prediction in scoring_predictions:
            benchmark_id = prediction["benchmark_id"]
            score = score_prediction_row(
                task=task,
                prediction=prediction,
                reference=reference_by_id[benchmark_id],
            )
            score_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "task": task,
                    **score,
                }
            )
        write_jsonl(run_dir / "scores.jsonl", score_rows)
        summary = summarize_score_rows(task=task, score_rows=score_rows)
        summary["prediction_source"] = prediction_source
        summary["predictions_path"] = str(run_dir / "predictions.jsonl")
        summary["scores_path"] = str(run_dir / "scores.jsonl")
    else:
        summary = {
            "status": "ok",
            "task": task,
            "split": split,
            "ability": ability,
            "distractors": distractors,
            "prediction_source": prediction_source,
            "predictions_path": str(run_dir / "predictions.jsonl"),
            "num_rows": len(exact_predictions),
            "local_scoring_available": False,
            "note": "No public reference_output.jsonl is available for this split. Predictions were written without scoring.",
        }

    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "summary.json", summary)
    return summary
