from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.runner import prepare_evaluation_run, run_public_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public benchmark evaluation.")
    parser.add_argument("--task", required=True, help="One of task1, task2, task3, or task4.")
    parser.add_argument("--split", required=True, choices=["examples", "dev", "validation", "test"])
    parser.add_argument("--ability", help="Task 4 ability: ability1..ability5")
    parser.add_argument("--distractors", type=int, help="Task 3 distractor pool size: 100, 300, 500, 700, or 1000.")
    parser.add_argument("--output-dir", help="Directory for run manifests and outputs.")
    parser.add_argument("--predictions", help="Path to a JSONL predictions file.")
    parser.add_argument(
        "--predict-program",
        help="Executable or interpreter to invoke once per row. Use with --predict-arg for a shell-free adapter call.",
    )
    parser.add_argument(
        "--predict-arg",
        action="append",
        default=[],
        help="Additional argument for --predict-program. Repeat for multiple arguments.",
    )
    parser.add_argument(
        "--predict-command",
        help=(
            "Deprecated shell command form. It receives one input row as JSON on stdin "
            "and returns one prediction row as JSON on stdout."
        ),
    )
    parser.add_argument("--oracle", action="store_true", help="Generate oracle predictions from reference_output.jsonl.")
    parser.add_argument("--limit", type=int, help="Optional row limit for the selected track.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the run manifest only and do not execute evaluation.",
    )
    args = parser.parse_args()

    if args.predict_arg and not args.predict_program:
        parser.error("--predict-arg requires --predict-program.")

    if args.dry_run:
        manifest = prepare_evaluation_run(
            task=args.task,
            split=args.split,
            ability=args.ability,
            distractors=args.distractors,
            output_dir=args.output_dir,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    predict_argv = None
    if args.predict_program:
        predict_argv = [args.predict_program, *args.predict_arg]

    summary = run_public_evaluation(
        task=args.task,
        split=args.split,
        ability=args.ability,
        distractors=args.distractors,
        output_dir=args.output_dir,
        predictions_path=args.predictions,
        predict_command=args.predict_command,
        predict_argv=predict_argv,
        oracle=args.oracle,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
