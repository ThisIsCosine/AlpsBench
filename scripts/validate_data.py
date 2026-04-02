from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.data import validate_public_data_layout  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public benchmark data layout.")
    parser.add_argument(
        "--split",
        action="append",
        choices=["examples", "dev", "validation", "test"],
        help="Validate only the selected split. Repeatable.",
    )
    args = parser.parse_args()

    splits = tuple(args.split) if args.split else ("examples", "dev", "validation", "test")
    summary = validate_public_data_layout(splits=splits)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
