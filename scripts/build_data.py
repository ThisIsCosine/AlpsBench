from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.runner import build_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public benchmark data layout.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing public-layout files.")
    args = parser.parse_args()

    summary = build_data(overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
