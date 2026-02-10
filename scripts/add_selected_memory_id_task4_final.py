import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Defaults are relative to repo root (same convention as other scripts/)
DEFAULT_INPUT_DIR = Path("data/wildchat/memories/selected/task4_final")
DEFAULT_OUTPUT_DIR = Path("data/wildchat/memories/selected/task4_final_with_selected_memory_id")


def _norm_label(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    return s.strip("/")


def _infer_target_label_from_path(jsonl_path: Path) -> Optional[str]:
    """
    Given a path like:
      .../task4_final/ability1/Social_Relationships/Colleagues/direct.jsonl
    return:
      Social_Relationships/Colleagues

    Works for ability variants such as:
      ability2_general, ability2_interaction, ability3, ability4, etc.
    """
    parts = list(jsonl_path.parts)
    ability_idx = None
    for i, part in enumerate(parts):
        if str(part).startswith("ability"):
            ability_idx = i
            break
    if ability_idx is None:
        return None
    if len(parts) < ability_idx + 3:
        # Must have: abilityX/<label...>/<file>.jsonl
        return None
    label_parts = parts[ability_idx + 1 : -1]
    if not label_parts:
        return None
    return _norm_label("/".join(str(p) for p in label_parts))


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            yield i, json.loads(raw)


def _safe_confidence(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return -1.0


def _canon_label(s: Any) -> str:
    """
    Canonical label form for fuzzy matching.
    - lowercased
    - '\' -> '/'
    - '-' -> '_'
    - trims surrounding '/'
    """
    s = _norm_label(s).lower()
    s = s.replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _pick_selected_memory_id(record: Dict[str, Any], *, target_label: str) -> Optional[str]:
    mems = record.get("memory_items")
    if mems is None:
        mems = record.get("memories")
    if not isinstance(mems, list):
        mems = []

    target_norm = _norm_label(target_label)
    target_canon = _canon_label(target_label)

    # 1) Strict label match (preferred; matches your "label符合路径" rule)
    candidates: List[Dict[str, Any]] = [
        m for m in mems if isinstance(m, dict) and _norm_label(m.get("label")) == target_norm
    ]

    # 2) Canonical label match (handles '-' vs '_' etc.)
    if not candidates:
        candidates = [
            m for m in mems if isinstance(m, dict) and _canon_label(m.get("label")) == target_canon
        ]

    # 3) Fallback to label_suggestion (useful when label is UNMAPPED)
    if not candidates:
        def _sug_ok(m: Dict[str, Any]) -> bool:
            sug = _canon_label(m.get("label_suggestion"))
            if not sug:
                return False
            return sug == target_canon or sug.startswith(target_canon + "/") or sug.startswith(target_canon + "_")

        candidates = [m for m in mems if isinstance(m, dict) and _sug_ok(m)]

    # 4) Last resort: if there's only one memory item, treat it as the target.
    # This keeps Task2 conflict generation workable even when taxonomy labeling is missing (e.g., label == UNMAPPED).
    if not candidates:
        only = [m for m in mems if isinstance(m, dict)]
        if len(only) == 1 and only[0].get("memory_id"):
            return str(only[0].get("memory_id"))

    # 5) Final fallback: pick the highest-confidence memory item.
    # This guarantees selected_memory_id exists so Task2Generator can always resolve a target.
    if not candidates:
        any_items = [m for m in mems if isinstance(m, dict) and m.get("memory_id")]
        if any_items:
            any_items.sort(
                key=lambda m: (_safe_confidence(m.get("confidence")), str(m.get("memory_id", ""))),
                reverse=True,
            )
            return str(any_items[0].get("memory_id"))

    if not candidates:
        return None

    # Prefer higher confidence; tie-break by memory_id for determinism.
    candidates.sort(
        key=lambda m: (_safe_confidence(m.get("confidence")), str(m.get("memory_id", ""))),
        reverse=True,
    )
    picked = candidates[0].get("memory_id")
    return str(picked) if picked else None


@dataclass
class FileStats:
    rel_path: str
    records_total: int = 0
    selected_set: int = 0
    selected_missing: int = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add selected_memory_id into task4_final selected records.\n"
            "For each jsonl file under task4_final/ability*/<LabelPath>/*.jsonl, "
            "we set record['selected_memory_id'] to the memory_id whose memory_items[].label matches <LabelPath>."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Input dir (default: {DEFAULT_INPUT_DIR.as_posix()})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output dir (default: {DEFAULT_OUTPUT_DIR.as_posix()})",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write back into input-dir (DANGEROUS). If set, --output-dir is ignored.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite record['selected_memory_id'] even if it is already set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only print summary.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_root = (repo_root / Path(args.input_dir)).resolve()
    output_root = input_root if args.in_place else (repo_root / Path(args.output_dir)).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Missing input dir: {input_root}")

    jsonl_files = sorted(input_root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files under: {input_root}")

    all_stats: List[FileStats] = []
    total_records = 0
    total_selected_set = 0
    total_selected_missing = 0
    missing_examples: List[Dict[str, Any]] = []

    report_path = output_root / "_selected_memory_id_report.jsonl"
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_handle = report_path.open("w", encoding="utf-8")
    else:
        report_handle = None

    try:
        for in_path in jsonl_files:
            rel_path = in_path.relative_to(input_root)
            target_label = _infer_target_label_from_path(in_path)
            if not target_label:
                # Still copy file as-is (or skip in dry-run) so output tree matches input.
                if not args.dry_run and not args.in_place:
                    out_path = output_root / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with in_path.open("r", encoding="utf-8") as src, out_path.open(
                        "w", encoding="utf-8"
                    ) as dst:
                        for line in src:
                            dst.write(line)
                continue

            stats = FileStats(rel_path=str(rel_path).replace("\\", "/"))
            out_handle = None
            if not args.dry_run:
                out_path = output_root / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_handle = out_path.open("w", encoding="utf-8")

            try:
                for line_idx, record in _iter_jsonl(in_path):
                    stats.records_total += 1
                    total_records += 1

                    existing = record.get("selected_memory_id")
                    should_set = args.overwrite_existing or not existing
                    if should_set:
                        selected_id = _pick_selected_memory_id(record, target_label=target_label)
                        record["selected_memory_id"] = selected_id
                    else:
                        selected_id = existing

                    if selected_id:
                        stats.selected_set += 1
                        total_selected_set += 1
                    else:
                        stats.selected_missing += 1
                        total_selected_missing += 1
                        if len(missing_examples) < 20:
                            missing_examples.append(
                                {
                                    "file": stats.rel_path,
                                    "line": line_idx,
                                    "target_label": target_label,
                                    "available_labels": sorted(
                                        {
                                            _norm_label(m.get("label"))
                                            for m in (record.get("memory_items") or [])
                                            if isinstance(m, dict)
                                        }
                                    )[:20],
                                }
                            )

                    if report_handle is not None:
                        report_handle.write(
                            json.dumps(
                                {
                                    "file": stats.rel_path,
                                    "line": line_idx,
                                    "target_label": target_label,
                                    "selected_memory_id": selected_id,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    if out_handle is not None:
                        out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            finally:
                if out_handle is not None:
                    out_handle.close()

            all_stats.append(stats)
    finally:
        if report_handle is not None:
            report_handle.close()

    print("Done.")
    print(f"Input:  {input_root}")
    if args.dry_run:
        print("Mode:   dry-run (no files written)")
    else:
        print(f"Output: {output_root}")
        print(f"Report: {report_path}")
    print(f"Files scanned: {len(jsonl_files)}")
    print(f"Records total: {total_records}")
    print(f"selected_memory_id set: {total_selected_set}")
    print(f"selected_memory_id missing: {total_selected_missing}")

    if missing_examples:
        print("Missing examples (first 10):")
        for ex in missing_examples[:10]:
            print(
                f"  - {ex['file']} line={ex['line']} target_label={ex['target_label']} "
                f"(available_labels_sample={ex['available_labels']})"
            )


if __name__ == "__main__":
    main()

