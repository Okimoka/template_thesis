#!/usr/bin/env python3
"""
Counts file types in the dataset (txt, tsv, idf)

LLM Code
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    default_dataset_root = script_root / "DATASET"
    parser = argparse.ArgumentParser(
        description=(
            "Count how many subjects have each combination of "
            "Eyetracking/txt, Eyetracking/tsv, and Eyetracking/idf."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root,
        help=(
            "Path to the DATASET directory. Defaults to DATASET relative "
            "to this script."
        ),
    )
    return parser.parse_args()


def scan_triplets(et_root: Path) -> tuple[Counter[tuple[bool, bool, bool]], int]:
    triplet_counts: Counter[tuple[bool, bool, bool]] = Counter()
    missing_eyetracking = 0

    with os.scandir(et_root) as entries:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue

            eyetracking_root = Path(entry.path) / "Eyetracking"
            if not eyetracking_root.is_dir():
                missing_eyetracking += 1

            triplet = (
                (eyetracking_root / "txt").is_dir(),
                (eyetracking_root / "tsv").is_dir(),
                (eyetracking_root / "idf").is_dir(),
            )
            triplet_counts[triplet] += 1

    return triplet_counts, missing_eyetracking


def build_typst_table(triplet_counts: Counter[tuple[bool, bool, bool]]) -> str:
    lines = [
        "#figure(",
        "  table(",
        "    columns: 5,",
        "    table.header(",
        "      [has_txt],",
        "      [has_tsv],",
        "      [has_idf = false],",
        "      [has_idf = true],",
        "      [row total],",
        "    ),",
    ]

    for has_txt in (False, True):
        for has_tsv in (False, True):
            count_idf_false = triplet_counts[(has_txt, has_tsv, False)]
            count_idf_true = triplet_counts[(has_txt, has_tsv, True)]
            lines.extend(
                [
                    f"    [{str(has_txt).lower()}],",
                    f"    [{str(has_tsv).lower()}],",
                    f"    [{count_idf_false}],",
                    f"    [{count_idf_true}],",
                    f"    [{count_idf_false + count_idf_true}],",
                ]
            )

    lines.extend(
        [
            "  ),",
            "  caption: [Counts of subjects by eyetracking folder combination.],",
            ")",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    et_root = dataset_root / "ET"

    if not dataset_root.is_dir():
        raise SystemExit(f"DATASET directory not found: {dataset_root}")
    if not et_root.is_dir():
        raise SystemExit(f"ET directory not found: {et_root}")

    triplet_counts, missing_eyetracking = scan_triplets(et_root)
    total_subjects = sum(triplet_counts.values())

    print(f"DATASET root: {dataset_root}")
    print(f"ET root:      {et_root}")
    print(f"Subjects scanned: {total_subjects}")
    print(f"Subjects without Eyetracking folder: {missing_eyetracking}")
    print()
    print("Triplet counts (has_txt, has_tsv, has_idf):")
    for has_txt in (False, True):
        for has_tsv in (False, True):
            for has_idf in (False, True):
                count = triplet_counts[(has_txt, has_tsv, has_idf)]
                if count:
                    print(
                        f"  ({str(has_txt).lower()}, "
                        f"{str(has_tsv).lower()}, "
                        f"{str(has_idf).lower()}): {count}"
                    )

    print()
    print("2x2 view with idf split:")
    print("  has_txt  has_tsv  has_idf=false  has_idf=true  row_total")
    for has_txt in (False, True):
        for has_tsv in (False, True):
            count_idf_false = triplet_counts[(has_txt, has_tsv, False)]
            count_idf_true = triplet_counts[(has_txt, has_tsv, True)]
            row_total = count_idf_false + count_idf_true
            print(
                f"  {str(has_txt).lower():<7}  "
                f"{str(has_tsv).lower():<7}  "
                f"{count_idf_false:<13}  "
                f"{count_idf_true:<12}  "
                f"{row_total}"
            )

    print()
    print("Typst table:")
    print(build_typst_table(triplet_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
