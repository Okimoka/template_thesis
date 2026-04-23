#!/usr/bin/env python3
"""
Create four age-quartile model folders for UNSYNCED_CLEAN_GROUPS.

Each output folder contains symlinks to the source models for one age bin,
ordered from youngest to oldest subjects.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path


SUBJECT_PATTERN = re.compile(r"^(sub-[A-Za-z0-9]+)")
DEFAULT_GROUP_NAMES = (
    "unsynced_clean_age_q1_youngest",
    "unsynced_clean_age_q2",
    "unsynced_clean_age_q3",
    "unsynced_clean_age_q4_oldest",
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Split UNSYNCED_CLEAN_GROUPS into four subject-age quartiles and "
            "create symlinked output folders inside sync_quality."
        )
    )
    parser.add_argument(
        "--participants-tsv",
        type=Path,
        default=script_dir / "participants.tsv",
        help="TSV with participant_id and age columns.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=repo_root / "UNSYNCED_CLEAN_GROUPS",
        help="Directory containing the fitted models to split by age.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir,
        help="Directory where the four symlink folders should be created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned quartiles without writing symlinks.",
    )
    return parser.parse_args()


def load_ages(participants_tsv: Path) -> dict[str, float]:
    ages: dict[str, float] = {}

    with participants_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = {"participant_id", "age"} - set(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing required column(s) in {participants_tsv}: {missing_text}")

        for row_number, row in enumerate(reader, start=2):
            participant_id = (row.get("participant_id") or "").strip()
            age_text = (row.get("age") or "").strip()
            if not participant_id or not age_text:
                continue

            try:
                ages[participant_id] = float(age_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid age value {age_text!r} for {participant_id!r} at line {row_number}"
                ) from exc

    return ages


def subject_from_filename(filename: str) -> str:
    match = SUBJECT_PATTERN.match(filename)
    if match:
        return match.group(1)
    return filename.split("_task-", 1)[0].split("|", 1)[0]


def collect_models(model_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)

    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        grouped[subject_from_filename(path.name)].append(path)

    return dict(grouped)


def split_into_quartiles(
    models_by_subject: dict[str, list[Path]],
    ages: dict[str, float],
) -> tuple[list[dict[str, object]], list[str]]:
    missing_age: list[str] = []
    eligible: list[tuple[str, float, list[Path]]] = []

    for subject, files in models_by_subject.items():
        age = ages.get(subject)
        if age is None:
            missing_age.append(subject)
            continue
        eligible.append((subject, age, sorted(files)))

    if not eligible:
        raise ValueError("No subjects with valid ages were found in the model directory.")

    eligible.sort(key=lambda item: (item[1], item[0]))

    base_size, remainder = divmod(len(eligible), 4)
    quartiles: list[dict[str, object]] = []
    start = 0

    for index, group_name in enumerate(DEFAULT_GROUP_NAMES):
        size = base_size + (1 if index < remainder else 0)
        entries = eligible[start : start + size]
        start += size

        if not entries:
            raise ValueError(f"Quartile {index + 1} would be empty.")

        quartiles.append(
            {
                "group_index": index + 1,
                "group_name": group_name,
                "entries": entries,
                "age_min": entries[0][1],
                "age_max": entries[-1][1],
            }
        )

    return quartiles, sorted(missing_age)


def reset_target_dir(target_dir: Path) -> None:
    if target_dir.exists() and not target_dir.is_dir():
        raise NotADirectoryError(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    for child in target_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
            continue
        raise RuntimeError(f"Refusing to remove non-file entry from {target_dir}: {child}")


def create_symlinks(entries: list[tuple[str, float, list[Path]]], target_dir: Path) -> int:
    reset_target_dir(target_dir)
    created = 0

    for _subject, _age, files in entries:
        for source in files:
            link_path = target_dir / source.name
            relative_source = Path(os.path.relpath(source, start=target_dir))
            link_path.symlink_to(relative_source)
            created += 1

    return created


def write_group_summary(output_root: Path, quartiles: list[dict[str, object]]) -> tuple[Path, Path]:
    groups_path = output_root / "unsynced_clean_age_quartile_groups.tsv"
    subjects_path = output_root / "unsynced_clean_age_quartile_subjects.tsv"

    with groups_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_index",
                "group_name",
                "age_min",
                "age_max",
                "n_subjects",
                "n_models",
                "first_subject",
                "last_subject",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for quartile in quartiles:
            entries = quartile["entries"]
            writer.writerow(
                {
                    "group_index": quartile["group_index"],
                    "group_name": quartile["group_name"],
                    "age_min": f"{quartile['age_min']:.4f}",
                    "age_max": f"{quartile['age_max']:.4f}",
                    "n_subjects": len(entries),
                    "n_models": sum(len(files) for _subject, _age, files in entries),
                    "first_subject": entries[0][0],
                    "last_subject": entries[-1][0],
                }
            )

    with subjects_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_index",
                "group_name",
                "subject",
                "age",
                "n_models",
                "model_files",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for quartile in quartiles:
            for subject, age, files in quartile["entries"]:
                writer.writerow(
                    {
                        "group_index": quartile["group_index"],
                        "group_name": quartile["group_name"],
                        "subject": subject,
                        "age": f"{age:.4f}",
                        "n_models": len(files),
                        "model_files": ";".join(path.name for path in files),
                    }
                )

    return groups_path, subjects_path


def describe_quartile(quartile: dict[str, object]) -> str:
    entries = quartile["entries"]
    return (
        f"{quartile['group_name']}: {len(entries)} subjects, "
        f"{sum(len(files) for _subject, _age, files in entries)} models, "
        f"ages {quartile['age_min']:.4f} .. {quartile['age_max']:.4f} "
        f"({entries[0][0]} .. {entries[-1][0]})"
    )


def main() -> None:
    args = parse_args()

    if not args.participants_tsv.exists():
        raise FileNotFoundError(args.participants_tsv)
    if not args.model_dir.is_dir():
        raise NotADirectoryError(args.model_dir)
    if not args.output_root.exists():
        args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.output_root.is_dir():
        raise NotADirectoryError(args.output_root)

    ages = load_ages(args.participants_tsv)
    models_by_subject = collect_models(args.model_dir)
    quartiles, missing_age = split_into_quartiles(models_by_subject, ages)

    print(f"Subjects with models: {len(models_by_subject)}")
    print(f"Models found: {sum(len(files) for files in models_by_subject.values())}")
    if missing_age:
        print(f"Subjects without age entry: {len(missing_age)}")
    for quartile in quartiles:
        print(describe_quartile(quartile))

    groups_path, subjects_path = write_group_summary(args.output_root, quartiles)
    print(f"Wrote group summary: {groups_path}")
    print(f"Wrote subject summary: {subjects_path}")

    if args.dry_run:
        return

    for quartile in quartiles:
        target_dir = args.output_root / quartile["group_name"]
        created = create_symlinks(quartile["entries"], target_dir)
        print(f"Created {created} symlink(s) in {target_dir}")


if __name__ == "__main__":
    main()
