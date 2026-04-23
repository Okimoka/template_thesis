#!/usr/bin/env python3
"""
Create a reproducible random split of modeled male subjects into two equal halves.

Written by LLM
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from collections import defaultdict
from pathlib import Path


SUBJECT_PATTERN = re.compile(r"^(sub-[A-Za-z0-9]+)")
DEFAULT_GROUP_NAMES = (
    "unsynced_clean_male_random_half_a",
    "unsynced_clean_male_random_half_b",
)
DEFAULT_GROUP_LABELS = (
    "Random Male Half A",
    "Random Male Half B",
)
DEFAULT_SEED = 20260422


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Create two reproducible random halves from modeled male subjects in "
            "UNSYNCED_CLEAN_GROUPS."
        )
    )
    parser.add_argument(
        "--participants-tsv",
        type=Path,
        default=script_dir / "participants.tsv",
        help="TSV with participant_id and sex columns.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=repo_root / "UNSYNCED_CLEAN_GROUPS",
        help="Directory containing the fitted models to split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir,
        help="Directory where the symlink folders and summary TSVs are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used to shuffle the modeled male subjects.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned split without writing symlinks.",
    )
    return parser.parse_args()


def subject_from_filename(filename: str) -> str:
    match = SUBJECT_PATTERN.match(filename)
    if match:
        return match.group(1)
    return filename.split("_task-", 1)[0].split("|", 1)[0]


def load_participants(participants_tsv: Path) -> dict[str, dict[str, str]]:
    participants: dict[str, dict[str, str]] = {}

    with participants_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = {"participant_id", "sex"} - set(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"Missing required column(s) in {participants_tsv}: {missing_text}"
            )

        for row in reader:
            participant_id = (row.get("participant_id") or "").strip()
            if participant_id:
                participants[participant_id] = {
                    key: (value or "").strip() for key, value in row.items()
                }

    return participants


def collect_models(model_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)

    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        grouped[subject_from_filename(path.name)].append(path)

    return dict(grouped)


def normalize_sex(value: str) -> str | None:
    normalized = value.strip().upper()
    if normalized in {"M", "MALE"}:
        return "M"
    if normalized in {"F", "FEMALE"}:
        return "F"
    return None


def reset_target_dir(target_dir: Path) -> None:
    if target_dir.exists() and not target_dir.is_dir():
        raise NotADirectoryError(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    for child in target_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
            continue
        raise RuntimeError(f"Refusing to remove non-file entry from {target_dir}: {child}")


def create_symlinks(entries: list[tuple[str, list[Path]]], target_dir: Path) -> int:
    reset_target_dir(target_dir)
    created = 0

    for _subject, files in entries:
        for source in files:
            link_path = target_dir / source.name
            relative_source = Path(os.path.relpath(source, start=target_dir))
            link_path.symlink_to(relative_source)
            created += 1

    return created


def modeled_male_subjects(
    participants: dict[str, dict[str, str]],
    models_by_subject: dict[str, list[Path]],
) -> tuple[list[tuple[str, list[Path]]], list[str], list[str]]:
    entries: list[tuple[str, list[Path]]] = []
    missing_sex: list[str] = []
    non_male: list[str] = []

    for subject, files in models_by_subject.items():
        row = participants.get(subject)
        if row is None:
            missing_sex.append(subject)
            continue

        sex = normalize_sex(row.get("sex", ""))
        if sex is None:
            missing_sex.append(subject)
            continue
        if sex != "M":
            non_male.append(subject)
            continue

        entries.append((subject, sorted(files)))

    entries.sort(key=lambda item: item[0])
    return entries, sorted(missing_sex), sorted(non_male)


def split_entries_randomly(
    entries: list[tuple[str, list[Path]]],
    seed: int,
) -> list[dict[str, object]]:
    if len(entries) % 2 != 0:
        raise ValueError(
            f"Expected an even number of modeled male subjects, found {len(entries)}."
        )

    shuffled_entries = list(entries)
    random.Random(seed).shuffle(shuffled_entries)
    half_size = len(shuffled_entries) // 2

    groups: list[dict[str, object]] = []
    for index, (group_name, group_label, chunk) in enumerate(
        zip(
            DEFAULT_GROUP_NAMES,
            DEFAULT_GROUP_LABELS,
            (shuffled_entries[:half_size], shuffled_entries[half_size:]),
        ),
        start=1,
    ):
        sorted_chunk = sorted(chunk, key=lambda item: item[0])
        groups.append(
            {
                "group_index": index,
                "group_name": group_name,
                "group_label": f"{group_label} (seed {seed})",
                "entries": sorted_chunk,
            }
        )

    return groups


def write_summary_files(output_root: Path, groups: list[dict[str, object]], seed: int) -> tuple[Path, Path]:
    groups_path = output_root / "unsynced_clean_male_random_half_groups.tsv"
    subjects_path = output_root / "unsynced_clean_male_random_half_subjects.tsv"

    with groups_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_index",
                "metric_name",
                "metric_column",
                "split_kind",
                "group_name",
                "group_label",
                "seed",
                "n_subjects",
                "n_models",
                "first_subject",
                "last_subject",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for group in groups:
            entries = group["entries"]
            writer.writerow(
                {
                    "group_index": group["group_index"],
                    "metric_name": "male_random_half",
                    "metric_column": "sex",
                    "split_kind": "random_half",
                    "group_name": group["group_name"],
                    "group_label": group["group_label"],
                    "seed": seed,
                    "n_subjects": len(entries),
                    "n_models": sum(len(files) for _subject, files in entries),
                    "first_subject": entries[0][0],
                    "last_subject": entries[-1][0],
                }
            )

    with subjects_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_index",
                "metric_name",
                "metric_column",
                "split_kind",
                "group_name",
                "group_label",
                "seed",
                "subject",
                "n_models",
                "model_files",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for group in groups:
            for subject, files in group["entries"]:
                writer.writerow(
                    {
                        "group_index": group["group_index"],
                        "metric_name": "male_random_half",
                        "metric_column": "sex",
                        "split_kind": "random_half",
                        "group_name": group["group_name"],
                        "group_label": group["group_label"],
                        "seed": seed,
                        "subject": subject,
                        "n_models": len(files),
                        "model_files": ";".join(path.name for path in files),
                    }
                )

    return groups_path, subjects_path


def main() -> None:
    args = parse_args()

    if not args.participants_tsv.exists():
        raise FileNotFoundError(args.participants_tsv)
    if not args.model_dir.is_dir():
        raise NotADirectoryError(args.model_dir)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.output_root.is_dir():
        raise NotADirectoryError(args.output_root)

    participants = load_participants(args.participants_tsv)
    models_by_subject = collect_models(args.model_dir)
    male_entries, missing_sex, non_male = modeled_male_subjects(participants, models_by_subject)
    groups = split_entries_randomly(male_entries, args.seed)
    groups_path, subjects_path = write_summary_files(args.output_root, groups, args.seed)

    print(f"Subjects with models: {len(models_by_subject)}")
    print(f"Modeled male subjects: {len(male_entries)}")
    if missing_sex:
        print(f"Subjects without usable sex entry: {len(missing_sex)}")
    print(f"Modeled non-male subjects ignored: {len(non_male)}")
    print(f"Seed: {args.seed}")
    print(f"Wrote group summary: {groups_path}")
    print(f"Wrote subject summary: {subjects_path}")

    for group in groups:
        entries = group["entries"]
        print(
            f"{group['group_name']}: {len(entries)} subjects, "
            f"{sum(len(files) for _subject, files in entries)} models "
            f"({entries[0][0]} .. {entries[-1][0]})"
        )

    if args.dry_run:
        return

    for group in groups:
        target_dir = args.output_root / group["group_name"]
        created = create_symlinks(group["entries"], target_dir)
        print(f"Created {created} symlink(s) in {target_dir}")


if __name__ == "__main__":
    main()
