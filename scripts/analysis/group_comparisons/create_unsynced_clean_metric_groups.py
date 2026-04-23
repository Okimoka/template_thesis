#!/usr/bin/env python3
"""
Create symlinked participant groups for UNSYNCED_CLEAN_GROUPS.

This script mirrors the age-quartile workflow, but creates the requested
metric-based group folders inside sync_quality:

- sex: female vs male
- p_factor: quartiles
- attention: quartiles
- internalizing: quartiles
- externalizing: quartiles
- ehq_total: lower half vs upper half

Written by LLM
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SUBJECT_PATTERN = re.compile(r"^(sub-[A-Za-z0-9]+)")


@dataclass(frozen=True)
class MetricSpec:
    metric_name: str
    column_name: str
    summary_stem: str
    split_kind: str
    group_names: tuple[str, ...]
    base_labels: tuple[str, ...]
    categorical_order: tuple[str, ...] = ()


GROUP_SPECS = (
    MetricSpec(
        metric_name="sex",
        column_name="sex",
        summary_stem="unsynced_clean_sex",
        split_kind="categorical",
        group_names=(
            "unsynced_clean_sex_female",
            "unsynced_clean_sex_male",
        ),
        base_labels=("Female", "Male"),
        categorical_order=("F", "M"),
    ),
    MetricSpec(
        metric_name="p_factor",
        column_name="p_factor",
        summary_stem="unsynced_clean_p_factor_quartile",
        split_kind="quantile",
        group_names=(
            "unsynced_clean_p_factor_q1_lowest",
            "unsynced_clean_p_factor_q2",
            "unsynced_clean_p_factor_q3",
            "unsynced_clean_p_factor_q4_highest",
        ),
        base_labels=("Q1", "Q2", "Q3", "Q4"),
    ),
    MetricSpec(
        metric_name="attention",
        column_name="attention",
        summary_stem="unsynced_clean_attention_quartile",
        split_kind="quantile",
        group_names=(
            "unsynced_clean_attention_q1_lowest",
            "unsynced_clean_attention_q2",
            "unsynced_clean_attention_q3",
            "unsynced_clean_attention_q4_highest",
        ),
        base_labels=("Q1", "Q2", "Q3", "Q4"),
    ),
    MetricSpec(
        metric_name="internalizing",
        column_name="internalizing",
        summary_stem="unsynced_clean_internalizing_quartile",
        split_kind="quantile",
        group_names=(
            "unsynced_clean_internalizing_q1_lowest",
            "unsynced_clean_internalizing_q2",
            "unsynced_clean_internalizing_q3",
            "unsynced_clean_internalizing_q4_highest",
        ),
        base_labels=("Q1", "Q2", "Q3", "Q4"),
    ),
    MetricSpec(
        metric_name="externalizing",
        column_name="externalizing",
        summary_stem="unsynced_clean_externalizing_quartile",
        split_kind="quantile",
        group_names=(
            "unsynced_clean_externalizing_q1_lowest",
            "unsynced_clean_externalizing_q2",
            "unsynced_clean_externalizing_q3",
            "unsynced_clean_externalizing_q4_highest",
        ),
        base_labels=("Q1", "Q2", "Q3", "Q4"),
    ),
    MetricSpec(
        metric_name="ehq_total",
        column_name="ehq_total",
        summary_stem="unsynced_clean_ehq_total_half",
        split_kind="half",
        group_names=(
            "unsynced_clean_ehq_total_lower_half",
            "unsynced_clean_ehq_total_upper_half",
        ),
        base_labels=("Lower Half", "Upper Half"),
    ),
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Create symlinked metric-based model groups for UNSYNCED_CLEAN_GROUPS "
            "inside sync_quality."
        )
    )
    parser.add_argument(
        "--participants-tsv",
        type=Path,
        default=script_dir / "participants.tsv",
        help="TSV with participant-level columns used for grouping.",
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
        "--dry-run",
        action="store_true",
        help="Print the planned groups without writing symlinks.",
    )
    return parser.parse_args()


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


def load_participants(participants_tsv: Path) -> dict[str, dict[str, str]]:
    participants: dict[str, dict[str, str]] = {}

    with participants_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"participant_id"}
        missing = required - set(reader.fieldnames or [])
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


def reset_target_dir(target_dir: Path) -> None:
    if target_dir.exists() and not target_dir.is_dir():
        raise NotADirectoryError(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    for child in target_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
            continue
        raise RuntimeError(f"Refusing to remove non-file entry from {target_dir}: {child}")


def create_symlinks(entries: list[tuple[str, object, list[Path]]], target_dir: Path) -> int:
    reset_target_dir(target_dir)
    created = 0

    for _subject, _metric_value, files in entries:
        for source in files:
            link_path = target_dir / source.name
            relative_source = Path(os.path.relpath(source, start=target_dir))
            link_path.symlink_to(relative_source)
            created += 1

    return created


def normalize_sex(value: str) -> str | None:
    normalized = value.strip().upper()
    if normalized in {"F", "FEMALE"}:
        return "F"
    if normalized in {"M", "MALE"}:
        return "M"
    return None


def format_numeric(value: float) -> str:
    return f"{value:.4f}"


def build_label(base_label: str, value_min: object | None, value_max: object | None) -> str:
    if isinstance(value_min, float) and isinstance(value_max, float):
        return f"{base_label} ({format_numeric(value_min)} to {format_numeric(value_max)})"
    return base_label


def split_ranked_entries(
    ranked_entries: list[tuple[str, float, list[Path]]],
    group_names: tuple[str, ...],
    base_labels: tuple[str, ...],
    split_kind: str,
) -> list[dict[str, object]]:
    if len(group_names) != len(base_labels):
        raise ValueError("group_names and base_labels must have the same length.")

    base_size, remainder = divmod(len(ranked_entries), len(group_names))
    groups: list[dict[str, object]] = []
    start = 0

    for index, (group_name, base_label) in enumerate(zip(group_names, base_labels), start=1):
        size = base_size + (1 if index <= remainder else 0)
        entries = ranked_entries[start : start + size]
        start += size

        if not entries:
            raise ValueError(f"Group {group_name} would be empty.")

        value_min = entries[0][1]
        value_max = entries[-1][1]
        groups.append(
            {
                "group_index": index,
                "group_name": group_name,
                "group_label": build_label(base_label, value_min, value_max),
                "entries": entries,
                "value_min": value_min,
                "value_max": value_max,
                "split_kind": split_kind,
            }
        )

    return groups


def build_categorical_groups(
    spec: MetricSpec,
    participants: dict[str, dict[str, str]],
    models_by_subject: dict[str, list[Path]],
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    grouped_entries: dict[str, list[tuple[str, str, list[Path]]]] = {
        key: [] for key in spec.categorical_order
    }
    missing_value: list[str] = []
    invalid_value: list[str] = []

    for subject, files in models_by_subject.items():
        row = participants.get(subject)
        if row is None:
            missing_value.append(subject)
            continue

        raw_value = row.get(spec.column_name, "")
        normalized_value = normalize_sex(raw_value)
        if normalized_value is None:
            if raw_value:
                invalid_value.append(f"{subject}:{raw_value}")
            else:
                missing_value.append(subject)
            continue

        grouped_entries[normalized_value].append((subject, normalized_value, sorted(files)))

    groups: list[dict[str, object]] = []
    for index, (group_value, group_name, base_label) in enumerate(
        zip(spec.categorical_order, spec.group_names, spec.base_labels),
        start=1,
    ):
        entries = sorted(grouped_entries[group_value], key=lambda item: item[0])
        if not entries:
            raise ValueError(f"No eligible subjects were found for group {group_name}.")
        groups.append(
            {
                "group_index": index,
                "group_name": group_name,
                "group_label": base_label,
                "entries": entries,
                "value_min": group_value,
                "value_max": group_value,
                "split_kind": spec.split_kind,
            }
        )

    return groups, sorted(missing_value), sorted(invalid_value)


def build_numeric_groups(
    spec: MetricSpec,
    participants: dict[str, dict[str, str]],
    models_by_subject: dict[str, list[Path]],
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    missing_value: list[str] = []
    invalid_value: list[str] = []
    ranked_entries: list[tuple[str, float, list[Path]]] = []

    for subject, files in models_by_subject.items():
        row = participants.get(subject)
        if row is None:
            missing_value.append(subject)
            continue

        raw_value = row.get(spec.column_name, "")
        if not raw_value:
            missing_value.append(subject)
            continue

        try:
            numeric_value = float(raw_value)
        except ValueError:
            invalid_value.append(f"{subject}:{raw_value}")
            continue

        ranked_entries.append((subject, numeric_value, sorted(files)))

    if not ranked_entries:
        raise ValueError(f"No eligible subjects with numeric values were found for {spec.metric_name}.")

    ranked_entries.sort(key=lambda item: (item[1], item[0]))
    groups = split_ranked_entries(
        ranked_entries,
        spec.group_names,
        spec.base_labels,
        spec.split_kind,
    )
    return groups, sorted(missing_value), sorted(invalid_value)


def build_groups_for_spec(
    spec: MetricSpec,
    participants: dict[str, dict[str, str]],
    models_by_subject: dict[str, list[Path]],
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    if spec.split_kind == "categorical":
        return build_categorical_groups(spec, participants, models_by_subject)
    if spec.split_kind in {"quantile", "half"}:
        return build_numeric_groups(spec, participants, models_by_subject)
    raise ValueError(f"Unsupported split_kind: {spec.split_kind}")


def summary_paths(output_root: Path, spec: MetricSpec) -> tuple[Path, Path]:
    groups_path = output_root / f"{spec.summary_stem}_groups.tsv"
    subjects_path = output_root / f"{spec.summary_stem}_subjects.tsv"
    return groups_path, subjects_path


def write_summary_files(output_root: Path, spec: MetricSpec, groups: list[dict[str, object]]) -> tuple[Path, Path]:
    groups_path, subjects_path = summary_paths(output_root, spec)

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
                "value_min",
                "value_max",
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
            value_min = group["value_min"]
            value_max = group["value_max"]
            writer.writerow(
                {
                    "group_index": group["group_index"],
                    "metric_name": spec.metric_name,
                    "metric_column": spec.column_name,
                    "split_kind": group["split_kind"],
                    "group_name": group["group_name"],
                    "group_label": group["group_label"],
                    "value_min": format_numeric(value_min) if isinstance(value_min, float) else value_min,
                    "value_max": format_numeric(value_max) if isinstance(value_max, float) else value_max,
                    "n_subjects": len(entries),
                    "n_models": sum(len(files) for _subject, _value, files in entries),
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
                "subject",
                "metric_value",
                "n_models",
                "model_files",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for group in groups:
            for subject, value, files in group["entries"]:
                writer.writerow(
                    {
                        "group_index": group["group_index"],
                        "metric_name": spec.metric_name,
                        "metric_column": spec.column_name,
                        "split_kind": group["split_kind"],
                        "group_name": group["group_name"],
                        "group_label": group["group_label"],
                        "subject": subject,
                        "metric_value": format_numeric(value) if isinstance(value, float) else value,
                        "n_models": len(files),
                        "model_files": ";".join(path.name for path in files),
                    }
                )

    return groups_path, subjects_path


def describe_group(group: dict[str, object]) -> str:
    entries = group["entries"]
    value_min = group["value_min"]
    value_max = group["value_max"]
    if isinstance(value_min, float) and isinstance(value_max, float):
        range_text = f"{format_numeric(value_min)} .. {format_numeric(value_max)}"
    else:
        range_text = f"{value_min}"
    return (
        f"{group['group_name']}: {len(entries)} subjects, "
        f"{sum(len(files) for _subject, _value, files in entries)} models, "
        f"values {range_text} ({entries[0][0]} .. {entries[-1][0]})"
    )


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

    print(f"Subjects with models: {len(models_by_subject)}")
    print(f"Models found: {sum(len(files) for files in models_by_subject.values())}")

    for spec in GROUP_SPECS:
        print()
        print(f"[{spec.metric_name}]")
        groups, missing_value, invalid_value = build_groups_for_spec(
            spec,
            participants,
            models_by_subject,
        )
        groups_path, subjects_path = write_summary_files(args.output_root, spec, groups)

        if missing_value:
            print(f"Subjects without usable {spec.column_name}: {len(missing_value)}")
        if invalid_value:
            print(f"Subjects with invalid {spec.column_name}: {len(invalid_value)}")

        for group in groups:
            print(describe_group(group))

        print(f"Wrote group summary: {groups_path}")
        print(f"Wrote subject summary: {subjects_path}")

        if args.dry_run:
            continue

        for group in groups:
            target_dir = args.output_root / group["group_name"]
            created = create_symlinks(group["entries"], target_dir)
            print(f"Created {created} symlink(s) in {target_dir}")


if __name__ == "__main__":
    main()
