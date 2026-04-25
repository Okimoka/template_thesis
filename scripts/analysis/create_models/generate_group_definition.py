#!/usr/bin/env python3
"""
Generate one Julia group definition from a derivatives folder.

Edit the constants below, then run:

    python3 generate_group_definition.py

The output file contains one assignment, for example:

    sample_group = [
        ("/path/to/recording1.fif", "/path/to/recording2.fif"),
        ("/path/to/recording3.fif",),
    ]
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


DERIVATIVES_ROOT = Path("/path/to/mergedDataset/derivatives")
OUTPUT_FILE = Path(__file__).with_name("sample_group.jl")
GROUP_NAME = "sample_group"

PROCESSINGS = ("clean", "eyelink") # or just ("clean") or ("eyelink")

# [] == all
SUBJECTS = []

TASKS = [
    "symbolSearch_run-1",
    "FunwithFractals_run-1",
    "ThePresent_run-1",
    "DiaryOfAWimpyKid_run-1",
    "DespicableMe_run-1",
    # "RestingState_run-1",
    # "surroundSupp_run-1",
    # "surroundSupp_run-2",
    # "contrastChangeDetection_run-1",
    # "contrastChangeDetection_run-2",
    # "contrastChangeDetection_run-3",
    # "seqLearning6target_run-1",
    # "seqLearning8target_run-1",
]

#maps the previously specified tasks to the freeview runs
DERIVATIVE_TASK_LABEL = "freeView"




VALID_PROCESSINGS = {"clean", "eyelink"}
SUBJECT_DIR_RE = re.compile(r"^sub-[^/]+$")
TASK_KEY_RE = re.compile(r"^(?P<task>.+)_run-(?P<run>\d+)$")


def as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def normalize_processings(processings) -> tuple[str, ...]:
    if isinstance(processings, str):
        processings = (processings,)
    return tuple(str(processing).strip().lower() for processing in processings)


def normalize_subject(subject: str) -> str:
    subject = subject.strip()
    if not subject:
        raise ValueError("SUBJECTS contains an empty subject id.")
    return subject if subject.startswith("sub-") else f"sub-{subject}"


def validate_group_name(name: str) -> None:
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) is None:
        raise ValueError(f"GROUP_NAME is not a valid Julia identifier: {name!r}")


def parse_task_key(task_key: str) -> tuple[str, int]:
    match = TASK_KEY_RE.match(task_key)
    if match is None:
        raise ValueError(
            f"Invalid task key {task_key!r}. Expected format like 'symbolSearch_run-1'."
        )
    return match.group("task"), int(match.group("run"))


def derivative_task_run(task_key: str, task_index: int) -> tuple[str, int]:
    original_task, original_run = parse_task_key(task_key)
    if DERIVATIVE_TASK_LABEL is None:
        return original_task, original_run
    return DERIVATIVE_TASK_LABEL, task_index + 1


def iter_subject_dirs(derivatives_root: Path) -> list[Path]:
    if SUBJECTS:
        requested_subjects = [normalize_subject(subject) for subject in SUBJECTS]
        return [derivatives_root / subject for subject in requested_subjects]

    with os.scandir(derivatives_root) as entries:
        return sorted(
            (
                Path(entry.path)
                for entry in entries
                if entry.is_dir() and SUBJECT_DIR_RE.match(entry.name) is not None
            ),
            key=lambda path: path.name,
        )


def expected_fif_path(
    subject_dir: Path,
    task_key: str,
    task_index: int,
    processing: str,
) -> Path:
    subject = subject_dir.name
    task, run = derivative_task_run(task_key, task_index)
    filename = f"{subject}_task-{task}_run-{run}_proc-{processing}_raw.fif"
    return subject_dir / "eeg" / filename


def build_groups(derivatives_root: Path) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    processings = normalize_processings(PROCESSINGS)

    for subject_dir in iter_subject_dirs(derivatives_root):
        if not subject_dir.is_dir():
            continue

        subject_paths: list[str] = []
        for task_index, task_key in enumerate(TASKS):
            for processing in processings:
                fif_path = expected_fif_path(
                    subject_dir,
                    task_key,
                    task_index,
                    processing,
                )
                if fif_path.is_file() or fif_path.is_symlink():
                    subject_paths.append(str(fif_path))

        if subject_paths:
            groups.append(tuple(subject_paths))

    return groups


def format_group_entry(group: tuple[str, ...]) -> str:
    if len(group) == 1:
        return f"({json.dumps(group[0])},)"
    return "(" + ", ".join(json.dumps(path) for path in group) + ")"


def write_julia_group(output_file: Path, group_name: str, groups: list[tuple[str, ...]]) -> None:
    lines = [f"{group_name} = ["]
    lines.extend(f"    {format_group_entry(group)}," for group in groups)
    lines.extend(["]", ""])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")


def validate_configuration() -> Path:
    validate_group_name(GROUP_NAME)
    derivatives_root = as_path(DERIVATIVES_ROOT)

    if not derivatives_root.is_dir():
        raise SystemExit(
            f"DERIVATIVES_ROOT does not exist or is not a directory: {derivatives_root}"
        )

    processings = normalize_processings(PROCESSINGS)
    unknown_processings = sorted(set(processings) - VALID_PROCESSINGS)
    if unknown_processings:
        raise ValueError(
            "Unsupported PROCESSINGS entries: "
            f"{', '.join(unknown_processings)}. Use only clean and/or eyelink."
        )

    if not processings:
        raise ValueError("PROCESSINGS must contain at least one entry.")

    if not TASKS:
        raise ValueError("TASKS must contain at least one recording key.")

    for task_key in TASKS:
        parse_task_key(task_key)

    return derivatives_root


def main() -> None:
    derivatives_root = validate_configuration()
    groups = build_groups(derivatives_root)
    output_file = as_path(OUTPUT_FILE)
    write_julia_group(output_file, GROUP_NAME, groups)

    recording_count = sum(len(group) for group in groups)
    print(f"Wrote {output_file}")
    print(f"{GROUP_NAME}: {len(groups)} subject tuple(s), {recording_count} recording(s)")


if __name__ == "__main__":
    main()
