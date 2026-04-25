#!/usr/bin/env python3
"""
Print synchronisation delays reported in subject HTML reports.

By default this scans:

    /home/oki/ehlers-work2/mergedDataset/derivatives/sub-*/eeg/*report.html

and prints one tab-separated row per delay found:

    subject_id    delay_samples
"""

from __future__ import annotations

import argparse
import mmap
import re
from pathlib import Path
from typing import Iterator


DEFAULT_DERIVATIVES_ROOT = Path("/home/oki/ehlers-work2/mergedDataset/derivatives")
DELAY_RE = re.compile(
    rb"There was an estimated synchronisation delay of\s+([+-]?\d+)\s+samples"
)


def iter_report_files(derivatives_root: Path) -> Iterator[Path]:
    """Return report HTML files from subject EEG derivative folders."""
    if not derivatives_root.is_dir():
        raise FileNotFoundError(f"Derivatives root does not exist: {derivatives_root}")

    for subject_dir in sorted(derivatives_root.glob("sub-*")):
        eeg_dir = subject_dir / "eeg"
        if eeg_dir.is_dir():
            yield from sorted(eeg_dir.glob("*report.html"))


def extract_delays(report_file: Path) -> list[int]:
    if report_file.stat().st_size == 0:
        return []

    with report_file.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped_file:
            return [int(match.group(1)) for match in DELAY_RE.finditer(mapped_file)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print synchronisation delay sample counts from subject report HTML files."
    )
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=DEFAULT_DERIVATIVES_ROOT,
        help=f"Derivatives folder to scan (default: {DEFAULT_DERIVATIVES_ROOT})",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not print the tab-separated header row.",
    )
    parser.add_argument(
        "--include-report",
        action="store_true",
        help="Also print the report HTML path as a third column.",
    )
    args = parser.parse_args()

    if not args.no_header:
        header = "subject_id\tdelay_samples"
        if args.include_report:
            header += "\treport"
        print(header)

    for report_file in iter_report_files(args.derivatives_root.expanduser()):
        subject_id = report_file.parent.parent.name
        for delay in extract_delays(report_file):
            row = f"{subject_id}\t{delay}"
            if args.include_report:
                row += f"\t{report_file}"
            print(row)


if __name__ == "__main__":
    main()
