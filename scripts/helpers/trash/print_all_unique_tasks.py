#!/usr/bin/env python3
"""
List unique (task, run) recordings in a BIDS-like dataset, and how often they occur

LLM Code
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

TaskRun = Tuple[str, int]

# Optional tqdm progress bar with a lightweight fallback
try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    def tqdm(it, total=None, desc=None, unit=None):  # type: ignore
        total = int(total) if total is not None else None
        i = 0
        for x in it:
            i += 1
            if total:
                if i == 1 or i == total or i % 100 == 0:
                    pct = (i / total) * 100
                    msg = f"{desc + ': ' if desc else ''}{i}/{total} ({pct:5.1f}%)"
                    print("\r" + msg, end="", file=sys.stderr, flush=True)
            yield x
        if total:
            print(file=sys.stderr)

EEG_RE = re.compile(r"^sub-(?P<id>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>\d+)_eeg\.set$")
MISC_RUN_RE = re.compile(
    r"^sub-(?P<id>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>\d+)_(?P<kind>Samples|Events)\.txt$"
)
MISC_NORUN_RE = re.compile(
    r"^sub-(?P<id>[^_]+)_task-(?P<task>[^_]+)_(?P<kind>Samples|Events)\.txt$"
)


def find_subject_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    subs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")]
    subs.sort(key=lambda p: p.name)
    return subs


def scan_eeg(subject_dir: Path) -> List[TaskRun]:
    eeg_dir = subject_dir / "eeg"
    if not eeg_dir.is_dir():
        return []
    found: set[TaskRun] = set()
    for f in eeg_dir.glob("*.set"):
        m = EEG_RE.match(f.name)
        if m:
            found.add((m.group("task"), int(m.group("run"))))
    return sorted(found, key=lambda tr: (tr[0], tr[1]))


def scan_misc(subject_dir: Path) -> List[TaskRun]:
    misc_dir = subject_dir / "misc"
    if not misc_dir.is_dir():
        return []
    found: set[TaskRun] = set()
    for f in misc_dir.glob("*.txt"):
        name = f.name
        m = MISC_RUN_RE.match(name)
        if m:
            found.add((m.group("task"), int(m.group("run"))))
            continue
        m = MISC_NORUN_RE.match(name)
        if m:
            found.add((m.group("task"), 0))
    return sorted(found, key=lambda tr: (tr[0], tr[1]))


def counter_rows(c: Counter[TaskRun]) -> List[Tuple[str, int, int]]:
    rows = [(task, run, cnt) for (task, run), cnt in c.items()]
    # Lexicographic by task, then run
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def print_table(title: str, c: Counter[TaskRun]) -> None:
    rows = counter_rows(c)
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("(none found)")
        return

    task_w = max(len("task"), max(len(r[0]) for r in rows))
    run_w = max(len("run"), max(len(str(r[1])) for r in rows))
    cnt_w = max(len("count"), max(len(str(r[2])) for r in rows))

    print(f"{'task':<{task_w}}  {'run':>{run_w}}  {'count':>{cnt_w}}")
    print(f"{'-'*task_w}  {'-'*run_w}  {'-'*cnt_w}")
    for task, run, cnt in rows:
        print(f"{task:<{task_w}}  {run:>{run_w}}  {cnt:>{cnt_w}}")

    print(f"\nUnique tuples: {len(c)}")
    print(f"Total subject-occurrences (sum of counts): {sum(c.values())}")


def write_csv(path: Path, eeg: Counter[TaskRun], misc: Counter[TaskRun], total: Counter[TaskRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = sorted(set(eeg.keys()) | set(misc.keys()) | set(total.keys()), key=lambda k: (k[0], k[1]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "run", "eeg_subject_count", "misc_subject_count", "total_subject_count"])
        for task, run in all_keys:
            w.writerow([task, run, eeg.get((task, run), 0), misc.get((task, run), 0), total.get((task, run), 0)])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize unique (task, run) tuples in eeg/ and misc/ folders.")
    ap.add_argument("dataset_root", type=Path, help="Path to BIDS dataset root (contains sub-* folders).")
    ap.add_argument("--csv", type=Path, default=None, help="Optional output CSV path for combined summary.")
    args = ap.parse_args(argv)

    root: Path = args.dataset_root
    subject_dirs = find_subject_dirs(root)
    n = len(subject_dirs)

    if n == 0:
        print(f"No subject folders found directly under: {root}")
        return 1

    eeg_counts: Counter[TaskRun] = Counter()
    misc_counts: Counter[TaskRun] = Counter()

    for sub in tqdm(subject_dirs, total=n, desc="Scanning subjects", unit="sub"):
        for tr in scan_eeg(sub):
            eeg_counts[tr] += 1
        for tr in scan_misc(sub):
            misc_counts[tr] += 1

    total_counts: Counter[TaskRun] = Counter()
    total_counts.update(eeg_counts)
    total_counts.update(misc_counts)

    print(f"\nScanned {n} subject folders under: {root}")

    print_table("EEG tuples (task, run) by #subjects", eeg_counts)
    print_table("MISC tuples (task, run) by #subjects", misc_counts)
    print_table("COMBINED tuples (EEG + MISC) by #subjects", total_counts)

    if args.csv is not None:
        write_csv(args.csv, eeg_counts, misc_counts, total_counts)
        print(f"\nWrote CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())