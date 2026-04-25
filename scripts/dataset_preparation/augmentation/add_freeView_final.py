#!/usr/bin/env python3
"""
Extend a BIDS dataset by adding a synthetic task via symlinks

python3 add_freeView_final.py /path/to/mergedDataset
python3 add_freeView_final.py --remove /path/to/mergedDataset

To control which tasks to merge and what to call the merged task, change TASKS and FREEVIEW_LABEL. For example, to add allTasks, change FREEVIEW_LABEL to "allTasks" and remove the comments to add in all of the other recording types.

The runs of the created mock tasks will be based on the index in TASKS. For example:
If "ThePresent_run-1" is the third entry in TASKS, the _run-3 of the new task will be ThePresent.
If you change "ThePresent_run-1" to be the first entry in TASKS, then _run-1 will be ThePresent.
This is just to be aware that _run-3 may not be the same task depending on your merged task definition (this problem does not occur with freeView and allTasks, as freeView is just a subset of the first few tasks).



Partly LLM written
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# ----------------------------
# User configuration
# ----------------------------

# This string controls BOTH:
#   1) which files are removed in --remove mode (substring match)
#   2) the new task label inserted into filenames (task-<THIS>)
FREEVIEW_LABEL = "freeView" #"allTasks"

TASKS = [
    "symbolSearch_run-1",
    "FunwithFractals_run-1",
    "ThePresent_run-1",
    "DiaryOfAWimpyKid_run-1",
    "DespicableMe_run-1",
    #"RestingState_run-1",
    #"surroundSupp_run-1",
    #"surroundSupp_run-2",
    #"contrastChangeDetection_run-1",
    #"contrastChangeDetection_run-2",
    #"contrastChangeDetection_run-3",
    #"seqLearning6target_run-1",
    #"seqLearning8target_run-1",
]
TASK_ID = {key: i + 1 for i, key in enumerate(TASKS)}  # 1-based recording id

# ----------------------------
# Patterns (as specified)
# ----------------------------

EEG_SET_RE = re.compile(
    r"^sub-(?P<sid>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>\d+)_eeg\.set$"
)
MISC_ET_RE = re.compile(
    r"^sub-(?P<sid>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>\d+)_(?P<kind>Samples|Events)\.txt$"
)


def iter_subject_dirs(dataset_root: Path):
    for p in dataset_root.iterdir():
        if p.is_dir() and p.name.startswith("sub-"):
            yield p


def collect_eeg_recordings(eeg_dir: Path, subject_id: str) -> set[str]:
    keys: set[str] = set()
    if not eeg_dir.is_dir():
        return keys

    for f in eeg_dir.iterdir():
        if not f.is_file() and not f.is_symlink():
            continue
        m = EEG_SET_RE.match(f.name)
        if not m:
            continue
        if m.group("sid") != subject_id:
            continue
        task = m.group("task")
        run = m.group("run")
        keys.add(f"{task}_run-{run}")
    return keys


def collect_eyetrack_recordings(misc_dir: Path, subject_id: str) -> set[str]:
    if not misc_dir.is_dir():
        return set()

    present: dict[str, set[str]] = {}
    for f in misc_dir.iterdir():
        if not f.is_file() and not f.is_symlink():
            continue
        m = MISC_ET_RE.match(f.name)
        if not m:
            continue
        if m.group("sid") != subject_id:
            continue
        task = m.group("task")
        run = m.group("run")
        kind = m.group("kind")  # Samples or Events
        key = f"{task}_run-{run}"
        present.setdefault(key, set()).add(kind)

    return {k for k, kinds in present.items() if {"Samples", "Events"} <= kinds}


def safe_unlink(path: Path) -> bool:
    """Remove a file or symlink. Returns True if removed, False if not present."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except IsADirectoryError:
        print(f"WARNING: Not removing directory {path}", file=sys.stderr)
        return False


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    """
    Create/replace symlink at link_path pointing to target_path.
    Uses a relative target (from link_path.parent) for portability.
    """
    if not target_path.exists():
        print(f"WARNING: Missing target for symlink: {target_path}", file=sys.stderr)
        return

    if link_path.exists() or link_path.is_symlink():
        safe_unlink(link_path)  # safe: link names contain FREEVIEW_LABEL, so should not collide with real data

    rel_target = Path(
        target_path.name if target_path.parent == link_path.parent
        else target_path.relative_to(link_path.parent)
    )
    link_path.symlink_to(rel_target)


def copy_resolved_file(src: Path, dst: Path) -> None:
    """
    Copy file contents to dst (dst is a real file, not a symlink).
    If src is a symlink, copy the target contents.
    """
    if not src.exists():
        print(f"WARNING: Missing source for copy: {src}", file=sys.stderr)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst, follow_symlinks=True)


def remove_freeview_files(subject_dir: Path) -> int:
    removed = 0
    for subfolder in ("eeg", "misc"):
        d = subject_dir / subfolder
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if FREEVIEW_LABEL in f.name:
                if f.is_dir() and not f.is_symlink():
                    print(f"WARNING: Skipping directory {f}", file=sys.stderr)
                    continue
                if safe_unlink(f):
                    removed += 1
    return removed


def extend_subject(subject_dir: Path) -> dict[str, int]:
    subj_folder_name = subject_dir.name  # e.g., sub-NDARAA504CRN
    subject_id = subj_folder_name[len("sub-"):]  # e.g., NDARAA504CRN

    eeg_dir = subject_dir / "eeg"
    misc_dir = subject_dir / "misc"

    eeg_keys = collect_eeg_recordings(eeg_dir, subject_id)
    et_keys = collect_eyetrack_recordings(misc_dir, subject_id)
    complete = eeg_keys & et_keys

    # order by TASKS mapping id
    complete_sorted = sorted(
        complete,
        key=lambda k: TASK_ID.get(k, 10**9),
    )

    created_links = 0
    created_copies = 0
    skipped_unknown = 0

    for key in complete_sorted:
        rec_id = TASK_ID.get(key)
        if rec_id is None:
            skipped_unknown += 1
            print(
                f"WARNING: {subj_folder_name}: recording '{key}' not in TASKS mapping; skipping",
                file=sys.stderr,
            )
            continue

        task, run = key.rsplit("_run-", 1)
        src_prefix = f"{subj_folder_name}_task-{task}_run-{run}"
        dst_prefix = f"{subj_folder_name}_task-{FREEVIEW_LABEL}_run-{rec_id}"

        # --- misc (eyetracking): 2 symlinks ---
        if misc_dir.is_dir():
            for kind in ("Samples", "Events"):
                src = misc_dir / f"{src_prefix}_{kind}.txt"
                dst = misc_dir / f"{dst_prefix}_{kind}.txt"
                ensure_symlink(dst, src)
                if dst.is_symlink():
                    created_links += 1

        # --- eeg: 3 symlinks + 1 copied channels.tsv ---
        if eeg_dir.is_dir():
            for suffix in ("_eeg.json", "_eeg.set", "_events.tsv"):
                src = eeg_dir / f"{src_prefix}{suffix}"
                dst = eeg_dir / f"{dst_prefix}{suffix}"
                ensure_symlink(dst, src)
                if dst.is_symlink():
                    created_links += 1

            src_ch = eeg_dir / f"{src_prefix}_channels.tsv"
            dst_ch = eeg_dir / f"{dst_prefix}_channels.tsv"
            copy_resolved_file(src_ch, dst_ch)
            if dst_ch.exists() and not dst_ch.is_symlink():
                created_copies += 1

    return {
        "complete_recordings": len(complete),
        "links_created": created_links,
        "copies_created": created_copies,
        "skipped_unknown": skipped_unknown,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extend a BIDS dataset by adding a synthetic task via symlinks (or remove them)."
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help=f"Remove files in eeg/ and misc/ whose name contains '{FREEVIEW_LABEL}'.",
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to dataset root (contains sub-* folders).",
    )
    args = parser.parse_args()

    root: Path = args.dataset_path
    if not root.exists() or not root.is_dir():
        print(f"ERROR: Not a directory: {root}", file=sys.stderr)
        return 2

    subjects = list(iter_subject_dirs(root))
    if not subjects:
        print(f"WARNING: No sub-* folders found in {root}", file=sys.stderr)

    if args.remove:
        total_removed = 0
        for sdir in subjects:
            total_removed += remove_freeview_files(sdir)
        print(f"Removed {total_removed} file(s)/symlink(s) containing '{FREEVIEW_LABEL}'.")
        return 0

    # create mode
    total_complete = 0
    total_links = 0
    total_copies = 0
    total_skipped = 0

    for sdir in subjects:
        stats = extend_subject(sdir)
        total_complete += stats["complete_recordings"]
        total_links += stats["links_created"]
        total_copies += stats["copies_created"]
        total_skipped += stats["skipped_unknown"]

    print(
        "Done.\n"
        f"  Subjects processed: {len(subjects)}\n"
        f"  Complete recordings (sum): {total_complete}\n"
        f"  Symlinks created: {total_links}\n"
        f"  channels.tsv copies created: {total_copies}\n"
        f"  Recordings skipped (not in TASKS): {total_skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())