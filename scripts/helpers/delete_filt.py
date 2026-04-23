#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
"""
Given a folder path as argument, this removes all filt_raw fif files.
The reason one might want to do this is to reduce the size of the processed dataset.
The intermediate step filt_raw does not contribute much to ET analyses, as it does not
yet contain the aligned ET data. So out of all outputted fif files, it is the safest to delete

LLM Code
"""
FIND_REGEX = r".*/sub-[A-Za-z0-9]+_task-allTasks_run-[0-9]+_proc-filt_raw\.fif$"

def delete_matching_files(root: Path, dry_run: bool = False) -> int:
    pattern = re.compile(FIND_REGEX)
    deleted_count = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        # Normalize separators so the regex works on Windows too.
        path_str = path.resolve().as_posix()

        if pattern.fullmatch(path_str):
            if dry_run:
                print(f"[DRY RUN] Would delete: {path}")
            else:
                try:
                    path.unlink()
                    print(f"Deleted: {path}")
                    deleted_count += 1
                except OSError as e:
                    print(f"Failed to delete {path}: {e}", file=sys.stderr)

    return deleted_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively find and delete files matching a regex."
    )
    parser.add_argument("folder", help="Root folder to search")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting anything",
    )
    args = parser.parse_args()

    root = Path(args.folder)

    if not root.exists():
        print(f"Error: folder does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    deleted = delete_matching_files(root, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Done. Deleted {deleted} file(s).")


if __name__ == "__main__":
    main()
