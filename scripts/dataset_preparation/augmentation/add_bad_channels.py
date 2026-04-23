# -*- coding: utf-8 -*-
from __future__ import annotations

"""

Given a BIDS dataset, this script searches recordings matching
`sub-<id>_task-<task>_run-*_eeg.set` and does the following:
- concatenate all runs of the same task
- run PyPREP bad channel detection
- Write results in status column of all `*_channels.tsv` files for the task
- Old channels files are preserved in `*_channels.tsv.bak`.

- Script uses resume logic so interrupted runs can be restarted safely.
    - already processed channels files are skipped
    - for incomplete channels files, backup is recovered and recomputed

A lot of the code is just file handling, actual detection happens in
nc = NoisyChannels(raw_concat, random_state=0)

Mostly LLM code
"""

# limit threads to one process to avoid using too much cpu
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
import re
import shutil
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import mne
from pyprep import NoisyChannels

# ---- USER INPUT ----
BIDS_ROOT = Path("/data/work/st156392/mergedDataset/")
SUBJECT_IDS: list[str] | None = None #["NDARAG584XLU","NDARRB942UWU","NDAREV527ZRF"]

TASKS_DEFAULT: list[str] = [
    #"symbolSearch",
    #"FunwithFractals",
    #"ThePresent",
    #"DiaryOfAWimpyKid",
    #"DespicableMe",
    #"RestingState",
    #"surroundSupp",
    #"contrastChangeDetection",
    #"seqLearning6target",
    #"seqLearning8target",
    #"allTasks",
    "freeView",
]

# Optional: band-pass for the bad-channel detection step
APPLY_BANDPASS_FOR_DETECTION = False
L_FREQ = 1.0
H_FREQ = 100.0
# --------------------

_RUN_RE = re.compile(r"_run-(\d+)_eeg\.set$", re.IGNORECASE)
_VALID_STATUS = {"good", "bad"}


def run_number_from_set(filename: str) -> int | None:
    m = _RUN_RE.search(filename)
    return int(m.group(1)) if m else None


def normalize_subject_id(s: str) -> str:
    return s[4:] if s.startswith("sub-") else s


def list_subjects(bids_root: Path) -> list[str]:
    subs = []
    for p in bids_root.iterdir():
        if p.is_dir() and p.name.startswith("sub-"):
            subs.append(p.name.replace("sub-", "", 1))

    def key(s: str):
        return (0, int(s)) if s.isdigit() else (1, s)

    return sorted(subs, key=key)


def discover_runs(bids_root: Path, subject: str, task: str) -> list[tuple[str, Path, Path]]:
    """
    Returns [(run_str, eeg_set_path, channels_tsv_path)] sorted by run number.
    Skips task if /eeg/ missing or no matching .set files for that task.
    """
    eeg_dir = bids_root / f"sub-{subject}" / "eeg"
    if not eeg_dir.is_dir():
        return []

    set_files = list(eeg_dir.glob(f"sub-{subject}_task-{task}_run-*_eeg.set"))
    if not set_files:
        return []

    def sort_key(p: Path):
        r = run_number_from_set(p.name)
        return (0, r) if r is not None else (1, p.name)

    set_files.sort(key=sort_key)

    runs: list[tuple[str, Path, Path]] = []
    for set_path in set_files:
        r = run_number_from_set(set_path.name)
        if r is None:
            continue
        run = str(r)
        ch_tsv = set_path.with_name(set_path.name.replace("_eeg.set", "_channels.tsv"))
        if not ch_tsv.exists():
            raise FileNotFoundError(f"Missing channels.tsv for {set_path.name}: {ch_tsv}")
        runs.append((run, set_path, ch_tsv))

    return runs


def read_channels_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "name" not in df.columns:
        raise RuntimeError(f"Missing required 'name' column in {path}")
    return df


# ---------------- RESUME/ROBUSTNESS HELPERS ----------------

def bak_path_for(ch_tsv: Path) -> Path:
    return ch_tsv.with_suffix(ch_tsv.suffix + ".bak")


def channels_status_complete(ch_tsv: Path) -> bool:
    """
    True if channels.tsv has a 'status' column and every row has status in {'good','bad'}.
    Also requires a non-empty table with a 'name' column.
    """
    if not ch_tsv.exists():
        return False
    try:
        df = pd.read_csv(ch_tsv, sep="\t", dtype=str, keep_default_na=False)
    except Exception:
        return False

    if "name" not in df.columns or "status" not in df.columns:
        return False
    if len(df) == 0:
        return False

    status = df["status"].astype(str).str.strip()
    if (status == "").any():
        return False
    if not set(status.unique()).issubset(_VALID_STATUS):
        return False
    return True


def restore_from_bak_if_needed(ch_tsv: Path) -> bool:
    """
    If a .bak exists AND the current channels.tsv is missing/corrupt/incomplete, restore from .bak.
    Returns True if restored.
    """
    bak = bak_path_for(ch_tsv)
    if not bak.exists():
        return False
    if channels_status_complete(ch_tsv):
        return False

    # restore
    shutil.copyfile(bak, ch_tsv)
    return True


def ensure_backup_exists(ch_tsv: Path) -> None:
    """
    Create .bak only if it doesn't exist already (so reruns don't overwrite the original backup).
    """
    bak = bak_path_for(ch_tsv)
    if not bak.exists():
        shutil.copyfile(ch_tsv, bak)


def atomic_write_tsv(df: pd.DataFrame, path: Path) -> None:
    """
    Write to a temp file then atomically replace the target, to avoid partial/truncated writes.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_csv(tmp, sep="\t", index=False, lineterminator="\n")
        os.replace(tmp, path)  # atomic on POSIX when same filesystem
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


# ---------------- CORE PROCESSING ----------------

def _detect_and_write_for_task(subject: str, bids_root: Path, task: str) -> dict:
    """
    Detect bad channels for one (subject, task) across all its runs, then write status to each run's channels.tsv.

    Resume behavior:
    - If every run has a .bak AND channels.tsv has a fully-populated 'status' column, skip this task.
    - If any run has .bak but channels.tsv is incomplete/corrupt, restore from .bak and re-run detection for the task.
    """
    run_triplets = discover_runs(bids_root, subject, task)
    if not run_triplets:
        return {"task": task, "skipped": True, "reason": "no_runs"}

    # 1) Resume check / repair: restore any incomplete channels.tsv that already have a .bak
    restored_any = False
    all_done = True
    for _run, _set_path, ch_tsv in run_triplets:
        bak = bak_path_for(ch_tsv)

        # Done requires BOTH: backup exists and status column fully populated
        done_this = bak.exists() and channels_status_complete(ch_tsv)
        if done_this:
            continue

        all_done = False

        # If there is a .bak but channels.tsv is incomplete/corrupt, restore it
        if bak.exists():
            if restore_from_bak_if_needed(ch_tsv):
                restored_any = True

    if all_done:
        return {"task": task, "skipped": True, "reason": "already_done"}

    # 2) Run detection (re-run if anything was incomplete or if task never processed)
    mne.set_log_level("WARNING")

    raws = []
    for _run, set_path, _ch_tsv in run_triplets:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        raws.append(raw)

    try:
        raw_concat = mne.concatenate_raws(raws, preload=True)
    except TypeError:
        raw_concat = mne.concatenate_raws(raws)

    if APPLY_BANDPASS_FOR_DETECTION:
        raw_concat.filter(l_freq=L_FREQ, h_freq=H_FREQ)

    nc = NoisyChannels(raw_concat, random_state=0)
    nc.find_all_bads()
    bads = set(nc.get_bads())

    # 3) Write outputs robustly (backup-once + atomic replace), then verify completeness
    for _run, _set_path, ch_tsv in run_triplets:
        ensure_backup_exists(ch_tsv)

        df = read_channels_tsv(ch_tsv)
        df["status"] = ["bad" if name in bads else "good" for name in df["name"]]

        atomic_write_tsv(df, ch_tsv)

        # quick post-write sanity check
        if not channels_status_complete(ch_tsv):
            raise RuntimeError(f"Post-write check failed (incomplete status column): {ch_tsv}")

    return {
        "task": task,
        "skipped": False,
        "n_runs": len(run_triplets),
        "n_bads": len(bads),
        "restored_incomplete_before_rerun": restored_any,
    }


def process_subject(subject: str, bids_root_str: str, tasks: list[str]) -> dict:
    bids_root = Path(bids_root_str)

    print(f"[{datetime.now().isoformat()}] starting {subject}", flush=True)

    task_results: list[dict] = []
    task_failures: list[tuple[str, str]] = []

    for task in tasks:
        try:
            res = _detect_and_write_for_task(subject, bids_root, task)
            task_results.append(res)

            if res.get("skipped"):
                continue

            print(
                f"[{datetime.now().isoformat()}] {subject} task={task} runs={res.get('n_runs')} "
                f"bads={res.get('n_bads')} restored_before={res.get('restored_incomplete_before_rerun')}",
                flush=True,
            )
        except Exception as e:
            task_failures.append((task, repr(e)))

    processed_tasks = [r["task"] for r in task_results if not r.get("skipped")]
    skipped_tasks = [r["task"] for r in task_results if r.get("skipped")]

    print(
        f"[{datetime.now().isoformat()}] done {subject} processed_tasks={len(processed_tasks)} "
        f"skipped_tasks={len(skipped_tasks)} failed_tasks={len(task_failures)}",
        flush=True,
    )

    return {
        "subject": subject,
        "processed_tasks": processed_tasks,
        "skipped_tasks": skipped_tasks,
        "task_results": task_results,
        "task_failures": task_failures,
    }


def choose_subjects(bids_root: Path, subject_ids: list[str] | None) -> list[str]:
    if subject_ids is not None and len(subject_ids) > 0:
        return [normalize_subject_id(s) for s in subject_ids]
    return list_subjects(bids_root)


def parse_tasks_arg(s: str) -> list[str]:
    s = s.strip()
    if s.lower() == "all":
        return TASKS_DEFAULT
    tasks = [t.strip() for t in s.split(",") if t.strip()]
    return tasks if tasks else TASKS_DEFAULT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(BIDS_ROOT), help="Dataset root (contains sub-*)")
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help='Tasks to process: "all" or comma-separated list (e.g. "freeView,RestingState")',
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=64,
        help="Number of parallel worker processes to use",
    )
    args = parser.parse_args()

    bids_root = Path(args.root)
    if not bids_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {bids_root}")

    subjects = choose_subjects(bids_root, SUBJECT_IDS)
    if not subjects:
        raise RuntimeError("No subjects selected (SUBJECT_IDS empty and no sub-* folders found).")

    tasks = parse_tasks_arg(args.tasks)
    if not tasks:
        raise RuntimeError("No tasks selected.")

    n_jobs = max(1, min(args.n_jobs, len(subjects)))
    print(f"Subjects: {len(subjects)} | Tasks: {len(tasks)} | workers: {n_jobs}", flush=True)

    results: list[dict] = []
    failures: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        fut_to_subj = {
            ex.submit(process_subject, subj, str(bids_root), tasks): subj for subj in subjects
        }
        for fut in as_completed(fut_to_subj):
            subj = fut_to_subj[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                failures.append((subj, repr(e)))

    total_processed_pairs = sum(len(r.get("processed_tasks", [])) for r in results)
    total_skipped_pairs = sum(len(r.get("skipped_tasks", [])) for r in results)
    total_task_failures = sum(len(r.get("task_failures", [])) for r in results)

    print(
        f"Finished. subjects_ok={len(results)} subjects_failed={len(failures)} "
        f"processed_task_pairs={total_processed_pairs} skipped_task_pairs={total_skipped_pairs} "
        f"failed_task_pairs={total_task_failures}",
        flush=True,
    )

    if failures:
        for subj, err in failures:
            print(f"FAILED subject {subj}: {err}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
