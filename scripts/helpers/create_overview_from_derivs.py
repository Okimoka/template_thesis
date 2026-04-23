"""
Create a CSV overview from *_proc-eyelink_metrics.json files.
These are generated when running the modified _05_sync_eyelink.py step

Behavior:
- Only searches: <DERIVS_ROOT>/sub-*/misc/**/_proc-eyelink_metrics.json
- Writes:
  1) OUT_CSV (main overview)
  2) OUT_LEGEND_CSV (legend, WIP)

LLM Code
"""

import json
import re
from pathlib import Path

import pandas as pd

PATTERN = "*_proc-eyelink_metrics.json"
RUN_RE = re.compile(r"_run-(\d+)_proc-eyelink_metrics\.json$")

LEGEND = {
    "subject": "Anonymized identifier for the subject",
    "run": "Run number parsed from the metrics filename",
    "release": "Release number which this subject appeared in",
    "TODO": "TODO",
}

# ----------------------------
# Hardcoded input/output paths
# ----------------------------
DERIVS_ROOT = Path("/data/work/st156392/mergedDataset/derivatives")
OUT_CSV = Path("/data/work/st156392/mergedDataset/derivatives/sync_metrics_overview.csv")
# Legend will be written next to OUT_CSV:
OUT_LEGEND_CSV = OUT_CSV.with_name(OUT_CSV.stem + "_legend.csv")


def extract_run(path: Path):
    match = RUN_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def metric_file_sort_key(path: Path):
    run = extract_run(path)
    return (path.parent.parent.name, run is None, run if run is not None else float("inf"), path.name)


def iter_metric_files(root: Path):
    """
    Search optimization:
    - Only folders beginning with 'sub-' under root
    - Only inside each 'sub-*/misc' folder
    - Only JSONs matching PATTERN
    """
    root = root.resolve()
    if not root.exists():
        return []

    files = []
    for subdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")):
        misc = subdir / "misc"
        if misc.is_dir():
            # Use rglob to be robust if files are nested under misc,
            # but still constrained to misc only.
            files.extend(misc.rglob(PATTERN))
    return sorted(files, key=metric_file_sort_key)


def main():
    root = DERIVS_ROOT.resolve()
    out_path = OUT_CSV.resolve()
    out_legend_path = OUT_LEGEND_CSV.resolve()

    files = iter_metric_files(root)

    rows = []
    columns = []  # first-seen column order across all files
    taskname = None

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)  # assume numbers, strings, or NaN/None

        if isinstance(data, dict):
            data = dict(data)
            data["run"] = extract_run(f)

        if taskname is None and isinstance(data, dict) and "task" in data:
            taskname = data.get("task")

        # preserve per-file key order and append new keys globally in first-seen order
        for k in data.keys():
            if k not in columns:
                columns.append(k)

        rows.append(data)

    df = pd.DataFrame(rows)
    if columns:
        df = df.reindex(columns=columns)

    legend_df = pd.DataFrame(
        {"column": list(LEGEND.keys()), "description": list(LEGEND.values())}
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSVs
    df.to_csv(out_path, index=False)
    legend_df.to_csv(out_legend_path, index=False)

    task_label = taskname if taskname else "N/A"
    print(
        f"Found {len(files)} files under {root}/sub-*/misc\n"
        f"Wrote {len(df)} rows, {len(df.columns)} columns -> {out_path}\n"
        f"Wrote legend ({len(legend_df)} rows) -> {out_legend_path}\n"
        f"First task value seen: {task_label}"
    )


if __name__ == "__main__":
    main()
