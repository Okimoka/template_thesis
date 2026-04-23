#!/usr/bin/env python3
"""
This was used for the experiment of checking whether more shared events actually
improved the result of the symbolSearch (freeView run-1) synchronization on average.

Exports a sync overview comparing all-events vs first/last-only fits.

This script scans ``mergedDataset/derivatives/sub-*/eeg`` for synchronized
``*task-freeView_run-1_proc-eyelink_raw.fif`` files, opens the corresponding
``task-freeView_report.h5`` report, and writes a CSV summarizing how the shared
event regression would look if only the first and last shared event pair had
been used.

LLM Code
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import numpy as np


DEFAULT_DATASET_ROOT = Path("/data/work/st156392/mergedDataset")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name(
    "freeview_run1_first_last_sync_overview.csv"
)
DEFAULT_EEG_REGEX = r"(?:trialResponse|newPage)"
DEFAULT_ET_REGEX = r"# Message: (?:14|20)"
FIF_GLOB = "*task-freeView_run-1_proc-eyelink_raw.fif"
REPORT_GLOB = "*task-freeView_report.h5"
METRICS_GLOB = "*task-freeView_run-1_proc-eyelink_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan freeView run-1 proc-eyelink FIF files and export a CSV "
            "comparing the shared-event fit from all events versus only the "
            "first and last event pair."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Root mergedDataset directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--eeg-regex",
        default=DEFAULT_EEG_REGEX,
        help=(
            "Regex used to select EEG-side shared-event annotations from the "
            "synchronized FIF (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--et-regex",
        default=DEFAULT_ET_REGEX,
        help=(
            "Regex used to select ET-side shared-event annotations before the "
            'added "ET_" prefix (default: %(default)s)'
        ),
    )
    parser.add_argument(
        "--skip-report-validation",
        action="store_true",
        help=(
            "Do not open report.h5 files. By default the script opens each "
            "report and checks whether it contains the run-1 Eyelink figure."
        ),
    )
    return parser.parse_args()


def first_match(paths: list[Path]) -> Path | None:
    return sorted(paths)[0] if paths else None


def compact_counts(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def maybe_int(value: Any) -> int | None:
    number = maybe_float(value)
    if number is None:
        return None
    return int(round(number))


def validate_report(report_path: Path | None, skip: bool) -> dict[str, Any]:
    result = {
        "report_path": str(report_path) if report_path is not None else "",
        "report_exists": bool(report_path and report_path.exists()),
        "report_opened": False,
        "report_has_run1_sync_figure": False,
        "report_error": "",
    }
    if skip or report_path is None or not report_path.exists():
        return result

    try:
        report = mne.open_report(report_path, verbose="ERROR")
        result["report_opened"] = True
        result["report_has_run1_sync_figure"] = any(
            item.name == "Eyelink data (run 1)" and item.section == "Synchronize Eyelink"
            for item in report._content
        )
    except Exception as exc:  # pragma: no cover - defensive branch
        result["report_error"] = f"{type(exc).__name__}: {exc}"
    return result


def load_metrics(metrics_path: Path | None) -> dict[str, Any]:
    if metrics_path is None or not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_shared_event_onsets(
    raw: mne.io.BaseRaw,
    eeg_regex: str,
    et_regex: str,
) -> tuple[np.ndarray, np.ndarray, Counter[str], Counter[str]]:
    eeg_pattern = re.compile(eeg_regex)
    et_pattern_prefixed = re.compile(f"ET_(?:{et_regex})")

    eeg_onsets: list[float] = []
    et_onsets: list[float] = []
    eeg_counts: Counter[str] = Counter()
    et_counts: Counter[str] = Counter()

    for annot in raw.annotations:
        description = str(annot["description"])
        onset = float(annot["onset"])
        if not description.startswith("ET_") and eeg_pattern.fullmatch(description):
            eeg_onsets.append(onset)
            eeg_counts[description] += 1
        elif et_pattern_prefixed.fullmatch(description):
            et_onsets.append(onset)
            et_counts[description] += 1

    eeg_onsets_arr = np.sort(np.asarray(eeg_onsets, dtype=float))
    et_onsets_arr = np.sort(np.asarray(et_onsets, dtype=float))

    if eeg_onsets_arr.shape != et_onsets_arr.shape:
        raise ValueError(
            "Shared EEG and ET event counts do not match "
            f"({len(eeg_onsets_arr)} vs {len(et_onsets_arr)})."
        )
    if len(eeg_onsets_arr) < 2:
        raise ValueError(f"Need at least 2 shared events, got {len(eeg_onsets_arr)}.")

    return eeg_onsets_arr, et_onsets_arr, eeg_counts, et_counts


def compute_counterfactual_metrics(
    raw_onsets: np.ndarray,
    et_onsets: np.ndarray,
    sfreq: float,
) -> dict[str, Any]:
    coef_all = np.polyfit(raw_onsets, et_onsets, 1)
    pred_all = np.poly1d(coef_all)(raw_onsets)
    res_all_ms = (et_onsets - pred_all) * 1000.0

    span_raw = float(raw_onsets[-1] - raw_onsets[0])
    if span_raw == 0.0:
        raise ValueError("First and last EEG shared-event onsets are identical.")

    first_last_slope = float((et_onsets[-1] - et_onsets[0]) / span_raw)
    first_last_intercept = float(et_onsets[0] - first_last_slope * raw_onsets[0])
    pred_first_last = first_last_slope * raw_onsets + first_last_intercept
    res_first_last_ms = (et_onsets - pred_first_last) * 1000.0
    line_diff_ms = (pred_first_last - pred_all) * 1000.0

    one_sample_ms = 1000.0 / sfreq
    four_samples_ms = 4.0 * one_sample_ms

    return {
        "shared_events_from_annotations": int(len(raw_onsets)),
        "first_raw_onset_s": float(raw_onsets[0]),
        "last_raw_onset_s": float(raw_onsets[-1]),
        "event_span_s": span_raw,
        "all_fit_slope": float(coef_all[0]),
        "all_fit_intercept": float(coef_all[1]),
        "all_mean_abs_residual_ms": float(np.mean(np.abs(res_all_ms))),
        "all_median_abs_residual_ms": float(np.median(np.abs(res_all_ms))),
        "all_max_abs_residual_ms": float(np.max(np.abs(res_all_ms))),
        "all_rmse_residual_ms": float(np.sqrt(np.mean(res_all_ms**2))),
        "all_within_1_sample": int(np.sum(np.abs(res_all_ms) <= one_sample_ms)),
        "all_within_4_samples": int(np.sum(np.abs(res_all_ms) <= four_samples_ms)),
        "first_last_fit_slope": first_last_slope,
        "first_last_fit_intercept": first_last_intercept,
        "first_last_mean_abs_residual_ms": float(np.mean(np.abs(res_first_last_ms))),
        "first_last_median_abs_residual_ms": float(np.median(np.abs(res_first_last_ms))),
        "first_last_max_abs_residual_ms": float(np.max(np.abs(res_first_last_ms))),
        "first_last_rmse_residual_ms": float(np.sqrt(np.mean(res_first_last_ms**2))),
        "first_last_within_1_sample": int(
            np.sum(np.abs(res_first_last_ms) <= one_sample_ms)
        ),
        "first_last_within_4_samples": int(
            np.sum(np.abs(res_first_last_ms) <= four_samples_ms)
        ),
        "delta_mean_abs_residual_ms": float(
            np.mean(np.abs(res_first_last_ms)) - np.mean(np.abs(res_all_ms))
        ),
        "delta_median_abs_residual_ms": float(
            np.median(np.abs(res_first_last_ms)) - np.median(np.abs(res_all_ms))
        ),
        "delta_max_abs_residual_ms": float(
            np.max(np.abs(res_first_last_ms)) - np.max(np.abs(res_all_ms))
        ),
        "delta_rmse_residual_ms": float(
            np.sqrt(np.mean(res_first_last_ms**2)) - np.sqrt(np.mean(res_all_ms**2))
        ),
        "mean_abs_line_diff_ms": float(np.mean(np.abs(line_diff_ms))),
        "max_abs_line_diff_ms": float(np.max(np.abs(line_diff_ms))),
    }


def build_row(
    subject_dir: Path,
    fif_path: Path,
    report_path: Path | None,
    metrics_path: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "subject": subject_dir.name,
        "fif_path": str(fif_path),
        "report_path": str(report_path) if report_path is not None else "",
        "metrics_path": str(metrics_path) if metrics_path is not None else "",
        "status": "ok",
        "error": "",
        "eeg_regex": args.eeg_regex,
        "et_regex": args.et_regex,
    }

    row.update(validate_report(report_path, skip=args.skip_report_validation))
    metrics_json = load_metrics(metrics_path)
    row["metrics_json_present"] = bool(metrics_json)

    try:
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
        row["sfreq_hz"] = float(raw.info["sfreq"])

        raw_onsets, et_onsets, eeg_counts, et_counts = extract_shared_event_onsets(
            raw=raw,
            eeg_regex=args.eeg_regex,
            et_regex=args.et_regex,
        )
        row["eeg_sync_label_counts"] = compact_counts(eeg_counts)
        row["et_sync_label_counts"] = compact_counts(et_counts)
        row.update(
            compute_counterfactual_metrics(
                raw_onsets=raw_onsets,
                et_onsets=et_onsets,
                sfreq=float(raw.info["sfreq"]),
            )
        )

        metrics_shared_events = maybe_int(metrics_json.get("shared_events"))
        row["shared_events_from_metrics_json"] = metrics_shared_events
        row["shared_event_count_matches_metrics_json"] = (
            metrics_shared_events == row["shared_events_from_annotations"]
            if metrics_shared_events is not None
            else ""
        )
        row["metrics_mean_abs_sync_error_ms"] = maybe_float(
            metrics_json.get("mean_abs_sync_error_ms")
        )
        row["metrics_median_abs_sync_error_ms"] = maybe_float(
            metrics_json.get("median_abs_sync_error_ms")
        )
        row["metrics_regression_slope"] = maybe_float(metrics_json.get("regression_slope"))
        row["metrics_within_1_sample"] = maybe_int(metrics_json.get("within_1_sample"))
        row["metrics_within_4_samples"] = maybe_int(metrics_json.get("within_4_samples"))
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    derivatives_dir = args.dataset_root / "derivatives"
    if not derivatives_dir.is_dir():
        raise FileNotFoundError(f"Derivatives directory not found: {derivatives_dir}")

    subject_dirs = sorted(
        path for path in derivatives_dir.iterdir() if path.is_dir() and path.name.startswith("sub-")
    )

    rows: list[dict[str, Any]] = []
    for subject_dir in subject_dirs:
        eeg_dir = subject_dir / "eeg"
        if not eeg_dir.is_dir():
            continue

        fif_path = first_match(list(eeg_dir.glob(FIF_GLOB)))
        if fif_path is None:
            continue

        report_path = first_match(list(eeg_dir.glob(REPORT_GLOB)))
        metrics_dir = subject_dir / "misc"
        metrics_path = first_match(list(metrics_dir.glob(METRICS_GLOB))) if metrics_dir.is_dir() else None

        rows.append(build_row(subject_dir, fif_path, report_path, metrics_path, args))

    rows.sort(key=lambda row: row["subject"])
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "subject",
        "status",
        "error",
        "fif_path",
        "report_path",
        "metrics_path",
        "report_exists",
        "report_opened",
        "report_has_run1_sync_figure",
        "report_error",
        "metrics_json_present",
        "eeg_regex",
        "et_regex",
        "eeg_sync_label_counts",
        "et_sync_label_counts",
        "sfreq_hz",
        "shared_events_from_annotations",
        "shared_events_from_metrics_json",
        "shared_event_count_matches_metrics_json",
        "first_raw_onset_s",
        "last_raw_onset_s",
        "event_span_s",
        "all_fit_slope",
        "all_fit_intercept",
        "all_mean_abs_residual_ms",
        "all_median_abs_residual_ms",
        "all_max_abs_residual_ms",
        "all_rmse_residual_ms",
        "all_within_1_sample",
        "all_within_4_samples",
        "first_last_fit_slope",
        "first_last_fit_intercept",
        "first_last_mean_abs_residual_ms",
        "first_last_median_abs_residual_ms",
        "first_last_max_abs_residual_ms",
        "first_last_rmse_residual_ms",
        "first_last_within_1_sample",
        "first_last_within_4_samples",
        "delta_mean_abs_residual_ms",
        "delta_median_abs_residual_ms",
        "delta_max_abs_residual_ms",
        "delta_rmse_residual_ms",
        "mean_abs_line_diff_ms",
        "max_abs_line_diff_ms",
        "metrics_mean_abs_sync_error_ms",
        "metrics_median_abs_sync_error_ms",
        "metrics_regression_slope",
        "metrics_within_1_sample",
        "metrics_within_4_samples",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    mne.set_log_level("ERROR")

    if not args.dataset_root.is_dir():
        print(
            f"Dataset root does not exist or is not a directory: {args.dataset_root}",
            file=sys.stderr,
        )
        return 1

    rows = collect_rows(args)
    write_csv(rows, args.output)

    ok_count = sum(row["status"] == "ok" for row in rows)
    error_count = len(rows) - ok_count

    print(f"Rows written: {len(rows)}")
    print(f"Successful rows: {ok_count}")
    print(f"Error rows: {error_count}")
    print(f"CSV written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
