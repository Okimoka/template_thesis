#!/usr/bin/env python3
"""
Compare minimally harmonized raw EEG recordings against preprocessing stages.

This script focuses on the five free-viewing task mappings introduced in
`add_freeView_final.py` and emits three CSV files:

1. recording_metrics.csv
   One row per subject/task recording with stage-specific metrics.
2. task_summary.csv
   One row per task with mean/std/median summaries for each metric/stage.
3. subject_timings.csv
   One row per subject with wall-clock evaluation time.

Method choices for comparability:
- Only recordings with both raw and clean files are included.
- Metrics are computed on the overlap with the clean recording's sample range.
- EEG channels only.
- One common EEG channel set is used across available stages in a recording.
- Bad channels are excluded using the union of stage bad-channel lists.
- A common average reference is applied at metric time.
- Native raw `.set` data are minimally harmonized for metrics by filtering to
  the requested passband when needed.

LLM Code
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.filter import filter_data
from mne.time_frequency import psd_array_welch

TASK_MAPPING: dict[str, str] = {
    "symbolSearch_run-1": "freeView_run-1",
    "FunwithFractals_run-1": "freeView_run-2",
    "ThePresent_run-1": "freeView_run-3",
    "DiaryOfAWimpyKid_run-1": "freeView_run-4",
    "DespicableMe_run-1": "freeView_run-5",
}
TASK_ORDER = {task_key: idx for idx, task_key in enumerate(TASK_MAPPING)}

STAGES: tuple[str, ...] = ("raw", "filt", "eyelink", "clean")
BETTER_HIGHER_METRICS = {
    "spectral_snr_db",
}
BETTER_LOWER_METRICS = {
    "prominence_20hz_db",
    "line_noise_max_db",
    "artifact_window_fraction",
    "window_max_p2p_95th_uv",
    "median_rms_uv",
    "eog_abs_corr_median",
    "eog_abs_corr_95th",
}
DELTA_SKIP_SUFFIXES = {
    "available",
    "original_duration_sec",
    "metric_duration_sec",
    "eog_channel_count",
    "retained_vs_raw_pct",
}
SUMMARY_SKIP_COLUMNS = {
    "subject",
    "task",
    "raw_task",
    "freeview_task",
    "raw_path",
    "filt_path",
    "eyelink_path",
    "clean_path",
}
DEFAULT_OUTPUT_DIR = Path.cwd() / "preprocessing_metric_outputs"
DEFAULT_DATASET_ROOT = Path("/home/oki/ehlers-work2/mergedDataset")
DEFAULT_RAW_ROOT = Path("/home/oki/ehlers-work2/DATASET/EEG/ds005512")


@dataclass(frozen=True)
class RecordingPaths:
    subject: str
    raw_task: str
    freeview_task: str
    raw_path: Path
    filt_path: Path | None
    eyelink_path: Path | None
    clean_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Merged dataset root containing raw BIDS-like data and derivatives.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Fallback root for readable raw EEGLAB .set files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV outputs will be written.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Only evaluate the first N eligible subjects (alphabetical order).",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Optional comma-separated subject IDs (without the sub- prefix) to evaluate.",
    )
    parser.add_argument(
        "--max-recordings-per-subject",
        type=int,
        default=None,
        help="Only evaluate the first N task recordings per selected subject.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated raw task keys to evaluate, e.g. symbolSearch_run-1.",
    )
    parser.add_argument(
        "--require-all-tasks",
        action="store_true",
        help="Only evaluate subjects that have all five raw+clean task pairs.",
    )
    parser.add_argument(
        "--stages",
        default="raw,filt,eyelink,clean",
        help="Comma-separated stage subset to evaluate. Valid stages: raw,filt,eyelink,clean.",
    )
    parser.add_argument(
        "--artifact-threshold-uv",
        type=float,
        default=200.0,
        help="Peak-to-peak threshold for the artifact-window fraction metric.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="Window size used for artifact-window metrics.",
    )
    parser.add_argument(
        "--l-freq",
        type=float,
        default=0.5,
        help="Lower cutoff for harmonizing native raw recordings.",
    )
    parser.add_argument(
        "--h-freq",
        type=float,
        default=100.0,
        help="Upper cutoff for harmonizing native raw recordings.",
    )
    return parser.parse_args()


def split_task_key(task_key: str) -> tuple[str, str]:
    task, run = task_key.rsplit("_run-", 1)
    return task, run


def candidate_raw_paths(
    dataset_root: Path,
    raw_root: Path,
    subject: str,
    raw_task: str,
    raw_run: str,
) -> list[Path]:
    subject_dir = f"sub-{subject}"
    return [
        dataset_root
        / subject_dir
        / "eeg"
        / f"{subject_dir}_task-{raw_task}_run-{raw_run}_eeg.set",
        raw_root
        / subject_dir
        / "eeg"
        / f"{subject_dir}_task-{raw_task}_eeg.set",
        raw_root
        / subject_dir
        / "eeg"
        / f"{subject_dir}_task-{raw_task}_run-{raw_run}_eeg.set",
    ]


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def path_lexists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def discover_subject_recordings(
    dataset_root: Path,
    raw_root: Path,
    require_all_tasks: bool,
) -> list[RecordingPaths]:
    derivatives_root = dataset_root / "derivatives"
    recordings_by_subject: dict[str, list[RecordingPaths]] = {}

    for subject_dir in sorted(p for p in derivatives_root.iterdir() if p.is_dir() and p.name.startswith("sub-")):
        subject = subject_dir.name.removeprefix("sub-")
        eeg_dir = subject_dir / "eeg"
        subject_rows: list[RecordingPaths] = []

        for raw_task_key, freeview_task_key in TASK_MAPPING.items():
            raw_task, raw_run = split_task_key(raw_task_key)
            freeview_task, freeview_run = split_task_key(freeview_task_key)

            raw_path = (
                dataset_root
                / f"sub-{subject}"
                / "eeg"
                / f"sub-{subject}_task-{raw_task}_run-{raw_run}_eeg.set"
            )
            clean_path = (
                eeg_dir / f"sub-{subject}_task-{freeview_task}_run-{freeview_run}_proc-clean_raw.fif"
            )
            if not path_lexists(raw_path) or not clean_path.exists():
                continue

            filt_path = (
                eeg_dir / f"sub-{subject}_task-{freeview_task}_run-{freeview_run}_proc-filt_raw.fif"
            )
            eyelink_path = (
                eeg_dir / f"sub-{subject}_task-{freeview_task}_run-{freeview_run}_proc-eyelink_raw.fif"
            )

            subject_rows.append(
                RecordingPaths(
                    subject=subject,
                    raw_task=raw_task_key,
                    freeview_task=freeview_task_key,
                    raw_path=raw_path,
                    filt_path=filt_path if filt_path.exists() else None,
                    eyelink_path=eyelink_path if eyelink_path.exists() else None,
                    clean_path=clean_path,
                )
            )

        if subject_rows and (not require_all_tasks or len(subject_rows) == len(TASK_MAPPING)):
            recordings_by_subject[subject] = sorted(
                subject_rows,
                key=lambda row: TASK_ORDER.get(row.raw_task, len(TASK_ORDER)),
            )

    return [row for subject in sorted(recordings_by_subject) for row in recordings_by_subject[subject]]


def parse_stage_list(stage_string: str) -> tuple[str, ...]:
    requested = tuple(stage.strip() for stage in stage_string.split(",") if stage.strip())
    if not requested:
        raise SystemExit("At least one stage must be requested via --stages.")

    invalid = [stage for stage in requested if stage not in STAGES]
    if invalid:
        valid = ", ".join(STAGES)
        raise SystemExit(f"Invalid stage(s): {', '.join(invalid)}. Valid stages: {valid}")

    if "clean" not in requested:
        requested = requested + ("clean",)

    deduped: list[str] = []
    for stage in requested:
        if stage not in deduped:
            deduped.append(stage)
    return tuple(deduped)


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def open_raw(path: Path):
    if path.suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
    if path.suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    raise ValueError(f"Unsupported file type: {path}")


def dataset_anchor(path: Path, anchor_name: str) -> Path | None:
    current = path
    while True:
        if current.name == anchor_name:
            return current
        if current.parent == current:
            return None
        current = current.parent


def symlink_target_candidates(raw_link: Path, raw_root: Path) -> list[Path]:
    if not raw_link.is_symlink():
        return []

    try:
        target = os.readlink(raw_link)
    except OSError:
        return []

    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = (raw_link.parent / target_path).resolve(strict=False)

    candidates: list[Path] = [target_path]
    local_dataset_root = dataset_anchor(raw_root, "DATASET")
    if local_dataset_root is not None and "DATASET" in target_path.parts:
        dataset_idx = target_path.parts.index("DATASET")
        remapped = local_dataset_root.joinpath(*target_path.parts[dataset_idx + 1 :])
        candidates.append(remapped)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def resolve_raw_open_path(recording: RecordingPaths, dataset_root: Path, raw_root: Path) -> Path:
    raw_task, raw_run = split_task_key(recording.raw_task)
    candidates = symlink_target_candidates(recording.raw_path, raw_root) + candidate_raw_paths(
        dataset_root=dataset_root,
        raw_root=raw_root,
        subject=recording.subject,
        raw_task=raw_task,
        raw_run=raw_run,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve a readable raw path for sub-{recording.subject} {recording.raw_task}"
    )


def stage_duration_seconds(raw) -> float:
    return raw.n_times / raw.info["sfreq"]


def eeg_channel_names(raw) -> list[str]:
    return [
        ch_name
        for idx, ch_name in enumerate(raw.ch_names)
        if mne.channel_type(raw.info, idx) == "eeg"
    ]


def compute_common_eeg_channels(raws: dict[str, object]) -> tuple[list[str], set[str]]:
    eeg_sets = [set(eeg_channel_names(raw)) for raw in raws.values()]
    common = set.intersection(*eeg_sets)

    excluded_bads: set[str] = set()
    for raw in raws.values():
        excluded_bads.update(raw.info["bads"])

    common -= excluded_bads
    common = {ch for ch in common if ch != "Cz"}
    return sorted(common), excluded_bads


def overlap_bounds_against_clean(stage_raw, clean_raw) -> tuple[int, int]:
    start = max(int(stage_raw.first_samp), int(clean_raw.first_samp))
    stop = min(int(stage_raw.last_samp), int(clean_raw.last_samp))
    if stop < start:
        raise ValueError("No sample overlap between stage and clean recordings")
    return start, stop


def extract_overlap_data(stage_raw, clean_raw, channel_names: list[str]) -> tuple[np.ndarray, float]:
    if not channel_names:
        raise ValueError("No common EEG channels available after exclusions")

    start_sample, stop_sample = overlap_bounds_against_clean(stage_raw, clean_raw)
    start_idx = start_sample - int(stage_raw.first_samp)
    stop_idx = stop_sample - int(stage_raw.first_samp) + 1
    sfreq = float(stage_raw.info["sfreq"])
    picks = [stage_raw.ch_names.index(ch) for ch in channel_names]
    data = stage_raw.get_data(
        picks=picks,
        start=start_idx,
        stop=stop_idx,
        reject_by_annotation=None,
    )
    return data, sfreq


def overlap_indices(stage_raw, clean_raw) -> tuple[int, int, float]:
    start_sample, stop_sample = overlap_bounds_against_clean(stage_raw, clean_raw)
    start_idx = start_sample - int(stage_raw.first_samp)
    stop_idx = stop_sample - int(stage_raw.first_samp) + 1
    sfreq = float(stage_raw.info["sfreq"])
    return start_idx, stop_idx, sfreq


def maybe_harmonize_native_raw(
    stage_name: str,
    data: np.ndarray,
    sfreq: float,
    stage_raw,
    l_freq: float,
    h_freq: float,
) -> np.ndarray:
    if stage_name != "raw":
        return data

    highpass = float(stage_raw.info.get("highpass") or 0.0)
    lowpass = float(stage_raw.info.get("lowpass") or sfreq / 2.0)
    if highpass <= l_freq + 1e-6 and lowpass >= h_freq - 1e-6:
        return filter_data(
            data,
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            verbose="ERROR",
        )
    return data


def average_reference(data: np.ndarray) -> np.ndarray:
    return data - data.mean(axis=0, keepdims=True)


def available_named_channels(raw, preferred_names: tuple[str, ...]) -> list[str]:
    return [name for name in preferred_names if name in raw.ch_names]


def band_power(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return np.full(psd.shape[0], np.nan)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1)


def center_prominence_db(
    psd: np.ndarray,
    freqs: np.ndarray,
    center_hz: float,
    center_width_hz: float = 1.0,
    side_gap_hz: float = 1.0,
    side_width_hz: float = 1.0,
) -> float:
    half_center = center_width_hz / 2.0
    center_mask = (freqs >= center_hz - half_center) & (freqs <= center_hz + half_center)
    left_mask = (freqs >= center_hz - side_gap_hz - side_width_hz) & (
        freqs < center_hz - side_gap_hz
    )
    right_mask = (freqs > center_hz + side_gap_hz) & (
        freqs <= center_hz + side_gap_hz + side_width_hz
    )
    side_mask = left_mask | right_mask

    if not np.any(center_mask) or not np.any(side_mask):
        return math.nan

    eps = np.finfo(float).tiny
    center_power = np.maximum(psd[:, center_mask].mean(axis=1), eps)
    side_power = np.maximum(psd[:, side_mask].mean(axis=1), eps)
    return float(np.nanmedian(10.0 * np.log10(center_power / side_power)))


def artifact_window_metrics(
    data: np.ndarray,
    sfreq: float,
    window_seconds: float,
    threshold_uv: float,
) -> tuple[float, float]:
    window_samples = max(int(round(window_seconds * sfreq)), 1)
    n_times = data.shape[1]
    starts = list(range(0, n_times, window_samples))
    if not starts:
        starts = [0]

    window_max_p2p_uv: list[float] = []
    for start in starts:
        stop = min(start + window_samples, n_times)
        window = data[:, start:stop]
        if window.shape[1] < 2:
            continue
        p2p_uv = np.ptp(window, axis=1) * 1e6
        window_max_p2p_uv.append(float(np.nanmax(p2p_uv)))

    if not window_max_p2p_uv:
        return math.nan, math.nan

    values = np.asarray(window_max_p2p_uv, dtype=float)
    frac = float(np.mean(values > threshold_uv))
    p95 = float(np.nanpercentile(values, 95))
    return frac, p95


def eog_contamination_metrics(
    eeg_data: np.ndarray,
    sfreq: float,
    stage_raw,
    clean_raw,
    eog_channel_names: tuple[str, ...] = ("HEOG", "VEOG"),
) -> dict[str, float]:
    eog_names = available_named_channels(stage_raw, eog_channel_names)
    if not eog_names:
        return {
            "eog_channel_count": 0.0,
            "eog_abs_corr_median": math.nan,
            "eog_abs_corr_95th": math.nan,
        }

    start_idx, stop_idx, _ = overlap_indices(stage_raw, clean_raw)
    eog_picks = [stage_raw.ch_names.index(ch) for ch in eog_names]
    eog_data = stage_raw.get_data(
        picks=eog_picks,
        start=start_idx,
        stop=stop_idx,
        reject_by_annotation=None,
    )

    eeg_o = filter_data(eeg_data, sfreq=sfreq, l_freq=1.0, h_freq=15.0, verbose="ERROR")
    eog_o = filter_data(eog_data, sfreq=sfreq, l_freq=1.0, h_freq=15.0, verbose="ERROR")

    decim = max(int(round(sfreq / 100.0)), 1)
    eeg_o = eeg_o[:, ::decim]
    eog_o = eog_o[:, ::decim]

    eeg_o = eeg_o - eeg_o.mean(axis=1, keepdims=True)
    eog_o = eog_o - eog_o.mean(axis=1, keepdims=True)

    eeg_norm = np.linalg.norm(eeg_o, axis=1)
    eog_norm = np.linalg.norm(eog_o, axis=1)
    valid_eeg = eeg_norm > 0
    valid_eog = eog_norm > 0
    if not np.any(valid_eeg) or not np.any(valid_eog):
        return {
            "eog_channel_count": float(len(eog_names)),
            "eog_abs_corr_median": math.nan,
            "eog_abs_corr_95th": math.nan,
        }

    eeg_o = eeg_o[valid_eeg]
    eeg_norm = eeg_norm[valid_eeg]
    eog_o = eog_o[valid_eog]
    eog_norm = eog_norm[valid_eog]

    corr = np.abs(eeg_o @ eog_o.T / np.outer(eeg_norm, eog_norm))
    max_abs_corr = corr.max(axis=1)
    return {
        "eog_channel_count": float(len(eog_names)),
        "eog_abs_corr_median": float(np.nanmedian(max_abs_corr)),
        "eog_abs_corr_95th": float(np.nanpercentile(max_abs_corr, 95)),
    }


def psd_metrics(data: np.ndarray, sfreq: float) -> dict[str, float]:
    n_times = data.shape[1]
    n_fft = min(2048, max(256, 2 ** int(math.floor(math.log2(n_times)))))
    psd, freqs = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=1.0,
        fmax=min(100.0, sfreq / 2.0),
        n_fft=n_fft,
        n_overlap=n_fft // 2,
        verbose="ERROR",
    )

    eps = np.finfo(float).tiny
    signal = np.maximum(band_power(psd, freqs, 1.0, 40.0), eps)
    noise = np.maximum(band_power(psd, freqs, 45.0, 100.0), eps)
    spectral_snr_db = float(np.nanmedian(10.0 * np.log10(signal / noise)))

    prominence_20hz_db = center_prominence_db(psd, freqs, 20.0)

    line_noise_candidates = [
        center_prominence_db(psd, freqs, center_hz)
        for center_hz in (60.0, 80.0, 100.0)
    ]
    line_noise_candidates = [value for value in line_noise_candidates if not math.isnan(value)]
    line_noise_max_db = float(max(line_noise_candidates)) if line_noise_candidates else math.nan

    return {
        "spectral_snr_db": spectral_snr_db,
        "prominence_20hz_db": prominence_20hz_db,
        "line_noise_max_db": line_noise_max_db,
    }


def stage_metrics(
    stage_name: str,
    stage_raw,
    clean_raw,
    common_channels: list[str],
    l_freq: float,
    h_freq: float,
    artifact_threshold_uv: float,
    window_seconds: float,
) -> dict[str, float]:
    data, sfreq = extract_overlap_data(stage_raw, clean_raw, common_channels)
    data = maybe_harmonize_native_raw(stage_name, data, sfreq, stage_raw, l_freq, h_freq)
    data = average_reference(data)

    metrics: dict[str, float] = {
        "original_duration_sec": stage_duration_seconds(stage_raw),
        "metric_duration_sec": data.shape[1] / sfreq,
        "median_rms_uv": float(np.nanmedian(np.sqrt(np.mean(data**2, axis=1)) * 1e6)),
    }

    artifact_frac, p95_p2p_uv = artifact_window_metrics(
        data,
        sfreq=sfreq,
        window_seconds=window_seconds,
        threshold_uv=artifact_threshold_uv,
    )
    metrics["artifact_window_fraction"] = artifact_frac
    metrics["window_max_p2p_95th_uv"] = p95_p2p_uv
    metrics.update(psd_metrics(data, sfreq))
    metrics.update(eog_contamination_metrics(data, sfreq, stage_raw, clean_raw))
    return metrics


def add_stage_delta_columns(row: dict[str, object], stages: tuple[str, ...]) -> None:
    metric_suffixes: set[str] = set()
    for key in row:
        for stage_name in stages:
            prefix = f"{stage_name}_"
            if key.startswith(prefix):
                metric_suffixes.add(key[len(prefix) :])
                break

    for earlier_stage, later_stage in zip(stages, stages[1:]):
        for suffix in sorted(metric_suffixes):
            if suffix in DELTA_SKIP_SUFFIXES:
                continue
            earlier_key = f"{earlier_stage}_{suffix}"
            later_key = f"{later_stage}_{suffix}"
            if earlier_key not in row or later_key not in row:
                continue
            earlier_value = row[earlier_key]
            later_value = row[later_key]
            if not isinstance(earlier_value, (int, float)) or not isinstance(later_value, (int, float)):
                continue
            if math.isnan(float(earlier_value)) or math.isnan(float(later_value)):
                continue

            delta_name = f"{later_stage}_minus_{earlier_stage}_{suffix}"
            row[delta_name] = float(later_value) - float(earlier_value)

            if suffix in BETTER_HIGHER_METRICS:
                row[f"{later_stage}_vs_{earlier_stage}_improvement_{suffix}"] = (
                    float(later_value) - float(earlier_value)
                )
            elif suffix in BETTER_LOWER_METRICS:
                row[f"{later_stage}_vs_{earlier_stage}_improvement_{suffix}"] = (
                    float(earlier_value) - float(later_value)
                )


def compute_recording_metrics(recording: RecordingPaths, args: argparse.Namespace) -> dict[str, object]:
    requested_stage_set = set(args.stages)
    raw_open_path = (
        resolve_raw_open_path(recording, args.dataset_root, args.raw_root)
        if "raw" in requested_stage_set
        else None
    )
    stage_paths = {
        "raw": raw_open_path,
        "filt": recording.filt_path,
        "eyelink": recording.eyelink_path,
        "clean": recording.clean_path,
    }

    raws = {
        stage_name: open_raw(path)
        for stage_name, path in stage_paths.items()
        if path is not None and stage_name in requested_stage_set
    }
    clean_raw = raws["clean"]
    common_channels, excluded_bads = compute_common_eeg_channels(raws)

    row: dict[str, object] = {
        "subject": recording.subject,
        "task": recording.raw_task,
        "raw_task": recording.raw_task,
        "freeview_task": recording.freeview_task,
        "raw_path": str(raw_open_path) if raw_open_path is not None else str(recording.raw_path),
        "filt_path": str(recording.filt_path) if recording.filt_path else "",
        "eyelink_path": str(recording.eyelink_path) if recording.eyelink_path else "",
        "clean_path": str(recording.clean_path),
        "n_common_eeg_channels": len(common_channels),
        "n_excluded_eeg_channels": len(excluded_bads),
        "excluded_bads": ";".join(sorted(excluded_bads)),
    }

    raw_duration = stage_duration_seconds(raws["raw"]) if "raw" in raws else math.nan

    for stage_name in args.stages:
        row[f"{stage_name}_available"] = stage_name in raws
        if stage_name not in raws:
            continue

        metrics = stage_metrics(
            stage_name=stage_name,
            stage_raw=raws[stage_name],
            clean_raw=clean_raw,
            common_channels=common_channels,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            artifact_threshold_uv=args.artifact_threshold_uv,
            window_seconds=args.window_seconds,
        )
        for metric_name, value in metrics.items():
            row[f"{stage_name}_{metric_name}"] = value

        row[f"{stage_name}_retained_vs_raw_pct"] = (
            100.0 * metrics["original_duration_sec"] / raw_duration
            if not math.isnan(raw_duration) and raw_duration > 0
            else math.nan
        )

    row["clean_retained_vs_raw_pct_pair"] = row.get("clean_retained_vs_raw_pct", math.nan)
    add_stage_delta_columns(row, args.stages)
    return row


def summarize_by_task(recording_df: pd.DataFrame, stages: tuple[str, ...]) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    numeric_columns = [
        col
        for col in recording_df.columns
        if col not in SUMMARY_SKIP_COLUMNS and pd.api.types.is_numeric_dtype(recording_df[col])
    ]

    for task, group in recording_df.groupby("task", sort=True):
        row: dict[str, object] = {"task": task, "N paired": int(len(group))}
        for stage_name in stages:
            availability_col = f"{stage_name}_available"
            if availability_col in group:
                row[f"N {stage_name} available"] = int(group[availability_col].sum())

        for col in numeric_columns:
            if col.endswith("_available"):
                continue
            values = group[col].dropna()
            if values.empty:
                continue
            pretty = col.replace("_", " ")
            row[f"{pretty} mean"] = float(values.mean())
            row[f"{pretty} std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{pretty} median"] = float(values.median())
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values("task").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.stages = parse_stage_list(args.stages)
    args.subjects = set(parse_csv_arg(args.subjects))
    args.tasks = set(parse_csv_arg(args.tasks))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_recordings = discover_subject_recordings(
        dataset_root=args.dataset_root,
        raw_root=args.raw_root,
        require_all_tasks=args.require_all_tasks,
    )
    if not all_recordings:
        raise SystemExit("No eligible raw+clean recording pairs were found.")

    if args.subjects:
        all_recordings = [row for row in all_recordings if row.subject in args.subjects]
    if args.tasks:
        all_recordings = [row for row in all_recordings if row.raw_task in args.tasks]
    if not all_recordings:
        raise SystemExit("No eligible recordings remained after applying subject/task filters.")

    selected_subjects = sorted({row.subject for row in all_recordings})
    if args.max_subjects is not None:
        selected_subjects = selected_subjects[: args.max_subjects]

    recordings = [row for row in all_recordings if row.subject in set(selected_subjects)]
    if not recordings:
        raise SystemExit("Subject filter removed every eligible recording.")

    recordings_by_subject: dict[str, list[RecordingPaths]] = {}
    for recording in recordings:
        recordings_by_subject.setdefault(recording.subject, []).append(recording)

    if args.max_recordings_per_subject is not None:
        recordings_by_subject = {
            subject: rows[: args.max_recordings_per_subject]
            for subject, rows in recordings_by_subject.items()
        }

    metric_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []

    global_start = time.perf_counter()
    for subject in sorted(recordings_by_subject):
        subject_rows = recordings_by_subject[subject]
        subject_start = time.perf_counter()

        for recording in subject_rows:
            record_start = time.perf_counter()
            print(
                f"  {subject} :: {recording.raw_task} -> {recording.freeview_task}",
                flush=True,
            )
            try:
                row = compute_recording_metrics(recording, args)
            except FileNotFoundError as exc:
                print(f"    skipped: {exc}", flush=True)
                continue
            row["record_eval_seconds"] = time.perf_counter() - record_start
            metric_rows.append(row)
            print(
                f"    finished in {row['record_eval_seconds']:.2f}s",
                flush=True,
            )

        subject_seconds = time.perf_counter() - subject_start
        completed_recordings = sum(1 for row in metric_rows if row["subject"] == subject)
        timing_rows.append(
            {
                "subject": subject,
                "n_recordings": completed_recordings,
                "subject_eval_seconds": subject_seconds,
                "seconds_per_recording": (
                    subject_seconds / completed_recordings if completed_recordings else math.nan
                ),
            }
        )
        print(
            f"{subject}: {subject_seconds:.2f}s for {completed_recordings} completed recording(s)",
            flush=True,
        )

    total_seconds = time.perf_counter() - global_start
    print(
        f"Processed {len(metric_rows)} recordings from {len(timing_rows)} subjects "
        f"in {total_seconds:.2f}s",
        flush=True,
    )

    if not metric_rows:
        raise SystemExit("No recordings were successfully processed.")

    recording_df = pd.DataFrame(metric_rows).sort_values(["subject", "task"]).reset_index(drop=True)
    timing_df = pd.DataFrame(timing_rows).sort_values("subject").reset_index(drop=True)
    summary_df = summarize_by_task(recording_df, args.stages)

    recording_csv = args.output_dir / "recording_metrics.csv"
    summary_csv = args.output_dir / "task_summary.csv"
    timing_csv = args.output_dir / "subject_timings.csv"

    recording_df.to_csv(recording_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    timing_df.to_csv(timing_csv, index=False)

    print(f"Wrote {recording_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {timing_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
