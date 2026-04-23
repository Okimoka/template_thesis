from types import SimpleNamespace
import mne
import os.path
import re
import numpy as np
from io import StringIO
from mne_bids import BIDSPath
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
from scipy.signal import correlate, find_peaks, peak_prominences
from scipy.stats import pearsonr
import csv
import json
from collections import Counter
from difflib import SequenceMatcher

from ..._config_utils import (
    _bids_kwargs,
    get_eeg_reference,
    get_runs,
    get_sessions,
    get_subjects,
    _get_ss,
    _get_ssrt
)
from ..._import_data import annotations_to_events, make_epochs
from ..._logging import gen_log_kwargs, logger
from ..._parallel import (
    get_parallel_backend,
    get_serial_report_exec_params,
    parallel_func,
)
from ..._reject import _get_reject
from ..._report import _open_report
from ..._run import _prep_out_files, _update_for_splits, failsafe_run, save_logs

"""
Parts of this code have been written by an LLM
"""

ZERO_INVALID_ET_CHANNELS: tuple[str, ...] = (
    "L Raw X [px]",
    "L Raw Y [px]",
    "R Raw X [px]",
    "R Raw Y [px]",
    "L Dia X [px]",
    "L Dia Y [px]",
    "L Mapped Diameter [mm]",
    "R Dia X [px]",
    "R Dia Y [px]",
    "R Mapped Diameter [mm]",
    "L CR1 X [px]",
    "L CR1 Y [px]",
    "L CR2 X [px]",
    "L CR2 Y [px]",
    "R CR1 X [px]",
    "R CR1 Y [px]",
    "R CR2 X [px]",
    "R CR2 Y [px]",
    "L POR X [px]",
    "L POR Y [px]",
    "R POR X [px]",
    "R POR Y [px]",
    "L EPOS X",
    "L EPOS Y",
    "L EPOS Z",
    "R EPOS X",
    "R EPOS Y",
    "R EPOS Z",
    "L GVEC X",
    "L GVEC Y",
    "L GVEC Z",
    "R GVEC X",
    "R GVEC Y",
    "R GVEC Z",
)

STEP_LIKE_ET_CHANNELS: tuple[str, ...] = (
    "Timing",
    "L Validity",
    "R Validity",
    "Pupil Confidence",
    "L Plane",
    "R Plane",
)

def ascii_sanitize(s: str) -> str:
    s = (s.replace("°", "deg")
           .replace("µ", "u")
           .replace("²", "2"))
    return s.encode("ascii", "ignore").decode("ascii")


def _interpolate_zero_invalid_et_channels(raw_et: mne.io.BaseRaw) -> None:
    for ch_name in ZERO_INVALID_ET_CHANNELS:
        if ch_name not in raw_et.ch_names:
            continue
        signal = raw_et._data[raw_et.ch_names.index(ch_name)]
        invalid_idx = np.flatnonzero(signal == 0)
        if invalid_idx.size == 0:
            continue
        valid_idx = np.flatnonzero(signal != 0)
        if valid_idx.size == 0:
            continue
        signal[invalid_idx] = np.interp(invalid_idx, valid_idx, signal[valid_idx])


def _round_step_like_et_channels(raw_et: mne.io.BaseRaw) -> None:
    for ch_name in STEP_LIKE_ET_CHANNELS:
        if ch_name not in raw_et.ch_names:
            continue
        ch_idx = raw_et.ch_names.index(ch_name)
        raw_et._data[ch_idx] = np.rint(raw_et._data[ch_idx])


def _annotation_onsets(annotations: list) -> np.ndarray:
    return np.asarray(
        [float(annotation["onset"]) for annotation in annotations],
        dtype=float,
    )


def _safe_relative_error(error: float, scale: float) -> float:
    eps = np.finfo(float).eps
    if scale > eps:
        return float(error / scale)
    return 0.0 if error <= eps else np.inf


def _evaluate_sync_alignment(
    reference_annotations: list,
    candidate_annotations: list,
) -> dict[str, float | int]:
    reference_onsets = _annotation_onsets(reference_annotations)
    candidate_onsets = _annotation_onsets(candidate_annotations)
    n_events = len(reference_onsets)
    label_matches = int(sum(
        str(ref["description"]) == str(cand["description"])
        for ref, cand in zip(reference_annotations, candidate_annotations)
    ))

    quality: dict[str, float | int] = dict(
        n_events=n_events,
        label_matches=label_matches,
        label_match_fraction=(label_matches / n_events if n_events else 0.0),
        slope=np.nan,
        intercept=np.nan,
        corr=np.nan,
        fit_rmse=np.nan,
        fit_rmse_rel=np.nan,
        max_abs_error=np.nan,
        interval_mae=np.nan,
        interval_mae_rel=np.nan,
    )
    if n_events < 2:
        return quality

    coef = np.polyfit(candidate_onsets, reference_onsets, 1)
    preds = np.poly1d(coef)(candidate_onsets)
    residuals = reference_onsets - preds
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    quality["slope"] = float(coef[0])
    quality["intercept"] = float(coef[1])
    quality["fit_rmse"] = rmse
    quality["max_abs_error"] = float(np.max(np.abs(residuals)))

    reference_diffs = np.diff(reference_onsets)
    candidate_diffs = np.diff(candidate_onsets)
    if reference_diffs.size:
        gap_scale = float(np.median(np.abs(reference_diffs)))
        quality["fit_rmse_rel"] = _safe_relative_error(rmse, gap_scale)

    valid = np.abs(candidate_diffs) > np.finfo(float).eps
    if np.any(valid):
        scale = np.median(reference_diffs[valid] / candidate_diffs[valid])
        interval_errors = np.abs(scale * candidate_diffs[valid] - reference_diffs[valid])
        interval_mae = float(np.mean(interval_errors))
        quality["interval_mae"] = interval_mae
        gap_scale = float(np.median(np.abs(reference_diffs[valid])))
        quality["interval_mae_rel"] = _safe_relative_error(interval_mae, gap_scale)

    if (
        n_events >= 3
        and np.std(reference_onsets) > np.finfo(float).eps
        and np.std(candidate_onsets) > np.finfo(float).eps
    ):
        quality["corr"] = float(np.corrcoef(candidate_onsets, reference_onsets)[0, 1])

    return quality


def _format_sync_alignment_quality(quality: dict[str, float | int]) -> str:
    def _fmt(value: float | int) -> str:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        value = float(value)
        return f"{value:.6g}" if np.isfinite(value) else str(value)

    return (
        f"label_matches={quality['label_matches']}/{quality['n_events']}, "
        f"interval_mae_rel={_fmt(quality['interval_mae_rel'])}, "
        f"fit_rmse_rel={_fmt(quality['fit_rmse_rel'])}, "
        f"corr={_fmt(quality['corr'])}, "
        f"slope={_fmt(quality['slope'])}"
    )


def _is_decisive_metric(
    best_value: float,
    other_value: float,
    *,
    threshold: float,
    factor: float,
) -> bool:
    return bool(
        np.isfinite(best_value)
        and best_value <= threshold
        and (
            not np.isfinite(other_value)
            or best_value * factor <= other_value
        )
    )


def _maybe_trim_mismatched_sync_edges(
    *,
    sync_annotations: list,
    et_sync_annotations: list,
    sync_indices: list[int],
    et_sync_indices: list[int],
) -> tuple[list[int], list[int], int]:
    if len(sync_indices) == len(et_sync_indices):
        return sync_indices, et_sync_indices, 0

    if len(sync_indices) > len(et_sync_indices):
        longer_name = "EEG"
        longer_indices = list(sync_indices)
        longer_annotations = list(sync_annotations)
        shorter_indices = list(et_sync_indices)
        shorter_annotations = list(et_sync_annotations)
    else:
        longer_name = "ET"
        longer_indices = list(et_sync_indices)
        longer_annotations = list(et_sync_annotations)
        shorter_indices = list(sync_indices)
        shorter_annotations = list(sync_annotations)

    diff = len(longer_indices) - len(shorter_indices)
    if diff <= 0 or len(shorter_indices) < 2:
        return sync_indices, et_sync_indices, 0

    candidates: dict[str, dict[str, list[int] | dict[str, float | int]]] = dict()
    for side in ("start", "end"):
        if side == "start":
            kept_indices = longer_indices[diff:]
            kept_annotations = longer_annotations[diff:]
        else:
            kept_indices = longer_indices[:-diff]
            kept_annotations = longer_annotations[:-diff]
        candidates[side] = dict(
            kept_indices=kept_indices,
            quality=_evaluate_sync_alignment(shorter_annotations, kept_annotations),
        )

    for side in ("start", "end"):
        logger.info(**gen_log_kwargs(
            message=(
                f"Edge-trim candidate for longer {longer_name} list ({side}, drop {diff}): "
                f"{_format_sync_alignment_quality(candidates[side]['quality'])}"
            )
        ))

    def _candidate_rank(side: str) -> tuple[float, float, float]:
        quality = candidates[side]["quality"]
        interval_mae_rel = float(quality["interval_mae_rel"])
        fit_rmse_rel = float(quality["fit_rmse_rel"])
        return (
            -float(quality["label_matches"]),
            interval_mae_rel if np.isfinite(interval_mae_rel) else np.inf,
            fit_rmse_rel if np.isfinite(fit_rmse_rel) else np.inf,
        )

    best_side = min(("start", "end"), key=_candidate_rank)
    other_side = "end" if best_side == "start" else "start"
    best_quality = candidates[best_side]["quality"]
    other_quality = candidates[other_side]["quality"]

    label_decisive = bool(
        best_quality["label_matches"] == best_quality["n_events"]
        and best_quality["label_matches"] > other_quality["label_matches"]
    )
    timing_decisive = bool(
        best_quality["n_events"] >= 3
        and _is_decisive_metric(
            float(best_quality["interval_mae_rel"]),
            float(other_quality["interval_mae_rel"]),
            threshold=0.05,
            factor=5.0,
        )
        and _is_decisive_metric(
            float(best_quality["fit_rmse_rel"]),
            float(other_quality["fit_rmse_rel"]),
            threshold=0.05,
            factor=5.0,
        )
    )

    if not (label_decisive or timing_decisive):
        logger.info(**gen_log_kwargs(
            message=(
                "Sync-event mismatch remained ambiguous after evaluating start/end "
                "edge trimming; automatic trimming was skipped."
            )
        ))
        return sync_indices, et_sync_indices, 0

    kept_indices = list(candidates[best_side]["kept_indices"])
    trimmed_metric = diff if best_side == "start" else -diff

    logger.info(**gen_log_kwargs(
        message=(
            f"Auto-trimmed {diff} sync event(s) from the {best_side} of the longer "
            f"{longer_name} list. sync_events_edge_trimmed={trimmed_metric} "
            "(positive=start, negative=end)."
        )
    ))

    if longer_name == "EEG":
        sync_indices = kept_indices
    else:
        et_sync_indices = kept_indices

    return sync_indices, et_sync_indices, trimmed_metric


def _find_first_last_shared_sync_indices(
    *,
    sync_annotations: list,
    et_sync_annotations: list,
    sync_indices: list[int],
    et_sync_indices: list[int],
) -> tuple[list[int], list[int]] | None:
    eeg_descriptions = [str(annotation["description"]) for annotation in sync_annotations]
    et_descriptions = [str(annotation["description"]) for annotation in et_sync_annotations]
    matcher = SequenceMatcher(a=eeg_descriptions, b=et_descriptions, autojunk=False)

    matched_pairs: list[tuple[int, int]] = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            matched_pairs.append((block.a + offset, block.b + offset))

    if len(matched_pairs) < 2:
        return None

    first_eeg_idx, first_et_idx = matched_pairs[0]
    last_eeg_idx, last_et_idx = matched_pairs[-1]
    if first_eeg_idx == last_eeg_idx or first_et_idx == last_et_idx:
        return None

    return (
        [sync_indices[first_eeg_idx], sync_indices[last_eeg_idx]],
        [et_sync_indices[first_et_idx], et_sync_indices[last_et_idx]],
    )


def _augment_two_point_sync_pair(
    *,
    sync_times: list[float],
    et_sync_times: list[float],
    raw: mne.io.BaseRaw,
    raw_et: mne.io.BaseRaw,
) -> tuple[list[float], list[float], bool]:
    sync_times = [float(time) for time in sync_times]
    et_sync_times = [float(time) for time in et_sync_times]
    if len(sync_times) != 2 or len(et_sync_times) != 2:
        return sync_times, et_sync_times, False

    eeg_delta = float(sync_times[1] - sync_times[0])
    et_delta = float(et_sync_times[1] - et_sync_times[0])
    if eeg_delta == 0:
        eeg_delta = 1.0 / float(raw.info["sfreq"])
    if et_delta == 0:
        et_delta = 1.0 / float(raw_et.info["sfreq"])

    return (
        [*sync_times, float(sync_times[-1] + eeg_delta)],
        [*et_sync_times, float(et_sync_times[-1] + et_delta)],
        True,
    )


def _sync_times_pass_realign_checks(
    *,
    sync_times: list[float],
    et_sync_times: list[float],
) -> bool:
    if len(sync_times) != len(et_sync_times) or len(sync_times) < 2:
        return False

    try:
        corr, p_value = pearsonr(
            np.asarray(et_sync_times, dtype=float),
            np.asarray(sync_times, dtype=float),
        )
    except Exception:
        return False

    return bool(np.isfinite(corr) and np.isfinite(p_value) and corr > 0 and p_value <= 0.05)


def _force_align_recording_starts(
    *,
    raw: mne.io.BaseRaw,
    raw_et: mne.io.BaseRaw,
) -> tuple[list[float], list[float]]:
    raw_et.load_data().resample(raw.info["sfreq"])
    with raw_et.info._unlock():
        raw_et.info["sfreq"] = raw.info["sfreq"]

    overlap_end = float(min(raw.times[-1], raw_et.times[-1]))
    if overlap_end > 0:
        raw.crop(0, overlap_end)
        raw_et.crop(0, overlap_end)
    else:
        overlap_end = 1.0 / float(raw.info["sfreq"])

    logger.info(**gen_log_kwargs(
        message=(
            "Force-aligned EEG and ET data to recording start times and cropped both "
            f"recordings to {overlap_end:.3f}s of shared duration."
        )
    ))
    return [0.0, overlap_end], [0.0, overlap_end]

def gaussian(x, A, mu, sigma):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2))

def fit_gauss_to_xcorr(lags, y, half_window):
    peak_idx = np.argmax(y) # upward peak only
    mu0 = lags[peak_idx] # initial guess for the peak location

    # restrict fit to specified window around the peak
    fit_mask  = np.abs(lags - mu0) <= half_window
    base_mask = ~fit_mask
    x_fit = lags[fit_mask]
    y_fit = y[fit_mask]

    # estimate baseline from edge range
    b = np.median(y[base_mask]) #or .mean()
    # sigma can be chosen smaller as well for tighter peaks
    initial_guess = (y[peak_idx] - b, mu0, half_window / 3)

    A, mu, sigma = curve_fit(
        gaussian, x_fit, y_fit - b, p0=initial_guess, maxfev=1600
    )[0]
    return abs(A), mu, abs(sigma), b

"""
Parses .txt files generated by SMI Vision's IDF Converter (*_Events.txt and *_Samples.txt)
and generates an mne.Raw object with corresponding Annotations.
"""
def read_raw_iview(event_fname: str):

    with open(event_fname, "r", encoding="utf-8", errors="ignore") as f:
        ev_lines = f.readlines()

    event_types = ["Saccade", "Fixation", "Blink", "User"]

    # TODO this is needed so the current pipeline doesn't crash
    # Could later instead be populated with dfs from the events file
    dummy_raw_extras = [{
        "dfs": {
            "fixations": pd.DataFrame(columns=['time', 'duration', 'end_time', 'eye', 'fix_avg_x', 'fix_avg_y', 'fix_avg_pupil_size', 'sacc_start_x', 'sacc_start_y', 'sacc_end_x', 'sacc_end_y', 'sacc_visual_angle', 'peak_velocity']),
            "saccades": pd.DataFrame(),
            "blinks": pd.DataFrame(),
        }
    }]

    # read table headers for the different events
    headers = {}
    for ev_type in event_types:
        marker = f"Table Header for {ev_type}"
        marker_idx = next(i for i, ln in enumerate(ev_lines) if ln.startswith(marker))
        headers[ev_type] = ev_lines[marker_idx + 1].rstrip("\n").split("\t")

    ann_on, ann_du, ann_ds, ann_ex = [], [], [], []

    for ev_type in event_types:
        cols = headers[ev_type]
        rows = [ln for ln in ev_lines if ln.startswith(ev_type)]
        if not rows:
            continue

        df = pd.read_csv(
            StringIO("".join(rows)),
            sep="\t",
            names=cols,
            header=None,
            engine="python",
            dtype="string",
        )

        #print("TYPE : " + str(ev_type))
        #print(df)

        start_us = pd.to_numeric(df["Start"]).to_numpy(dtype=float)
        ann_on.extend((start_us / 1e6).tolist())

        #First cols will always be "Event Type, Trial, Number, Start, Duration"
        #Everything after duration is extra
        extra_cols = cols[5:] if len(cols) > 5 else None
        if extra_cols:
            extras_df = df[extra_cols].apply(pd.to_numeric)
            ann_ex.extend([d or None for d in extras_df.to_dict(orient="records")])
        else:
            ann_ex.extend([None] * len(df))

        if(ev_type == "User"):
            ann_ds.extend(df["Description"].to_list())
            ann_du.extend([0.0] * len(df))
        else:
            dur_us = pd.to_numeric(df["Duration"]).to_numpy(dtype=float)
            ann_ds.extend(df["Event Type"].to_list())
            ann_du.extend((dur_us / 1e6).tolist())



    # events file done, now parse samples file
    sample_fname = event_fname.copy().update(suffix="Samples")

    # some subjects only have an events file (vice versa does not exist)
    # in this case, create minimal RawArray containing annotations from events file
    if not os.path.isfile(sample_fname):
        # can't use times[0] like later, so min(ann_on) is used (which is usually equivalent)
        ann_on = [e - min(ann_on) for e in ann_on]
        annotations = mne.Annotations(onset=ann_on, duration=ann_du, description=ann_ds, extras=ann_ex)

        # can't estimate sfreq from len(samples)/duration, so use sfreq specified in header
        sfreq = next((int(line.split("\t")[1]) for line in ev_lines if line.startswith("Sample Rate:\t")),60)
        info_events = mne.create_info(ch_names=[],sfreq=sfreq)

        dur_s = (max(ann_on) - min(ann_on))
        n_times = max(2, int(np.ceil(dur_s * sfreq)) + 1)
        raw_et = mne.io.RawArray(np.empty((0, n_times)), info_events)

        raw_et.set_annotations(annotations)
        raw_et._raw_extras = dummy_raw_extras
        return raw_et, -1


    with open(sample_fname, "r", encoding="utf-8", errors="ignore") as f:
        sm_lines = f.readlines()

    header_idx = next(i for i, ln in enumerate(sm_lines) if ln.startswith("Time"))
    header = sm_lines[header_idx].rstrip("\n").split("\t")

    df = pd.read_csv(
        sample_fname,
        sep="\t",
        skiprows=header_idx + 1,
        names=header,
        header=None,
        engine="python",
        dtype="string",
    )

    df = df.loc[df["Type"] != "MSG"].copy() # already have MSG events from events file

    numeric_cols = [c for c in df.columns if c != "Type"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    #bad samples will be 0.00 on all these cols
    #comment these out to count number of nan samples
    zero_cols = header[3:11]
    et_nan_samples = int(df[zero_cols].eq(0).all(axis=1).sum())

    #remove extra columns (usually Frame and Aux1)
    df = df.dropna(axis=1, how="all")

    times = df["Time"].to_numpy(dtype=float)
    ch_names = [c for c in df.columns if c not in ("Time", "Type")]
    data = df[ch_names].to_numpy(dtype=float).T  # (n_ch, n_times)

    #if just using regular sfreq, some annotations may fall outside raw time range
    dur_s = (times[-1] - times[0]) / 1e6
    sfreq_avg = (len(times) - 1) / dur_s

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq_avg, ch_types=["misc"] * len(ch_names))
    ann_on = [e - (times[0]/1e6) for e in ann_on] # normalize onset using lowest timestamp
    annotations = mne.Annotations(onset=ann_on, duration=ann_du, description=ann_ds, extras=ann_ex)



    raw_et = mne.io.RawArray(data, info)
    raw_et.set_annotations(annotations)
    raw_et._raw_extras = dummy_raw_extras

    #sorted_annotations = sorted(raw_et.annotations, key=lambda x: x["onset"])
    #print(sorted_annotations)

    return raw_et, et_nan_samples

def _check_HEOG_ET_vars(cfg):
    # helper function for sorting out heog and et channels
    bipolar = False
    if isinstance(cfg.sync_heog_ch, tuple):
        heog_ch = "bi_HEOG"
        bipolar = True
    else:
        heog_ch = cfg.sync_heog_ch
    
    if isinstance(cfg.sync_et_ch, tuple):
        et_ch = list(cfg.sync_et_ch)
    else:
        et_ch = [cfg.sync_et_ch]
    
    return heog_ch, et_ch, bipolar

def _select_eog_candidate(
    candidates: str | list[str] | tuple[str, ...],
    *,
    present_channels: set[str],
    bad_channels: set[str],
) -> tuple[str | None, bool]:
    if isinstance(candidates, str):
        candidates = [candidates]
    else:
        candidates = list(candidates)

    bad_candidates = list()
    bad_candidate_indices = list()
    for idx, candidate in enumerate(candidates):
        if candidate not in present_channels:
            continue
        if candidate in bad_channels:
            bad_candidates.append(candidate)
            bad_candidate_indices.append(idx)
            continue
        if candidate in present_channels and candidate not in bad_channels:
            return candidate, idx > 0

    if bad_candidates:
        return bad_candidates[-1], bad_candidate_indices[-1] > 0

    return None, False

def _get_eog_electrode_metrics(
    *,
    cfg: SimpleNamespace,
    raw: mne.io.BaseRaw,
) -> tuple[dict[str, str | None], dict[str, bool]]:
    eog_electrodes_used: dict[str, str | None] = {
        "HEOG_anode": None,
        "HEOG_cathode": None,
        "VEOG_anode": None,
        "VEOG_cathode": None,
    }
    eog_electrode_is_fallback: dict[str, bool] = {
        key: False for key in eog_electrodes_used
    }

    if not cfg.eeg_bipolar_channels:
        return eog_electrodes_used, eog_electrode_is_fallback

    present_channels = set(raw.ch_names)
    bad_channels = set(raw.info["bads"])

    for eog_name in ("HEOG", "VEOG"):
        if eog_name not in cfg.eeg_bipolar_channels:
            continue
        anode_cfg, cathode_cfg = cfg.eeg_bipolar_channels[eog_name]
        anode_used, anode_is_fallback = _select_eog_candidate(
            anode_cfg,
            present_channels=present_channels,
            bad_channels=bad_channels,
        )
        cathode_used, cathode_is_fallback = _select_eog_candidate(
            cathode_cfg,
            present_channels=present_channels,
            bad_channels=bad_channels,
        )
        eog_electrodes_used[f"{eog_name}_anode"] = anode_used
        eog_electrode_is_fallback[f"{eog_name}_anode"] = anode_is_fallback
        eog_electrodes_used[f"{eog_name}_cathode"] = cathode_used
        eog_electrode_is_fallback[f"{eog_name}_cathode"] = cathode_is_fallback

    return eog_electrodes_used, eog_electrode_is_fallback

def _mark_calibration_as_bad(raw, cfg):
    # marks recalibration beginnings and ends as one bad segment
    cur_idx = None
    cur_start_time = 0.
    last_status = None
    for annot in raw.annotations:
        calib_match = re.match(cfg.sync_calibration_string, annot["description"])
        if not calib_match: continue
        calib_status, calib_idx = calib_match.group(1), calib_match.group(2)
        if calib_idx  == cur_idx and calib_status == "end":
            duration = annot["onset"] - cur_start_time
            raw.annotations.append(cur_start_time, duration, f"BAD_Recalibrate {calib_idx}")
            cur_idx, cur_start_time = None, 0.
        elif calib_status == "start" and cur_idx is None:
            cur_idx = calib_idx
            cur_start_time = annot["onset"]
        elif calib_status == last_status:
            logger.info(**gen_log_kwargs(message=f"Encountered apparent duplicate calibration event ({calib_status}, {calib_idx}) - skipping"))
        elif calib_status == "start" and cur_idx is not None:
            raise ValueError(f"Annotation {annot['description']} could not be assigned membership")
        last_status = calib_status
        
    return raw
        

def get_input_fnames_sync_eyelink(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
) -> dict:
    
    # Get from config file whether `task` is specified in the et file name
    if cfg.et_has_task == True:
        et_task = cfg.task
    else:
        et_task = None

    bids_basename = BIDSPath(
        subject=subject,
        session=session,
        run=run,
        task=cfg.task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
        extension=".fif",
    )

    et_bids_basename = BIDSPath(
        subject=subject,
        session=session,
        task=et_task,
        acquisition=cfg.acq,
        recording=cfg.rec,
        datatype="misc",
        root=cfg.bids_root,
        suffix="et",
        check=False,
        extension=".asc",
    )

 
    in_files = dict()

    key = f"raw_run-{run}"
    in_files[key] = bids_basename.copy().update(
        processing=cfg.processing, suffix="raw"
    )

    

    if cfg.et_has_run:
        et_bids_basename.update(run=run)

    # _update_for_splits(in_files, key, single=True) # TODO: Find out if we need to add this or not

    # filename formats to check (suffix, extension)
    variants = [
        ("et", ".asc"),
        ("physio", ".asc"),
        ("et", ".edf"),
        ("physio", ".edf"),
        ("Events", ".txt") # SMI
    ]

    for suffix, ext in variants:
        candidate = et_bids_basename.copy().update(suffix=suffix, extension=ext)

        if os.path.isfile(candidate.fpath):
            key = f"et_run-{run}"
            in_files[key] = candidate
            return in_files

        if suffix == "et" and ext == ".asc":
            logger.info(**gen_log_kwargs(
                message=f"Couldn't find {candidate} file. Trying suffix='physio' before checking .edf."
            ))
        elif suffix == "physio" and ext == ".asc":
            logger.info(**gen_log_kwargs(
                message=f"Also couldn't find {candidate}; checking .edf for suffix='et'."
            ))
        elif suffix == "et" and ext == ".edf":
            logger.info(**gen_log_kwargs(
                message=f"Also didn't find {candidate} file, checking .edf with suffix _physio"
            ))
        elif suffix == "physio" and ext == ".edf":
            logger.info(**gen_log_kwargs(
                message=f"Also didn't find {candidate} file, last try, .txt with suffix _Events"
            ))

    # previous candidates all failed
    logger.error(**gen_log_kwargs(
        message=f"Also didn't find {candidate} file, no valid files exist for ET sync."
    ))
    raise FileNotFoundError(
        f"For run {run}, could neither find .asc nor .edf nor .txt eye-tracking file. "
        f"Please double-check the file names."
    )


@failsafe_run(
    get_input_fnames=get_input_fnames_sync_eyelink,
)
def sync_eyelink(
 *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
    in_files: dict,
) -> dict:
    
    """Run Sync for Eyelink."""
    import matplotlib.pyplot as plt
    from scipy.signal import correlate

    raw_fname = in_files.pop(f"raw_run-{run}")
    et_fname = in_files.pop(f"et_run-{run}")
    logger.info(**gen_log_kwargs(message=f"Found the following eye-tracking files: {et_fname}"))
    out_files = dict()
    bids_basename = raw_fname.copy().update(processing=None, split=None) #TODO: Do we need the `split=None` here?
    out_files["eyelink_eeg"] = bids_basename.copy().update(processing="eyelink", suffix="raw")
    del bids_basename

    participants_info = dict(
        release_number=np.nan,
        availability=np.nan
    )

    if os.path.isfile(cfg.bids_root / "participants.tsv"):
        with (cfg.bids_root / "participants.tsv").open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if (row.get('participant_id') or '').strip() == f"sub-{subject}":

                    release_val = (row.get('release_number') or '').strip()
                    if release_val:
                        participants_info['release_number'] = release_val

                    # Update the value at column named by cfg.task if present/non-empty
                    task_val = (row.get(cfg.task) or '').strip()
                    if task_val:
                        participants_info["availability"] = task_val



    # Ideally, this would be done in one of the previous steps where all folders are created (in `_01_init_derivatives_dir.py`). 
    logger.info(**gen_log_kwargs(message=f"Create `misc` folder for eye-tracking events."))
    out_dir_misc = cfg.deriv_root / f"sub-{subject}"
    if session is not None:
        out_dir_misc /= f"ses-{session}"

    out_dir_misc /= "misc"
    out_dir_misc.mkdir(exist_ok=True, parents=True) # TODO: Check whether the parameter settings make sense or if there is a danger that something could be accidentally overwritten

    out_files["eyelink_et_events"] = et_fname.copy().update(root=cfg.deriv_root, suffix="et_events", extension=".tsv")
    
    msg = f"Syncing Eyelink ({et_fname.basename}) and EEG data ({raw_fname.basename})."
    logger.info(**gen_log_kwargs(message=msg))
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    eog_electrodes_used, eog_electrode_is_fallback = _get_eog_electrode_metrics(
        cfg=cfg, raw=raw
    )

    et_format = et_fname.extension
    nan_values = 0

    if et_format == '.edf':
        logger.info(**gen_log_kwargs(message=f"Converting {et_fname} file to `.asc` using edf2asc."))
        import subprocess
        subprocess.run(["edf2asc", et_fname]) # TODO: Still needs to be tested
        et_fname.update(extension='.asc')
        raw_et = mne.io.read_raw_eyelink(et_fname, find_overlaps=False) # TODO: Make find_overlaps optional
    elif et_format == '.asc':
        raw_et = mne.io.read_raw_eyelink(et_fname, find_overlaps=False) # TODO: Make find_overlaps optional
    elif et_format == '.txt':
        raw_et, nan_values = read_raw_iview(et_fname)

    else:
        raise AssertionError("ET file is neither an `.asc` nor an `.edf` nor a `.txt`. This should not have happened.")


    metrics = dict(
        subject=subject,
        release=participants_info["release_number"],
        task=cfg.task,
        availability=participants_info["availability"],
        et_nan_values = nan_values,
        eeg_sampling_rate_hz = raw.info["sfreq"],
        et_sampling_rate_hz = raw_et.info["sfreq"],
        eeg_channel_cnt = len(raw.ch_names),
        eeg_bad_channel_count = len(raw.info["bads"]),
        et_channel_cnt = len(raw_et.ch_names),
        eog_electrode_used_HEOG_anode = eog_electrodes_used["HEOG_anode"],
        eog_electrode_used_HEOG_cathode = eog_electrodes_used["HEOG_cathode"],
        eog_electrode_used_VEOG_anode = eog_electrodes_used["VEOG_anode"],
        eog_electrode_used_VEOG_cathode = eog_electrodes_used["VEOG_cathode"],
        eog_electrode_is_fallback_HEOG_anode = eog_electrode_is_fallback["HEOG_anode"],
        eog_electrode_is_fallback_HEOG_cathode = eog_electrode_is_fallback["HEOG_cathode"],
        eog_electrode_is_fallback_VEOG_anode = eog_electrode_is_fallback["VEOG_anode"],
        eog_electrode_is_fallback_VEOG_cathode = eog_electrode_is_fallback["VEOG_cathode"],
        sync_events_edge_trimmed = 0,  # +N=start, -N=end on the longer list
        sync_failsafe_level = 0,
    )

    # If the user did not specify a regular expression for the eye-tracking sync events, it is assumed that it's
    # identical to the regex for the EEG sync events
    if not cfg.sync_eventtype_regex_et:
        cfg.sync_eventtype_regex_et = cfg.sync_eventtype_regex

    #et_sync_times = [annotation["onset"] for annotation in raw_et.annotations if re.search(cfg.sync_eventtype_regex_et,annotation["description"])]

    #sync_times    = [annotation["onset"] for annotation in raw.annotations    if re.search(cfg.sync_eventtype_regex,   annotation["description"])]

    #"""
    
    et_sync_pattern = re.compile(cfg.sync_eventtype_regex_et)
    sync_pattern = re.compile(cfg.sync_eventtype_regex)

    et_sync_indices = [
        idx for idx, annotation in enumerate(raw_et.annotations)
        if et_sync_pattern.fullmatch(str(annotation["description"]))
    ]
    sync_indices = [
        idx for idx, annotation in enumerate(raw.annotations)
        if sync_pattern.fullmatch(str(annotation["description"]))
    ]
    et_sync_annotations = [raw_et.annotations[idx] for idx in et_sync_indices]
    sync_annotations = [raw.annotations[idx] for idx in sync_indices]
    et_sync_times = [annotation["onset"] for annotation in et_sync_annotations]
    sync_times = [annotation["onset"] for annotation in sync_annotations]

    if len(et_sync_times) != len(sync_times):
        eeg_match_counts = Counter(
            str(annotation["description"]) for annotation in sync_annotations
        )
        et_match_counts = Counter(
            str(annotation["description"]) for annotation in et_sync_annotations
        )

        logger.info(**gen_log_kwargs(
            message=(
                f"Sync-event mismatch details. "
                f"EEG regex={cfg.sync_eventtype_regex!r} matched {len(sync_times)} / {len(raw.annotations)} annotations; "
                f"ET regex={cfg.sync_eventtype_regex_et!r} matched {len(et_sync_times)} / {len(raw_et.annotations)} annotations."
            )
        ))
        logger.info(**gen_log_kwargs(
            message=f"EEG matched annotation counts (top 10): {dict(eeg_match_counts.most_common(10))}"
        ))
        logger.info(**gen_log_kwargs(
            message=f"ET matched annotation counts (top 10): {dict(et_match_counts.most_common(10))}"
        ))

        et_start_recording_onsets = [
            float(annotation["onset"]) for annotation in raw_et.annotations
            if "start_eye_recording" in str(annotation["description"])
        ]
        if et_start_recording_onsets:
            last_start_recording = et_start_recording_onsets[-1]
            et_matches_before_last_start = [
                annotation for annotation in et_sync_annotations
                if float(annotation["onset"]) < last_start_recording
            ]
            et_before_counts = Counter(
                str(annotation["description"])
                for annotation in et_matches_before_last_start
            )
            logger.info(**gen_log_kwargs(
                message=(
                    f"ET contains {len(et_start_recording_onsets)} 'start_eye_recording' markers. "
                    f"Sync matches before the last marker ({last_start_recording:.3f}s): "
                    f"{len(et_matches_before_last_start)}; at/after it: "
                    f"{len(et_sync_annotations) - len(et_matches_before_last_start)}."
                )
            ))
            if et_matches_before_last_start:
                logger.info(**gen_log_kwargs(
                    message=(
                        f"Potential pre-task ET sync matches before last start marker: "
                        f"{dict(et_before_counts)}"
                    )
                ))
    #"""

    sync_strategies: list[dict[str, object]] = []
    if len(sync_indices) == len(et_sync_indices) and len(sync_indices) > 1:
        sync_strategies.append(dict(
            level=0,
            name="full matched sync list",
            sync_indices=list(sync_indices),
            et_sync_indices=list(et_sync_indices),
            edge_trimmed=0,
        ))
    elif len(sync_indices) == len(et_sync_indices):
        logger.info(**gen_log_kwargs(
            message=(
                f"Only {len(sync_indices)} shared sync event(s) were available in the "
                "full matched lists; skipping direct realignment attempt."
            )
        ))

    trimmed_sync_indices, trimmed_et_sync_indices, trimmed_metric = (
        _maybe_trim_mismatched_sync_edges(
            sync_annotations=sync_annotations,
            et_sync_annotations=et_sync_annotations,
            sync_indices=sync_indices,
            et_sync_indices=et_sync_indices,
        )
    )
    if trimmed_metric != 0:
        sync_strategies.append(dict(
            level=1,
            name="edge-trimmed sync list",
            sync_indices=trimmed_sync_indices,
            et_sync_indices=trimmed_et_sync_indices,
            edge_trimmed=int(trimmed_metric),
        ))

    first_last_shared = _find_first_last_shared_sync_indices(
        sync_annotations=sync_annotations,
        et_sync_annotations=et_sync_annotations,
        sync_indices=sync_indices,
        et_sync_indices=et_sync_indices,
    )
    if first_last_shared is not None:
        sync_strategies.append(dict(
            level=2,
            name="first and last shared sync events",
            sync_indices=first_last_shared[0],
            et_sync_indices=first_last_shared[1],
            edge_trimmed=0,
        ))

    selected_sync_times_report: list[float] | None = None
    selected_et_sync_times_report: list[float] | None = None
    sync_times_for_realign: list[float] | None = None
    et_sync_times_for_realign: list[float] | None = None

    # Check whether the eye-tracking data contains nan values. If yes replace them with zeros.
    if np.isnan(raw_et._data).any():

        # Set all nan values in the eye-tracking data to 0 (to make resampling possible)
        # TODO: Decide whether this is a good approach or whether interpolation (e.g. of blinks) is useful
        # TODO: Decide about setting the values (e.g. for blinks) back to nan after synchronising the signals
        # TODO: Tip: With `mne.preprocessing.annotate_nan` you could get the timings comparatively easy, and then after `realign_raw` put nans on top.
        np.nan_to_num(raw_et._data, copy=False, nan=0.0)
        logger.info(**gen_log_kwargs(message=f"The eye-tracking data contained nan values. They were replaced with zeros."))

    _interpolate_zero_invalid_et_channels(raw_et)

    # realign_raw behaves unexpectedly if no meas_date is set
    if raw.info["meas_date"] is None:
        raw.set_meas_date(946684800) # use Jan 1st 2000 as dummy (default anonymized meas_date)
    raw_et.set_meas_date(raw.info["meas_date"])

    et_pre_n, et_pre_f   = raw_et.n_times, float(raw_et.info["sfreq"])
    eeg_pre_n, eeg_pre_f = raw.n_times, float(raw.info["sfreq"])

    for strategy in sync_strategies:
        strategy_sync_indices = list(strategy["sync_indices"])
        strategy_et_sync_indices = list(strategy["et_sync_indices"])
        strategy_sync_times = [
            float(raw.annotations[idx]["onset"]) for idx in strategy_sync_indices
        ]
        strategy_et_sync_times = [
            float(raw_et.annotations[idx]["onset"]) for idx in strategy_et_sync_indices
        ]
        if len(strategy_sync_times) < 2 or len(strategy_et_sync_times) < 2:
            continue

        sync_times_trial, et_sync_times_trial, added_synthetic_pair = (
            _augment_two_point_sync_pair(
                sync_times=strategy_sync_times,
                et_sync_times=strategy_et_sync_times,
                raw=raw,
                raw_et=raw_et,
            )
        )
        if added_synthetic_pair:
            logger.info(**gen_log_kwargs(
                message=(
                    f"Sync strategy level {strategy['level']} ({strategy['name']}) "
                    "uses 2 anchor events; added one synthetic sync event pair to "
                    "stabilize realign_raw."
                )
            ))

        if not _sync_times_pass_realign_checks(
            sync_times=sync_times_trial,
            et_sync_times=et_sync_times_trial,
        ):
            logger.info(**gen_log_kwargs(
                message=(
                    f"Skipping sync strategy level {strategy['level']} "
                    f"({strategy['name']}) because its anchor events failed the "
                    "linear-correlation safety check."
                )
            ))
            continue

        try:
            mne.preprocessing.realign_raw(
                raw, raw_et, sync_times_trial, et_sync_times_trial
            )
        except Exception as err:
            logger.info(**gen_log_kwargs(
                message=(
                    f"Sync strategy level {strategy['level']} ({strategy['name']}) "
                    f"failed during realignment: {err}"
                )
            ))
            continue

        selected_sync_times_report = strategy_sync_times
        selected_et_sync_times_report = strategy_et_sync_times
        sync_times_for_realign = sync_times_trial
        et_sync_times_for_realign = et_sync_times_trial
        metrics["sync_events_edge_trimmed"] = int(strategy["edge_trimmed"])
        metrics["sync_failsafe_level"] = int(strategy["level"])
        logger.info(**gen_log_kwargs(
            message=(
                f"Using sync strategy level {strategy['level']} "
                f"({strategy['name']})."
            )
        ))
        break

    if selected_sync_times_report is None or selected_et_sync_times_report is None:
        selected_sync_times_report, selected_et_sync_times_report = (
            _force_align_recording_starts(raw=raw, raw_et=raw_et)
        )
        sync_times_for_realign = list(selected_sync_times_report)
        et_sync_times_for_realign = list(selected_et_sync_times_report)
        metrics["sync_events_edge_trimmed"] = 0
        metrics["sync_failsafe_level"] = 3
        logger.info(**gen_log_kwargs(
            message="Using sync strategy level 3 (force align to recording start)."
        ))

    assert len(selected_sync_times_report) > 1, (
        f"Not enough distinct sync events for realignment "
        f"({len(selected_sync_times_report)})"
    )

    # Sort annotations by onset
    #sorted_annotations = sorted(raw_et.annotations, key=lambda x: x["onset"])
    #for annot in sorted_annotations[:10]:
    #    print(f"onset: {annot['onset']}, duration: {annot['duration']}, description: {annot['description']}")

    _round_step_like_et_channels(raw_et)


    # Sort annotations by onset
    #sorted_annotations = sorted(raw_et.annotations, key=lambda x: x["onset"])
    #for annot in sorted_annotations[:10]:
    #    print(f"onset: {annot['onset']}, duration: {annot['duration']}, description: {annot['description']}")


    metrics["et_samples_trimmed"]  = max(0, int(round(et_pre_n  - raw_et.n_times * (et_pre_f  / float(raw_et.info["sfreq"])))))
    metrics["eeg_samples_trimmed"] = max(0, int(round(eeg_pre_n - raw.n_times    * (eeg_pre_f / float(raw.info["sfreq"])))))
    raw_et.rename_channels(ascii_sanitize)

    # Add ET data to EEG
    raw.add_channels([raw_et], force_update_info=True)

    # Also add ET annotations to EEG
    # first mark et sync event descriptions so we can differentiate them later
    # TODO: For now all ET events will be marked with ET and added to the EEG annotations, maybe later filter for certain events only
    raw_et.annotations.description = np.array(list(map(lambda desc: "ET_" + desc, raw_et.annotations.description)))
    
    #avoid calling internal function _combine_annotations
    #raw.set_annotations(mne.annotations._combine_annotations(raw.annotations,
    #                                                            raw_et.annotations,
    #                                                            0,
    #                                                            raw.first_samp,
    #                                                            raw_et.first_samp,
    #                                                            raw.info["sfreq"]))
    
    shift = (raw.first_samp - raw_et.first_samp) / raw.info["sfreq"]

    et_shifted = mne.Annotations(
        onset=raw_et.annotations.onset + shift, # shift ET annotations to match EEG
        orig_time=raw.annotations.orig_time, # match orig_time to raw EEG
        duration=raw_et.annotations.duration,
        description=raw_et.annotations.description,
        ch_names=raw_et.annotations.ch_names,
        # extras for mne>=1.11, for older versions this attribute is skipped
        **({"extras": getattr(raw_et.annotations, "extras", None)} if hasattr(raw_et.annotations, "extras") else {})
    )
    raw.set_annotations(raw.annotations + et_shifted)
    
    msg = f"Saving synced data to disk."
    logger.info(**gen_log_kwargs(message=msg))
    raw.save(
        out_files["eyelink_eeg"],
        overwrite=True,
        split_naming="bids", # TODO: Find out if we need to add this or not
        split_size=cfg._raw_split_size, # ???
    )
    # no idea what the split stuff is...
    _update_for_splits(out_files, "eyelink_eeg") # TODO: Find out if we need to add this or not

    # Extract and concatenate eye-tracking event data frames
    et_dfs = raw_et._raw_extras[0]["dfs"]
    df_list = [] # List to collect extracted data frames before concatenation

    # Extract fixations, saccades and blinks data frames
    for df_name, trial_type in zip(["fixations", "saccades", "blinks"], ["fixation", "saccade", "blink"]):
        df = et_dfs[df_name]
        df["trial_type"] = trial_type
        df_list.append(df)

    et_combined_df = pd.concat(df_list, ignore_index=True)
    et_combined_df.rename(columns={"time":"onset"}, inplace=True)
    et_combined_df.sort_values(by="onset", inplace=True, ignore_index=True)
    et_combined_df = et_combined_df[ # Adapt column order
        [
            "onset", # needs to be first (BIDS convention)
            "duration",
            "end_time",
            "trial_type",
            "eye",
            "fix_avg_x",
            "fix_avg_y",
            "fix_avg_pupil_size",
            "sacc_start_x",
            "sacc_start_y",
            "sacc_end_x",
            "sacc_end_y",
            "sacc_visual_angle",
            "peak_velocity"
        ]
    ] 

    # Synchronize eye-tracking events with EEG data

    # Recalculate regression coefficients (because the realign_raw function does not output them)
    # Code snippet from `mne.preprocessing.realign_raw` function:
    # https://github.com/mne-tools/mne-python/blob/b44c46ae7f9b6ffc5318b5d64f12906c1f2d875c/mne/preprocessing/realign.py#L69-L71
    if metrics["sync_failsafe_level"] == 3:
        zero_ord, first_ord = 0.0, 1.0
    else:
        poly = Polynomial.fit(
            x=et_sync_times_for_realign,
            y=sync_times_for_realign,
            deg=1,
        )
        converted = poly.convert(domain=(-1, 1))
        [zero_ord, first_ord] = converted.coef

    # Synchronize time stamps of ET events
    et_combined_df["onset"] = (et_combined_df["onset"] * first_ord + zero_ord)
    et_combined_df["end_time"] = (et_combined_df["end_time"] * first_ord + zero_ord)
    # TODO: To be super correct, we would need to recalculate duration column as well - but typically the slope is so close to "1" that this would typically result in <1ms differences

    msg = f"Saving synced eye-tracking events to disk."
    logger.info(**gen_log_kwargs(message=msg))
    et_combined_df.to_csv(out_files["eyelink_et_events"], sep="\t", index=False)

    # Add to report
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 19.2))
    msg = f"Adding figure to report."
    logger.info(**gen_log_kwargs(message=msg))
    tags = ("sync", "eyelink")
    title = "Synchronize Eyelink"
    report_figure_title = (
        "Eyelink data" if run is None else f"Eyelink data (run {run})"
    )
    if metrics["sync_failsafe_level"] == 3:
        caption = (
            "Recording-start alignment was used to line up the Eyelink and M/EEG "
            "files after the event-based sync strategies failed. The Eyelink data "
            "was added as annotations and appended as new channels."
        )
    else:
        caption = (
            f"The `realign_raw` function from MNE was used to align an Eyelink `asc` file to the M/EEG file."
            f"The Eyelink-data was added as annotations and appended as new channels."
        )
    if cfg.sync_heog_ch is None or cfg.sync_et_ch is None:
        # we need both an HEOG channel and ET channel specified to do cross-correlation
        msg = f"HEOG and/or ET channel not specified; cannot produce cross-correlation for report."
        logger.info(**gen_log_kwargs(message=msg))
        caption += "\nHEOG and/or eye tracking channels were not specified and no cross-correlation was performed."
        axes[0,0].text(0.5, 0.5, 'HEOG/ET cross-correlation unavailable', fontsize=34,
                       horizontalalignment='center', verticalalignment='center', transform=axes[0,0].transAxes)
        axes[0,0].axis("off")
    else:
        # return _prep_out_files(exec_params=exec_params, out_files=out_files)
        # calculate cross correlation of HEOG with ET
        heog_ch, et_ch, bipolar = _check_HEOG_ET_vars(cfg)
        if bipolar:
            # create bipolar HEOG
            raw = mne.set_bipolar_reference(raw, *cfg.sync_heog_ch, ch_name=heog_ch, drop_refs=False)
        raw.filter(l_freq=cfg.sync_heog_highpass, h_freq=cfg.sync_heog_lowpass, picks=heog_ch) # get rid of drift and high freq noise
        _mark_calibration_as_bad(raw, cfg)
        # extract HEOG and ET as arrays
        heog_array = raw.get_data(picks=[heog_ch], reject_by_annotation="omit")
        et_array = raw.get_data(picks=et_ch, reject_by_annotation="omit")
        if len(et_array) > 1:
            et_array = et_array.mean(axis=0, keepdims=True)
        # cross correlate them
        corr = correlate(heog_array[0], et_array[0], mode="same") / heog_array.shape[1]
        # plot cross correlation
        # figure out how much we plot
        midpoint = len(corr) // 2
        plot_samps = (-cfg.sync_plot_samps, cfg.sync_plot_samps) if isinstance(cfg.sync_plot_samps, int) else cfg.sync_plot_samps
        if isinstance(plot_samps, tuple):
            x_range = np.arange(plot_samps[0], plot_samps[1])
            y_range = np.arange(midpoint+plot_samps[0], midpoint+plot_samps[1])
        else: # None
            y_range = np.arange(len(corr))
            x_range = y_range - midpoint
        xcorr_plot = corr[y_range]

        # gauss fit overlay
        if cfg.sync_gauss_window is not None:
            try:
                A, mu, sigma, b = fit_gauss_to_xcorr(
                    x_range, xcorr_plot, cfg.sync_gauss_window
                )
                gauss_plot = gaussian(x_range, A, mu, sigma) + b
                axes[0, 0].plot(x_range, gauss_plot, linestyle="--", linewidth=2)
                caption += (
                    f"\nEstimated synchronisation delay (Gaussian peak) = {mu:.0f} "
                    f"samples ({mu/raw.info['sfreq']:.3f} s)."
                )
                metrics["gauss_A"] = float(A)
                metrics["gauss_mu"] = float(mu)
                metrics["gauss_sigma"] = float(sigma)
                metrics["gauss_b"] = float(b)
                denom_full = np.linalg.norm(xcorr_plot) * np.linalg.norm(gauss_plot)
                metrics["gauss_xcorr_cosine_similarity_full"] = (
                    float(np.dot(xcorr_plot, gauss_plot) / denom_full)
                    if denom_full > 0
                    else np.nan
                )
                metrics["gauss_xcorr_cosine_similarity_full_n"] = int(xcorr_plot.size)
                # Compare fit quality in the central 95% bell region (mu ± 1.96*sigma),
                # limited to the plotting window defined by sync_plot_samps.
                bell_mask = np.abs(x_range - mu) <= (1.96 * sigma)
                if np.any(bell_mask):
                    xcorr_bell = xcorr_plot[bell_mask]
                    gauss_bell = gauss_plot[bell_mask]
                    denom = np.linalg.norm(xcorr_bell) * np.linalg.norm(gauss_bell)
                    metrics["gauss_xcorr_cosine_similarity_95"] = (
                        float(np.dot(xcorr_bell, gauss_bell) / denom)
                        if denom > 0
                        else np.nan
                    )
                    metrics["gauss_xcorr_cosine_similarity_95_n"] = int(
                        np.count_nonzero(bell_mask)
                    )
                else:
                    metrics["gauss_xcorr_cosine_similarity_95"] = np.nan
                    metrics["gauss_xcorr_cosine_similarity_95_n"] = 0
            except Exception as err:
                msg = (
                    "Gaussian fit for HEOG/ET cross-correlation failed. "
                    "Skipping Gaussian overlay and setting Gaussian metrics to -1. "
                    f"Error: {err}"
                )
                logger.info(**gen_log_kwargs(message=msg))
                metrics["gauss_A"] = -1.0
                metrics["gauss_mu"] = -1.0
                metrics["gauss_sigma"] = -1.0
                metrics["gauss_b"] = -1.0
                metrics["gauss_xcorr_cosine_similarity_full"] = -1.0
                metrics["gauss_xcorr_cosine_similarity_full_n"] = -1
                metrics["gauss_xcorr_cosine_similarity_95"] = -1.0
                metrics["gauss_xcorr_cosine_similarity_95_n"] = -1

        # plot
        axes[0,0].plot(x_range, xcorr_plot, color="black")
        axes[0,0].axvline(linestyle="--", alpha=0.3)
        axes[0,0].set_title("Cross correlation HEOG and ET")
        axes[0,0].set_xlabel("Samples")
        axes[0,0].set_ylabel("X correlation")
        # calculate delay
        delay_idx = abs(corr).argmax() - midpoint
        delay_time = delay_idx * (raw.times[1] - raw.times[0])
        caption += f"\nThere was an estimated synchronisation delay of {delay_idx} samples ({delay_time:.3f} seconds.)"



        
        idx_all = find_peaks(corr)[0]
        order = np.argsort(corr[idx_all])[::-1][:10]
        idx = idx_all[order]
        # save xcorr as artifact, for later analysis if needed
        artifact = {
            "xc_plot": xcorr_plot.astype(np.float16),
            "heog_shape": heog_array.shape[1],
            "xc_peak_indexes": idx.tolist(),
            "xc_peak_heights": corr[idx].tolist(),
            "xc_peak_prominences": peak_prominences(corr, idx)[0].tolist(),
        }

        # save in deriv folder of subject
        deriv_root = cfg.deriv_root / f"sub-{subject}" / "eeg"
        deriv_root.mkdir(parents=True, exist_ok=True)
        artifact_basename = f"sub-{subject}_sync-eyelink_task-{cfg.task}"
        if run is not None:
            artifact_basename += f"_run-{run}"
        artifact_fname = deriv_root / f"{artifact_basename}_xcorr-artifact.npz"
        np.savez_compressed(artifact_fname, **artifact)



        # Record metrics from the same xcorr array used for plotting above.
        metrics["xcorr_zero"] = float(corr[midpoint])
        # TODO maybe adjust heights, prominence
        peaks, props = find_peaks(corr, height=0, prominence=0)
        order = np.argsort(corr[peaks])[::-1]  # sort peaks by height, desc
        metrics["snr"] = np.nan

        if isinstance(plot_samps, tuple):
            limit_left, limit_right = int(plot_samps[0]), int(plot_samps[1])
        else:
            half_window = int(cfg.sync_gauss_window) if cfg.sync_gauss_window else len(corr) // 4
            limit_left, limit_right = -half_window, half_window

        left_edge = int(np.clip(midpoint + limit_left, 0, len(corr)))
        right_edge = int(np.clip(midpoint + limit_right, 0, len(corr)))
        if left_edge > right_edge:
            left_edge, right_edge = right_edge, left_edge

        noise_parts = []
        if left_edge > 0:
            noise_parts.append(corr[:left_edge])
        if right_edge < len(corr):
            noise_parts.append(corr[right_edge:])
        noise_std = np.std(np.concatenate(noise_parts)) if noise_parts else 0.0

        if peaks.size > 0:
            j1 = peaks[order[0]]
            metrics["xcorr_peak"] = float(corr[j1])
            metrics["xcorr_peak_idx"] = int(j1 - midpoint)  # lag in samples
            metrics["xcorr_peak_prominence"] = float(props["prominences"][order[0]])
            if noise_std > 0:
                metrics["snr"] = float(corr[j1] / noise_std)
        if peaks.size > 1:
            j2 = peaks[order[1]]
            metrics["xcorr_second_peak"] = float(corr[j2])
            metrics["xcorr_second_peak_idx"] = int(j2 - midpoint)  # lag in samples
            metrics["xcorr_second_peak_prominence"] = float(props["prominences"][order[1]])





    # regression between synced events
    raw_onsets = np.asarray(selected_sync_times_report, dtype=float)
    et_onsets = (
        np.asarray(selected_et_sync_times_report, dtype=float) * first_ord + zero_ord
    )

    if len(raw_onsets) != len(et_onsets):
        raise ValueError(f"Lengths of raw {len(raw_onsets)} and ET {len(et_onsets)} onsets do not match.")
    
    metrics["shared_events"] = len(raw_onsets)
    if len(raw_onsets) > 1:
        sorted_raw_onsets = np.sort(np.asarray(raw_onsets, dtype=float))
        metrics["longest_gap_between_events"] = float(np.max(np.diff(sorted_raw_onsets)))
    else:
        metrics["longest_gap_between_events"] = np.nan

    # regress and plot
    coef = np.polyfit(raw_onsets, et_onsets, 1)
    preds = np.poly1d(coef)(raw_onsets)
    resids = et_onsets - preds
    axes[0,1].plot(raw_onsets, et_onsets, "o", alpha=0.3, color="black")
    axes[0,1].plot(raw_onsets, preds, "--k")
    axes[0,1].set_title("Regression")
    axes[0,1].set_xlabel("Raw onsets (seconds)")
    axes[0,1].set_ylabel("ET onsets (seconds)")
    # residuals
    axes[1,0].plot(np.arange(len(resids)), resids, "o", alpha=0.3, color="black")
    axes[1,0].axhline(linestyle="--")
    axes[1,0].set_title("Residuals")
    axes[1,0].set_ylabel("Residual (seconds)")
    axes[1,0].set_xlabel("Samples")
    # histogram of distances between events in time
    axes[1,1].hist(np.array(raw_onsets) - np.array(et_onsets), bins=11, range=(-5,5), color="black")
    axes[1,1].set_title("Raw - ET event onset distances histogram")
    axes[1,1].set_xlabel("milliseconds")
    # this doesn't seem to help, though it should...
    fig.tight_layout()

    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        run=run,
        task=cfg.task,
    ) as report:
        caption = caption
        report.add_figure(
            fig=fig,
            title=report_figure_title,
            section=title,
            caption=caption,
            tags=tags[1],
            replace=True,
        )
        plt.close(fig)
        del caption



    metrics["mean_abs_sync_error_ms"] = (float(np.mean(np.abs(resids))) * 1000.0 if resids.size > 0 else np.nan)
    metrics["median_abs_sync_error_ms"] = (float(np.median(np.abs(resids))) * 1000.0 if resids.size > 0 else np.nan)

    if len(raw_onsets):
        _diff_samples_abs = np.abs((np.array(raw_onsets) - np.array(et_onsets)) * float(raw.info["sfreq"]))
        metrics["within_1_sample"] = int(np.sum(_diff_samples_abs <= 1.0))
        metrics["within_4_samples"] = int(np.sum(_diff_samples_abs <= 4.0))

    # regression + correlation across all events
    if len(raw_onsets):
        _coef_all = np.polyfit(raw_onsets, et_onsets, 1)
        metrics["regression_slope"] = float(_coef_all[0])
        metrics["regression_intercept"] = float(_coef_all[1])


    #print(raw_et.annotations["description"])
    # Saccade stats
    if getattr(raw_et, "annotations", None) is not None:

        _is_saccade = np.array(["Saccade" in str(desc) for desc in raw_et.annotations.description], dtype=bool)
        _is_fixation = np.array(["Fixation" in str(desc) for desc in raw_et.annotations.description], dtype=bool)
        _is_blink = np.array(["Blink" in str(desc) for desc in raw_et.annotations.description], dtype=bool)
        metrics["n_saccades"] = int(_is_saccade.sum())
        metrics["n_blinks"] = int(_is_blink.sum())

        metrics["avg_saccade_duration_ms"] = float(np.mean(np.asarray(raw_et.annotations.duration, float)[_is_saccade]) * 1000.0)
        metrics["avg_fixation_duration_ms"] = float(np.mean(np.asarray(raw_et.annotations.duration, float)[_is_fixation]) * 1000.0)
        metrics["avg_blink_duration_ms"] = float(np.mean(np.asarray(raw_et.annotations.duration, float)[_is_blink]) * 1000.0)
        
        #metrics["avg_saccade_amplitude"] = float(np.mean(np.asarray(raw_et.annotations.extra_5, float)[_is_saccade]))

        anns = raw_et.annotations.to_data_frame()
        filtered = anns.loc[pd.Series(_is_saccade, index=anns.index)]
        try:
            metrics["avg_saccade_amplitude"] = filtered["Amplitude"].mean()
        except Exception as e:
            print("Error computing avg_saccade_amplitude:", e)

    ###metrics["snr"] = _compute_snr(raw, power_line_freq=60.0)

    #n_saccades=np.nan,
    #avg_saccade_amplitude=np.nan,
    #avg_saccade_duration_ms=np.nan,
    #avg_fixation_duration_ms=np.nan,

    #n_blinks=np.nan,
    #avg_blink_duration_ms=np.nan
    metrics_bids = raw_fname.copy().update(
        run=run, split=None,
        processing="eyelink",
        suffix="metrics",
        extension=".json",
    )
    metrics_fname = out_dir_misc / metrics_bids.basename

    metrics_fname.parent.mkdir(parents=True, exist_ok=True)

    def _json_default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(metrics_fname, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)



    return _prep_out_files(exec_params=exec_params, out_files=out_files)



def get_config(
   *,
    config: SimpleNamespace,
    subject: str,
    session: str | None = None,
) -> SimpleNamespace:
    #logger.info(**gen_log_kwargs(message=f"config {config}"))

    cfg = SimpleNamespace(
        runs=get_runs(config=config, subject=subject),
        remove_blink_saccades   = config.remove_blink_saccades,
        et_has_run = config.et_has_run,
        et_has_task = config.et_has_task,
        sync_eventtype_regex    = config.sync_eventtype_regex,
        sync_eventtype_regex_et = config.sync_eventtype_regex_et,
        sync_heog_ch = config.sync_heog_ch,
        sync_et_ch = config.sync_et_ch,
        sync_heog_highpass = config.sync_heog_highpass,
        sync_heog_lowpass = config.sync_heog_lowpass,
        sync_plot_samps = config.sync_plot_samps,
        sync_gauss_window = config.sync_gauss_window,
        sync_calibration_string = config.sync_calibration_string,
        eeg_bipolar_channels = config.eeg_bipolar_channels,
        processing= "filt" if config.regress_artifact is None else "regress",
        _raw_split_size=config._raw_split_size,

        **_bids_kwargs(config=config),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Sync Eyelink."""
    if not config.sync_eyelink:
        msg = "Skipping, sync_eyelink is set to False …"
        logger.info(**gen_log_kwargs(message=msg, emoji="skip"))
        return

    ssrt = _get_ssrt(config=config)
    report_exec_params = get_serial_report_exec_params(
        exec_params=config.exec_params,
        step="Syncing Eyelink",
    )
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            sync_eyelink,
            exec_params=report_exec_params,
            n_iter=len(_get_ss(config=config)),
        )
        logs = parallel(
            run_func(
                cfg=get_config(config=config, subject=subject),
                exec_params=report_exec_params,
                subject=subject,
                session=session,
                run=run,
                task=task,
            )
            for subject, session, run, task in ssrt
        )
    save_logs(config=config, logs=logs)
