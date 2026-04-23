"""Apply low- and high-pass filters.

The data are bandpass filtered to the frequencies defined in the config
(config.h_freq - config.l_freq Hz) using linear-phase fir filter with
delay compensation.
The transition bandwidth is automatically defined. See
`Background information on filtering
<http://mne-tools.github.io/dev/auto_tutorials/plot_background_filtering.html>`_
for more. The filtered data are saved to separate files to the subject's 'MEG'
directory.

To save space, the raw data can be resampled.

If config.interactive = True plots raw data and power spectral density.

All code relating to detect_freqs is LLM written
"""  # noqa: E501

from collections.abc import Iterable
from html import escape
from types import SimpleNamespace
from typing import Any, Literal

import mne
import numpy as np
from meegkit import dss
from mne.io.pick import _picks_to_idx
from mne.preprocessing import EOGRegression

from mne_bids_pipeline._config_utils import _get_ssrt
from mne_bids_pipeline._import_data import (
    _get_run_rest_noise_path,
    _import_data_kwargs,
    _read_raw_msg,
    import_er_data,
    import_experimental_data,
)
from mne_bids_pipeline._logging import gen_log_kwargs, logger
from mne_bids_pipeline._parallel import (
    get_parallel_backend,
    get_serial_report_exec_params,
    parallel_func,
)
from mne_bids_pipeline._report import _add_raw, _open_report
from mne_bids_pipeline._run import (
    _prep_out_files,
    _update_for_splits,
    failsafe_run,
    save_logs,
)
from mne_bids_pipeline.typing import InFilesT, IntArrayT, OutFilesT, RunKindT, RunTypeT

_DETECT_N_FFT_SECONDS = 10.0
_DETECT_SEARCH_HALF_WIDTH_HZ = 1.0
_DETECT_BASELINE_HZ = 8.0
_DETECT_BASELINE_INNER_HZ = 3.0
_DETECT_BASELINE_PERCENTILE = 25.0
_DETECT_PEAK_THRESHOLD_DB = 6.0


def get_input_fnames_frequency_filter(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
) -> InFilesT:
    """Get paths of files required by filter_data function."""
    kind: RunKindT = "sss" if cfg.use_maxwell_filter else "orig"
    return _get_run_rest_noise_path(
        cfg=cfg,
        subject=subject,
        session=session,
        run=run,
        task=task,
        kind=kind,
        mf_reference_run=cfg.mf_reference_run,
    )


def _normalize_notch_freqs(
    freqs: float | Iterable[float] | None,
) -> list[float]:
    """Normalize notch frequencies to a sorted unique float list."""
    if freqs is None:
        return []
    if np.isscalar(freqs):
        values = [float(freqs)]
    else:
        values = [float(freq) for freq in freqs]
    return sorted({freq for freq in values if freq > 0})


def _combine_notch_freqs(
    static_freqs: float | Iterable[float] | None,
    detected_freqs: Iterable[float],
) -> float | list[float] | None:
    """Combine configured and dynamically detected notch frequencies."""
    combined = _normalize_notch_freqs(static_freqs)
    combined.extend(float(freq) for freq in detected_freqs)
    combined = sorted({freq for freq in combined if freq > 0})

    if not combined:
        return None
    if len(combined) == 1:
        return combined[0]
    return combined


def _compute_channel_median_psd(
    data: np.ndarray,
    *,
    sfreq: float,
    fmin: float,
    fmax: float,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequencies and a robust PSD summary (median across channels, in dB)."""
    psds, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_fft // 2,
        average="mean",
        verbose=False,
    )
    psd_db = 10.0 * np.log10(np.maximum(psds, np.finfo(float).tiny))
    return freqs, np.median(psd_db, axis=0)


def _evaluate_candidate_peak(
    *,
    freqs: np.ndarray,
    psd_db: np.ndarray,
    target: float,
) -> dict[str, float | bool] | None:
    """Evaluate one candidate frequency on an already-computed PSD."""
    if len(freqs) == 0 or len(psd_db) == 0 or target > freqs[-1]:
        return None

    peak_mask = (freqs >= target - _DETECT_SEARCH_HALF_WIDTH_HZ) & (
        freqs <= target + _DETECT_SEARCH_HALF_WIDTH_HZ
    )
    left_base_mask = (freqs >= target - _DETECT_BASELINE_HZ) & (
        freqs <= target - _DETECT_BASELINE_INNER_HZ
    )
    right_base_mask = (freqs >= target + _DETECT_BASELINE_INNER_HZ) & (
        freqs <= target + _DETECT_BASELINE_HZ
    )

    baseline_candidates = []
    if np.sum(left_base_mask) >= 2:
        baseline_candidates.append(
            np.percentile(psd_db[left_base_mask], _DETECT_BASELINE_PERCENTILE)
        )
    if np.sum(right_base_mask) >= 2:
        baseline_candidates.append(
            np.percentile(psd_db[right_base_mask], _DETECT_BASELINE_PERCENTILE)
        )

    if not np.any(peak_mask) or not baseline_candidates:
        return None

    peak_local_idx = np.argmax(psd_db[peak_mask])
    peak_global_idx = np.where(peak_mask)[0][peak_local_idx]
    peak_freq = float(freqs[peak_global_idx])
    peak_db = float(psd_db[peak_global_idx])
    baseline_db = float(np.median(baseline_candidates))
    prominence_db = peak_db - baseline_db

    return {
        "detected_freq": peak_freq,
        "prominence_db": prominence_db,
        "detected": bool(prominence_db >= _DETECT_PEAK_THRESHOLD_DB),
    }


def _evaluate_candidate_frequencies(
    raw: mne.io.BaseRaw,
    *,
    cfg: SimpleNamespace,
) -> list[dict[str, float | bool | None]]:
    """Evaluate all candidate frequencies on the current raw data."""
    candidate_freqs = _normalize_notch_freqs(getattr(cfg, "detect_freqs", None))
    if not candidate_freqs:
        return []

    picks = mne.pick_types(raw.info, meg=True, eeg=True)
    if len(picks) == 0:
        return []

    data = raw.get_data(picks=picks)
    sfreq = raw.info["sfreq"]
    n_fft = int(round(_DETECT_N_FFT_SECONDS * sfreq))
    n_fft = max(256, min(n_fft, data.shape[1]))

    freqs, psd_db = _compute_channel_median_psd(
        data=data,
        sfreq=sfreq,
        fmin=max(0.0, min(candidate_freqs) - _DETECT_BASELINE_HZ - 1.0),
        fmax=min(sfreq / 2.0 - 0.5, max(candidate_freqs) + _DETECT_BASELINE_HZ + 1.0),
        n_fft=n_fft,
    )

    evaluations = []
    for target in candidate_freqs:
        evaluation = _evaluate_candidate_peak(freqs=freqs, psd_db=psd_db, target=target)
        row: dict[str, float | bool | None] = {
            "candidate_freq": float(target),
            "detected_freq": None,
            "prominence_db": None,
            "detected": False,
        }
        if evaluation is not None:
            row.update(evaluation)
        evaluations.append(row)

    return evaluations


def _log_candidate_frequency_summary(
    evaluations: list[dict[str, float | bool | None]],
    *,
    residual: bool,
) -> None:
    """Log summary information for candidate frequencies."""
    summaries = []
    detected = []
    for evaluation in evaluations:
        candidate_freq = float(evaluation["candidate_freq"])
        detected_freq = evaluation["detected_freq"]
        prominence_db = evaluation["prominence_db"]
        is_detected = bool(evaluation["detected"])

        if detected_freq is None or prominence_db is None:
            summaries.append(f"{candidate_freq:.2f} Hz (not evaluable)")
            continue

        summaries.append(
            f"{candidate_freq:.2f}->{float(detected_freq):.2f} Hz "
            f"(prominence {float(prominence_db):.2f} dB, "
            f"{'detected' if is_detected else 'below threshold'})"
        )
        if is_detected:
            detected.append(float(detected_freq))

    if summaries:
        prefix = "Residual adaptive frequency summary" if residual else (
            "Adaptive notch candidate summary"
        )
        msg = prefix + ": " + "; ".join(summaries)
        logger.info(**gen_log_kwargs(message=msg))

    if residual and detected:
        msg = (
            "Candidate peaks still above threshold after notch filtering at "
            f"{detected} Hz."
        )
    elif residual:
        msg = (
            "No candidate peaks remained above the detection threshold after "
            "notch filtering."
        )
    elif detected:
        msg = f"Applying notch filter at dynamically detected frequencies {detected} Hz."
    else:
        msg = (
            "No candidate peaks exceeded the detection threshold; not adding "
            "dynamic notch frequencies."
        )
    logger.info(**gen_log_kwargs(message=msg))


def _detect_notch_frequencies(
    raw: mne.io.BaseRaw,
    *,
    cfg: SimpleNamespace,
) -> tuple[list[float], list[dict[str, float | bool | None]]]:
    """Detect subject/run-specific notch frequencies from a constrained list."""
    candidate_freqs = _normalize_notch_freqs(getattr(cfg, "detect_freqs", None))
    if not candidate_freqs:
        return [], []

    msg = (
        "Detecting run-specific notch frequencies from candidate frequencies "
        f"{candidate_freqs} Hz."
    )
    logger.info(**gen_log_kwargs(message=msg))

    evaluations = _evaluate_candidate_frequencies(raw, cfg=cfg)
    _log_candidate_frequency_summary(evaluations, residual=False)

    detected = [
        float(evaluation["detected_freq"])
        for evaluation in evaluations
        if evaluation["detected"] and evaluation["detected_freq"] is not None
    ]
    return sorted({freq for freq in detected if freq > 0}), evaluations


def _format_freq(freq: float | None) -> str:
    """Format a frequency for display."""
    if freq is None:
        return "n/a"
    return f"{float(freq):.2f}"


def _format_prominence(prominence_db: float | None) -> str:
    """Format a prominence value for display."""
    if prominence_db is None:
        return "n/a"
    return f"{float(prominence_db):.2f}"


def _format_freq_list(freqs: float | Iterable[float] | None) -> str:
    """Format a frequency or list of frequencies for display."""
    normalized = _normalize_notch_freqs(freqs)
    if not normalized:
        return "none"
    return ", ".join(f"{freq:.2f}" for freq in normalized) + " Hz"


def _matching_notch_freqs(
    *,
    candidate_freq: float,
    applied_notch_freqs: Iterable[float],
) -> list[float]:
    """Return applied notch frequencies matching a candidate window."""
    return [
        float(freq)
        for freq in applied_notch_freqs
        if abs(float(freq) - candidate_freq) <= _DETECT_SEARCH_HALF_WIDTH_HZ
    ]


def _adaptive_frequency_qc_title(*, run: str | None, task: str | None) -> str:
    """Return a report title for adaptive frequency QC."""
    if run is not None:
        run_label = str(run)
        if not run_label.startswith("run-"):
            run_label = f"run-{run_label}"
        return f"Adaptive Frequency QC, {run_label}"
    if task is not None:
        return f"Adaptive Frequency QC, task-{task}"
    return "Adaptive Frequency QC"


def _adaptive_frequency_qc_tags(*, run: str | None, task: str | None) -> tuple[str, ...]:
    """Return report tags for adaptive frequency QC."""
    if run is not None:
        run_tag = str(run)
        if not run_tag.startswith("run-"):
            run_tag = f"run-{run_tag}"
        return ("filtered", "adaptive-frequency-qc", run_tag)
    if task is not None:
        return ("filtered", "adaptive-frequency-qc", f"task-{task}")
    return ("filtered", "adaptive-frequency-qc")


def _adaptive_frequency_qc_html(
    *,
    pre_evaluations: list[dict[str, float | bool | None]],
    post_evaluations: list[dict[str, float | bool | None]],
    static_notch_freqs: float | Iterable[float] | None,
    applied_notch_freqs: float | Iterable[float] | None,
) -> str:
    """Build HTML for the adaptive frequency QC table."""
    post_by_candidate = {
        float(evaluation["candidate_freq"]): evaluation for evaluation in post_evaluations
    }
    applied_notch_freqs_list = _normalize_notch_freqs(applied_notch_freqs)
    detected_freqs = [
        float(evaluation["detected_freq"])
        for evaluation in pre_evaluations
        if evaluation["detected"] and evaluation["detected_freq"] is not None
    ]

    rows = []
    for pre_evaluation in pre_evaluations:
        candidate_freq = float(pre_evaluation["candidate_freq"])
        post_evaluation = post_by_candidate.get(candidate_freq, {})
        matched_notches = _matching_notch_freqs(
            candidate_freq=candidate_freq,
            applied_notch_freqs=applied_notch_freqs_list,
        )
        rows.append(
            (
                f"{candidate_freq:.2f}",
                _format_freq(pre_evaluation["detected_freq"]),
                _format_prominence(pre_evaluation["prominence_db"]),
                "yes" if pre_evaluation["detected"] else "no",
                _format_freq_list(matched_notches),
                _format_freq(post_evaluation.get("detected_freq")),
                _format_prominence(post_evaluation.get("prominence_db")),
                "yes" if post_evaluation.get("detected", False) else "no",
            )
        )

    header_cells = (
        "Candidate (Hz)",
        "Peak Before (Hz)",
        "Prominence Before (dB)",
        "Detected",
        "Applied Notch (Hz)",
        "Peak After (Hz)",
        "Residual Prominence (dB)",
        "Residual Above Threshold",
    )
    header_html = "".join(
        f"<th>{escape(header_cell)}</th>" for header_cell in header_cells
    )
    body_html = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )

    summary_html = (
        '<p class="mb-0">'
        f"Configured notch frequencies: {escape(_format_freq_list(static_notch_freqs))}."
        "</p>\n"
        '<p class="mb-0">'
        f"Dynamically detected frequencies: {escape(_format_freq_list(detected_freqs))}."
        "</p>\n"
        '<p class="mb-0">'
        f"Applied notch frequencies: {escape(_format_freq_list(applied_notch_freqs))}. "
        f"Residual prominence is measured after notch filtering and before band-pass filtering. "
        f"Detection threshold: {_DETECT_PEAK_THRESHOLD_DB:.2f} dB."
        "</p>\n"
    )
    table_html = (
        '<table class="table table-striped table-sm mb-0">\n'
        f"<thead><tr>{header_html}</tr></thead>\n"
        f"<tbody>{body_html}</tbody>\n"
        "</table>"
    )
    return summary_html + table_html


def zapline(
    raw: mne.io.BaseRaw,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
    fline: float | None,
    iter_: bool,
) -> None:
    """Use Zapline to remove line frequencies."""
    if fline is None:
        return

    msg = f"Zapline filtering data at with {fline=} Hz."
    logger.info(**gen_log_kwargs(message=msg))
    sfreq = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, meg=True, eeg=True)
    data = raw.get_data(picks).T  # transpose to (n_samples, n_channels)
    func = dss.dss_line_iter if iter_ else dss.dss_line
    out, _ = func(data, fline, sfreq)
    raw._data[picks] = out.T


def notch_filter(
    raw: mne.io.BaseRaw,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
    freqs: float | Iterable[float] | None,
    trans_bandwidth: float | Literal["auto"],
    notch_widths: float | Iterable[float] | None,
    run_type: RunTypeT,
    picks: IntArrayT | None,
    notch_extra_kws: dict[str, Any],
) -> None:
    """Filter data channels (MEG and EEG)."""
    if freqs is None and (notch_extra_kws.get("method") != "spectrum_fit"):
        msg = f"Not applying notch filter to {run_type} data."
    elif notch_extra_kws.get("method") == "spectrum_fit":
        msg = f"Applying notch filter to {run_type} data with spectrum fitting."
    else:
        msg = f"Notch filtering {run_type} data at {freqs} Hz."

    logger.info(**gen_log_kwargs(message=msg))

    if (freqs is None) and (notch_extra_kws.get("method") != "spectrum_fit"):
        return

    raw.notch_filter(
        freqs=freqs,
        trans_bandwidth=trans_bandwidth,
        notch_widths=notch_widths,
        n_jobs=1,
        picks=picks,
        **notch_extra_kws,
    )


def bandpass_filter(
    raw: mne.io.BaseRaw,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
    l_freq: float | None,
    h_freq: float | None,
    l_trans_bandwidth: float | Literal["auto"],
    h_trans_bandwidth: float | Literal["auto"],
    run_type: RunTypeT,
    picks: IntArrayT | None,
    bandpass_extra_kws: dict[str, Any],
) -> None:
    """Filter data channels (MEG and EEG)."""
    if l_freq is not None and h_freq is None:
        msg = f"High-pass filtering {run_type} data; lower bound: {l_freq} Hz"
    elif l_freq is None and h_freq is not None:
        msg = f"Low-pass filtering {run_type} data; upper bound: {h_freq} Hz"
    elif l_freq is not None and h_freq is not None:
        msg = f"Band-pass filtering {run_type} data; range: {l_freq} – {h_freq} Hz"
    else:
        msg = f"Not applying frequency filtering to {run_type} data."

    logger.info(**gen_log_kwargs(message=msg))

    if l_freq is None and h_freq is None:
        return

    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        n_jobs=1,
        picks=picks,
        **bandpass_extra_kws,
    )


def resample(
    raw: mne.io.BaseRaw,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
    sfreq: float,
    run_type: RunTypeT,
) -> None:
    if not sfreq:
        return

    msg = f"Resampling {run_type} data to {sfreq:.1f} Hz"
    logger.info(**gen_log_kwargs(message=msg))
    raw.resample(sfreq, npad="auto")


@failsafe_run(
    get_input_fnames=get_input_fnames_frequency_filter,
)
def filter_data(
    *,
    cfg: SimpleNamespace,
    exec_params: SimpleNamespace,
    subject: str,
    session: str | None,
    run: str,
    task: str | None,
    in_files: InFilesT,
) -> OutFilesT:
    """Filter data from a single subject."""
    out_files = dict()
    in_key = f"raw_task-{task}_run-{run}"
    bids_path_in = in_files.pop(in_key)
    bids_path_bads_in = in_files.pop(f"{in_key}-bads", None)
    msg, run_type = _read_raw_msg(bids_path_in=bids_path_in, run=run, task=task)
    logger.info(**gen_log_kwargs(message=msg))
    if cfg.use_maxwell_filter:
        raw = mne.io.read_raw_fif(bids_path_in)
    elif run is None and task == "noise":
        raw = import_er_data(
            cfg=cfg,
            bids_path_er_in=bids_path_in,
            bids_path_ref_in=in_files.pop("raw_ref_run", None),
            bids_path_er_bads_in=bids_path_bads_in,
            # take bads from this run (0)
            bids_path_ref_bads_in=in_files.pop("raw_ref_run-bads", None),
            prepare_maxwell_filter=False,
        )
    else:
        data_is_rest = run is None and task == "rest"
        raw = import_experimental_data(
            cfg=cfg,
            bids_path_in=bids_path_in,
            bids_path_bads_in=bids_path_bads_in,
            data_is_rest=data_is_rest,
        )

    out_files[in_key] = bids_path_in.copy().update(
        root=cfg.deriv_root,
        subject=subject,  # save under subject's directory so all files are there
        session=session,
        processing="filt",
        extension=".fif",
        suffix="raw",
        split=None,
        task=task,
        run=run,
        check=False,
    )

    if cfg.regress_artifact is None:
        picks = None
    else:
        # Need to figure out the correct picks to use
        model = EOGRegression(**cfg.regress_artifact)
        picks_regress = _picks_to_idx(
            raw.info, model.picks, none="data", exclude=model.exclude
        )
        picks_artifact = _picks_to_idx(raw.info, model.picks_artifact)
        picks_data = _picks_to_idx(raw.info, "data", exclude=())  # raw.filter default
        picks = np.unique(np.r_[picks_regress, picks_artifact, picks_data])

    raw.load_data()

    pre_candidate_evaluations: list[dict[str, float | bool | None]] = []
    post_candidate_evaluations: list[dict[str, float | bool | None]] = []
    detected_notch_freqs: list[float] = []
    if getattr(cfg, "detect_freqs", None):
        detected_notch_freqs, pre_candidate_evaluations = _detect_notch_frequencies(
            raw, cfg=cfg
        )
        notch_freqs = _combine_notch_freqs(cfg.notch_freq, detected_notch_freqs)
    else:
        zapline(
            raw=raw,
            subject=subject,
            session=session,
            run=run,
            task=task,
            fline=cfg.zapline_fline,
            iter_=cfg.zapline_iter,
        )
        notch_freqs = cfg.notch_freq

    notch_filter(
        raw=raw,
        subject=subject,
        session=session,
        run=run,
        task=task,
        freqs=notch_freqs,
        trans_bandwidth=cfg.notch_trans_bandwidth,
        notch_widths=cfg.notch_widths,
        run_type=run_type,
        picks=picks,
        notch_extra_kws=cfg.notch_extra_kws,
    )
    if pre_candidate_evaluations:
        post_candidate_evaluations = _evaluate_candidate_frequencies(raw, cfg=cfg)
        _log_candidate_frequency_summary(post_candidate_evaluations, residual=True)
    bandpass_filter(
        raw=raw,
        subject=subject,
        session=session,
        run=run,
        task=task,
        h_freq=cfg.h_freq,
        l_freq=cfg.l_freq,
        h_trans_bandwidth=cfg.h_trans_bandwidth,
        l_trans_bandwidth=cfg.l_trans_bandwidth,
        run_type=run_type,
        picks=picks,
        bandpass_extra_kws=cfg.bandpass_extra_kws,
    )
    resample(
        raw=raw,
        subject=subject,
        session=session,
        run=run,
        task=task,
        sfreq=cfg.raw_resample_sfreq,
        run_type=run_type,
    )

    # For example, might need to create
    # derivatives/mne-bids-pipeline/sub-emptyroom/ses-20230412/meg
    out_files[in_key].fpath.parent.mkdir(exist_ok=True, parents=True)
    raw.save(
        out_files[in_key],
        overwrite=True,
        split_naming="bids",
        split_size=cfg._raw_split_size,
    )
    _update_for_splits(out_files, in_key)
    fmax = 1.5 * cfg.h_freq if cfg.h_freq is not None else np.inf
    if exec_params.interactive:
        # Plot raw data and power spectral density.
        raw.plot(n_channels=50, butterfly=True)
        raw.compute_psd(fmax=fmax).plot()

    with _open_report(
        cfg=cfg,
        exec_params=exec_params,
        subject=subject,
        session=session,
        run=run,
        task=task,
    ) as report:
        if pre_candidate_evaluations:
            msg = "Adding adaptive frequency QC summary to report"
            logger.info(**gen_log_kwargs(message=msg))
            report.add_html(
                _adaptive_frequency_qc_html(
                    pre_evaluations=pre_candidate_evaluations,
                    post_evaluations=post_candidate_evaluations,
                    static_notch_freqs=cfg.notch_freq,
                    applied_notch_freqs=notch_freqs,
                ),
                title=_adaptive_frequency_qc_title(run=run, task=task),
                section="Filtering",
                tags=_adaptive_frequency_qc_tags(run=run, task=task),
                replace=True,
            )
        msg = "Adding filtered raw data to report"
        logger.info(**gen_log_kwargs(message=msg))
        _add_raw(
            cfg=cfg,
            report=report,
            bids_path_in=out_files[in_key],
            title="Raw (filtered)",
            tags=("filtered",),
            raw=raw,
        )

    assert len(in_files) == 0, in_files.keys()
    return _prep_out_files(exec_params=exec_params, out_files=out_files)


def get_config(
    *,
    config: SimpleNamespace,
    subject: str,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        notch_freq=config.notch_freq,
        zapline_fline=config.zapline_fline,
        zapline_iter=config.zapline_iter,
        detect_freqs=getattr(config, "detect_freqs", None),
        l_trans_bandwidth=config.l_trans_bandwidth,
        h_trans_bandwidth=config.h_trans_bandwidth,
        notch_trans_bandwidth=config.notch_trans_bandwidth,
        notch_widths=config.notch_widths,
        raw_resample_sfreq=config.raw_resample_sfreq,
        regress_artifact=config.regress_artifact,
        notch_extra_kws=config.notch_extra_kws,
        bandpass_extra_kws=config.bandpass_extra_kws,
        **_import_data_kwargs(config=config, subject=subject),
    )
    return cfg


def main(*, config: SimpleNamespace) -> None:
    """Run filter."""
    ssrt = _get_ssrt(config=config)
    report_exec_params = get_serial_report_exec_params(
        exec_params=config.exec_params,
        step="Filtering data",
    )
    with get_parallel_backend(config.exec_params):
        parallel, run_func = parallel_func(
            filter_data, exec_params=report_exec_params, n_iter=len(ssrt)
        )
        logs = parallel(
            run_func(
                cfg=get_config(
                    config=config,
                    subject=subject,
                ),
                exec_params=report_exec_params,
                subject=subject,
                session=session,
                run=run,
                task=task,
            )
            for subject, session, run, task in ssrt
        )
    save_logs(config=config, logs=logs)
