"""Read metadata inputs and construct all BIDS JSON sidecars."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import yaml

from ._column_names_helpers import canonical_sample_column
from ._samples import PreparedSampleRecording, SamplesHeader


# read user-provided metadata
def _read_yaml(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path).expanduser().resolve()
    try:
        with metadata_path.open("r", encoding="utf-8") as source:
            value = yaml.safe_load(source)
    except yaml.YAMLError as error:
        raise ValueError(
            f"{metadata_path}: cannot parse metadata YAML: {error}"
        ) from error
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{metadata_path}: metadata YAML root must be a mapping")
    return value


# there's metadata in the user-provided YAML as well as in the Samples/Events files
# the YAML takes priority, additional metadata will only be added if it's a new key
def _fill_missing(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Fill gaps: ``{A: 1}`` plus ``{A: 2, B: 3}`` -> ``{A: 1, B: 3}``."""
    result = dict(primary)
    for key, value in fallback.items():
        if value is None:
            continue
        if key not in result or result[key] is None:
            result[key] = value
            continue
        current = result[key]
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _fill_missing(current, value)
    return result


def _numbers(value: str | None) -> list[float]:
    # Pull signed integers/decimals from formatted metadata. For example,
    # "Position(400;300)" becomes [400.0, 300.0].
    if value is None:
        return []
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    return [float(item) for item in re.findall(number_pattern, value)]


def _int_if_whole(value: float) -> int | float:
    # ScreenResolution [800, 600] is easier to read than [800.0, 600.0]
    # not necessary by BIDS, but cleaner JSON output
    return int(value) if value.is_integer() else value


# Inferring metadata from the *_Samples.txt header


def _infer_from_samples_header(
    sample_header: SamplesHeader,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract physio and events.json metadata from a *_Samples.txt header."""

    # Metadata concerning filtering in the sample header
    # is written into BIDS RawDataFilters later
    sample_header_raw_data_filter_keys = (
        "Heuristic",
        "Heuristic Stage",
        "Bilateral",
        "Gaze Cursor Filter",
        "Saccade Length [px]",
        "Filter Depth [ms]",
    )

    physio: dict[str, Any] = {
        "Manufacturer": "SensoMotoric Instruments",
        "SampleCoordinateSystem": "gaze-on-screen",
    }
    bids_events: dict[str, Any] = {
        "StimulusPresentation": {"ScreenOrigin": ["top", "left"]}
    }

    versions = []
    if value := sample_header.first("Version"):
        versions.append(value)
    if value := sample_header.first("IDF Version"):
        versions.append(f"IDF {value}")
    if value := sample_header.first("iView X Version"):
        versions.append(f"iView X {value}")
    if versions:
        physio["SoftwareVersions"] = "; ".join(versions)

    system_id = sample_header.first("System ID")
    if system_id and "//" in system_id:
        serial, model = system_id.split("//", 1)
        model = model.split("///", 1)[0]
        if serial.strip():
            physio["DeviceSerialNumber"] = serial.strip()
        if model.strip():
            physio["ManufacturersModelName"] = model.strip()

    if any(
        canonical_sample_column(column).startswith(("L CR", "R CR"))
        for column in sample_header.columns
    ):
        physio["EyeTrackingMethod"] = "P-CR"

    # example calibration metadata (points 2-4 omitted)
    #    ## [Calibration]
    #    ## Calibration Area:	1440	900
    #    ## Calibration Point 0:	Position(720;450)
    #    ## Calibration Point 1:	Position(360;45)
    #    ## ...

    calibration_positions = []
    for key in sorted(sample_header.entries):
        if not key.startswith("Calibration Point "):
            continue
        values = _numbers(sample_header.first(key))
        if len(values) >= 2:
            calibration_positions.append(
                [_int_if_whole(values[0]), _int_if_whole(values[1])]
            )
    if calibration_positions:
        physio["CalibrationCount"] = 1
        physio["CalibrationPosition"] = calibration_positions
        physio["CalibrationUnit"] = "pixel"
    if value := sample_header.first("Calibration Type"):
        physio["CalibrationType"] = value

    raw_data_filter_settings = [
        f"{key}={value}"
        for key in sample_header_raw_data_filter_keys
        if (value := sample_header.first(key)) is not None
    ]
    if raw_data_filter_settings:
        physio["RawDataFilters"] = "; ".join(raw_data_filter_settings)

    resolution = _numbers(sample_header.first("Calibration Area"))
    if len(resolution) >= 2:
        bids_events["StimulusPresentation"]["ScreenResolution"] = [
            _int_if_whole(resolution[0]),
            _int_if_whole(resolution[1]),
        ]
    dimensions = _numbers(sample_header.first("Stimulus Dimension [mm]"))
    if len(dimensions) >= 2:
        bids_events["StimulusPresentation"]["ScreenSize"] = [
            dimensions[0] / 1000,
            dimensions[1] / 1000,
        ]
    distance = _numbers(sample_header.first("Head Distance [mm]"))
    if distance:
        bids_events["StimulusPresentation"]["ScreenDistance"] = distance[0] / 1000

    return physio, bids_events


def _split_yaml_metadata(
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # These fields belong in the run-level BIDS events.json rather than the
    # per-recording physio JSON files.
    bids_events_json_metadata_keys = {
        "TaskName",
        "InstitutionAddress",
        "InstitutionName",
    }
    physio: dict[str, Any] = {}
    bids_events: dict[str, Any] = {}
    for key, value in metadata.items():
        if key == "StimulusPresentation" or key in bids_events_json_metadata_keys:
            bids_events[key] = value
        else:
            physio[key] = value
    return physio, bids_events


# Constructing complete output metadata


def prepare_metadata(
    sample_header: SamplesHeader,
    metadata_file: str | Path,
    start_time: float,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    yaml_physio, yaml_bids_events = _split_yaml_metadata(_read_yaml(metadata_file))
    inferred_physio, inferred_bids_events = _infer_from_samples_header(sample_header)

    if math.isfinite(sample_header.sampling_frequency):
        inferred_physio["SamplingFrequency"] = sample_header.sampling_frequency
    physio_metadata = _fill_missing(yaml_physio, inferred_physio)
    sampling_frequency = physio_metadata.get("SamplingFrequency")
    if not isinstance(sampling_frequency, (int, float)) or not math.isfinite(
        float(sampling_frequency)
    ):
        raise ValueError(
            f"{sample_header.path}: SamplingFrequency must be supplied by metadata "
            "YAML or the SMI Sample Rate header"
        )
    sampling_frequency = float(sampling_frequency)
    physio_metadata["SamplingFrequency"] = sampling_frequency
    physio_metadata["StartTime"] = (
        start_time  # user-supplied, cannot be inferred from ET data
    )
    physio_metadata["PhysioType"] = "eyetrack"

    bids_events_metadata = _fill_missing(yaml_bids_events, inferred_bids_events)
    stimulus = bids_events_metadata.get("StimulusPresentation")
    required_display = {
        "ScreenDistance",
        "ScreenOrigin",
        "ScreenResolution",
        "ScreenSize",
    }
    if not isinstance(stimulus, dict):
        raise ValueError("StimulusPresentation metadata must be a mapping")
    missing_display = sorted(
        key for key in required_display if stimulus.get(key) is None
    )
    if missing_display:
        raise ValueError(
            "gaze-on-screen recordings require StimulusPresentation fields: "
            + ", ".join(missing_display)
        )
    return physio_metadata, bids_events_metadata, sampling_frequency


# Constructing the accompanying BIDS JSON files


def make_physio_sidecar(
    recording: PreparedSampleRecording,
    physio_metadata: dict[str, Any],
    task_name: str | None,
) -> dict[str, Any]:
    """Describe the exact columns written to one eye's ``_physio.tsv.gz``."""
    sidecar = dict(physio_metadata)
    if task_name is not None:
        sidecar["TaskName"] = task_name
    sidecar["RecordedEye"] = recording.eye
    sidecar["Columns"] = [column.name for column in recording.output_columns]
    for column in recording.output_columns:
        sidecar[column.name] = {"Description": column.description}
        if column.units is not None:
            sidecar[column.name]["Units"] = column.units
    sidecar["timestamp"]["Origin"] = "SMI iView X system startup"
    return sidecar


def _event_extra_column_metadata(source: str) -> dict[str, Any]:
    """Describe event ``End`` in us or, for example, ``Location X`` in pixels."""
    metadata: dict[str, Any] = {"Description": f"SMI event field {source!r}."}
    if source == "End":
        metadata["Units"] = "us"
    elif any(
        token in source for token in ("Location", "Loc.", "Dispersion", "Pupil Size")
    ):
        metadata["Units"] = "pixel"
    return metadata


def make_physioevents_sidecar(
    output_columns: list[str],
    extra_columns: list[tuple[str, str]],
    task_name: str | None,
) -> dict[str, Any]:
    """Describe the columns written to one eye's ``_physioevents.tsv.gz``."""
    sidecar: dict[str, Any] = {
        "Columns": output_columns,
        "Description": "Events and messages logged by the SMI eye tracker.",
        "OnsetSource": "timestamp",
        "onset": {
            "Description": "SMI device timestamp at event onset.",
            "Units": "us",
        },
        "duration": {"Description": "Event duration.", "Units": "s"},
        "trial_type": {
            "Description": "Event category identified by SMI or the converter.",
            "Levels": {
                "fixation": "Fixation identified by the SMI event detector.",
                "saccade": "Saccade identified by the SMI event detector.",
                "blink": "Blink identified by the SMI event detector.",
                "message": "Message logged by the recording device.",
                "trigger": "Trigger-line event logged by the recording device.",
            },
        },
        "message": {"Description": "Free-text message logged by the eye tracker."},
    }
    if task_name is not None:
        sidecar["TaskName"] = task_name
    for source, name in extra_columns:
        sidecar[name] = _event_extra_column_metadata(source)
    return sidecar
