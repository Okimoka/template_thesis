"""Public SMI-to-BIDS conversion function."""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path

from ._events import prepare_event_recordings, read_events
from ._io import commit_files, write_gzip_dataframe, write_gzip_rows, write_json
from ._metadata import (
    make_physio_sidecar,
    make_physioevents_sidecar,
    prepare_metadata,
)
from ._samples import prepare_sample_recordings, read_samples_header


def _validate_bids_prefix(prefix: str) -> None:
    # Checks the prefix that was supplied by the user to use for output files
    # e.g. "sub-01_task-taskname_run-1". A suffix such as _physio is not
    # accepted because smi2bids adds recording- and the file suffix itself.
    if not prefix or "/" in prefix or "\\" in prefix or "." in prefix:
        raise ValueError(f"invalid BIDS prefix {prefix!r}")
    entities = prefix.split("_")
    parsed: list[tuple[str, str]] = []
    for entity in entities:
        match = re.fullmatch(r"([a-zA-Z0-9]+)-([a-zA-Z0-9+]+)", entity)
        if match is None:
            raise ValueError(f"invalid BIDS entity {entity!r} in prefix {prefix!r}")
        parsed.append((match.group(1), match.group(2)))
    keys = [key for key, _ in parsed]
    if keys[0] != "sub" or "task" not in keys:
        raise ValueError("BIDS prefix must begin with sub- and include a task- entity")
    if len(keys) != len(set(keys)):
        raise ValueError(f"BIDS prefix contains duplicate entities: {prefix!r}")
    if "recording" in keys:
        raise ValueError("BIDS prefix must not include recording-; smi2bids adds it")


# BIDS RecordedEye uses eye1 for the left and eye2 for the right eye
def _eye_label(eye: str) -> str:
    return "eye1" if eye == "left" else "eye2"


def smi2bids(
    samples_file: str | Path,
    events_file: str | Path,
    metadata_file: str | Path,
    output_dir: str | Path,
    bids_prefix: str,
    start_time: float,
) -> None:
    """Convert an SMI *_Samples.txt/*_Events.txt pair into BIDS eye tracking files."""

    # check correctness of user-provided bids prefix and start_time
    _validate_bids_prefix(bids_prefix)
    if not isinstance(start_time, (int, float)) or not math.isfinite(float(start_time)):
        raise ValueError(f"StartTime must be a finite number, got {start_time!r}")

    # the *_Samples.txt header supplies metadata that might not be
    # in the user-provided YAML. The effective sampling frequency is used while
    # checking and splitting the samples data into recordings by eye.
    sample_header = read_samples_header(samples_file)
    physio_metadata, bids_events_metadata, sampling_frequency = prepare_metadata(
        sample_header, metadata_file, start_time
    )
    sample_recordings = prepare_sample_recordings(sample_header, sampling_frequency)

    # Parse fixations, saccades, blinks, messages, triggers from events file
    # eye-specific rows are used for the eye-specific recordings
    # shared rows are used for both recordings.
    events_path = Path(events_file).expanduser().resolve()
    if not events_path.is_file():
        raise FileNotFoundError(f"{events_path}: *_Events.txt file does not exist")
    event_recordings = prepare_event_recordings(
        read_events(events_path),
        list(sample_recordings),
    )

    # init events metadata
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    events_destination = output_path / f"{bids_prefix}_events.json"

    # build the complete output bundle in a temporary directory,
    # only replace into the output_dir when all TSV.GZ and JSON files are written
    with tempfile.TemporaryDirectory(prefix=".smi2bids-", dir=output_path) as temp_name:
        temp_root = Path(temp_name)
        generated: dict[Path, Path] = {}

        # write continuous samples and their matching JSON sidecar for each eye.
        for eye, recording in sample_recordings.items():
            stem = f"{bids_prefix}_recording-{_eye_label(eye)}_physio"
            tsv_temp = temp_root / f"{stem}.tsv.gz"
            json_temp = temp_root / f"{stem}.json"
            write_gzip_dataframe(tsv_temp, recording.table)
            write_json(
                json_temp,
                make_physio_sidecar(
                    recording,
                    physio_metadata,
                    bids_events_metadata.get("TaskName"),
                ),
            )
            generated[tsv_temp] = output_path / tsv_temp.name
            generated[json_temp] = output_path / json_temp.name

        # write discrete eye-tracker events and messages for each eye.
        for eye, recording in event_recordings.items():
            stem = f"{bids_prefix}_recording-{_eye_label(eye)}_physioevents"
            tsv_temp = temp_root / f"{stem}.tsv.gz"
            json_temp = temp_root / f"{stem}.json"
            write_gzip_rows(tsv_temp, recording.rows)
            write_json(
                json_temp,
                make_physioevents_sidecar(
                    recording.output_columns,
                    recording.extra_columns,
                    bids_events_metadata.get("TaskName"),
                ),
            )
            generated[tsv_temp] = output_path / tsv_temp.name
            generated[json_temp] = output_path / json_temp.name

        # the ordinary run-level events.json is shared by both eye recordings.
        events_temp = temp_root / events_destination.name
        write_json(events_temp, bids_events_metadata)
        generated[events_temp] = events_destination

        # atomically write the complete bundle, restoring old files if a move
        # fails partway through.
        commit_files(generated, temp_root)
    return None
