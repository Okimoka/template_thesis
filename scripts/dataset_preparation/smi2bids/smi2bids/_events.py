"""Read SMI *_Events.txt files and prepare BIDS physioevents tables."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path

import pandas as pd

from ._column_names_helpers import snake_case_column


logger = logging.getLogger(__name__)


# Parsed and prepared event data


@dataclass
class EventRow:
    """One parsed row from an SMI *_Events.txt table."""

    line_number: int
    onset: str  # e.g. "448528079", retained in SMI microseconds
    duration: str  # e.g. "0.016643", converted to seconds
    trial_type: str  # e.g. "fixation", "saccade", "blink", or "message"
    message: str  # e.g. "StimulusOnset" for UserEvent; otherwise "n/a"
    eye: str | None  # e.g. "left" for Fixation L; None for a shared message
    extras: dict[str, str] = field(  # e.g. {"Trial": "1", "End": "448544722"}
        default_factory=dict
    )


@dataclass
class PreparedEventRecording:
    """One eye's output rows and columns, ready for TSV/JSON writing.

    ``extra_columns`` contains ``(SMI source name, BIDS output name)`` pairs,
    for example ``("End", "end_timestamp")``. ``_metadata.py`` uses those
    pairs to describe the same columns in the JSON sidecar.
    """

    eye: str
    rows: list[list[str]]
    output_columns: list[str]
    extra_columns: list[tuple[str, str]]


_EVENT_SECTION_BY_PREFIX = {
    "fixation": "Fixations",
    "saccade": "Saccades",
    "blink": "Blinks",
    "userevent": "User Events",
    "trigger": "Trigger Line Events",
}

_KNOWN_EVENT_COLUMNS = {
    "Trial": "trial",
    "Number": "event_number",
    "End": "end_timestamp",
    "Location X": "location_x",
    "Location Y": "location_y",
    "Dispersion X": "dispersion_x",
    "Dispersion Y": "dispersion_y",
    "Plane": "plane",
    "Avg. Pupil Size X": "average_pupil_size_x",
    "Avg Pupil Size Y": "average_pupil_size_y",
    "Start Loc.X": "start_location_x",
    "Start Loc.Y": "start_location_y",
    "End Loc.X": "end_location_x",
    "End Loc.Y": "end_location_y",
    "Peak Speed": "peak_speed",
    "Peak Speed At": "peak_speed_at",
    "Average Speed": "average_speed",
    "Peak Accel.": "peak_acceleration",
    "Peak Decel.": "peak_deceleration",
    "Average Accel.": "average_acceleration",
    "Port Status": "port_status",
}


# Reading *_Events.txt


def _decimal(value: str, path: Path, line_number: int, field: str) -> Decimal:
    """Parse exact decimal text; for example ``"1.20"`` -> ``Decimal('1.20')``."""
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise ValueError(
            f"{path}:{line_number}: {field} must be numeric, got {value!r}"
        ) from error


def _event_section(event_type: str) -> str | None:
    """Map ``Fixation L`` to ``Fixations`` and ``UserEvent`` to ``User Events``."""
    normalized = event_type.strip().lower().replace(" ", "")
    for prefix, section in _EVENT_SECTION_BY_PREFIX.items():
        if normalized.startswith(prefix):
            return section
    return None


def _event_eye(event_type: str) -> str | None:
    """
    Maps event type to which specific eye it belongs to
    ``Fixation L`` -> ``left``
    ``Saccade R`` -> ``right``.
    shared ``UserEvent`` -> ``None``
    """
    match = re.search(r"(?:^|\s)(L|R)$", event_type.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return "left" if match.group(1).upper() == "L" else "right"


def _duration_seconds(value: str, path: Path, line_number: int) -> str:
    """Convert SMI microseconds: ``"16643"`` -> ``"0.016643"``."""
    duration = _decimal(value, path, line_number, "event duration")
    if duration < 0:
        raise ValueError(f"{path}:{line_number}: event duration must not be negative")
    seconds = duration / Decimal(1_000_000)
    text = format(seconds, "f").rstrip("0").rstrip(".")
    return text or "0"


def read_events(path: str | Path) -> list[EventRow]:
    """Read every declared table section from an SMI *_Events.txt export.

    For example, ``Fixation L`` uses the ``Table Header for Fixations``
    columns, whereas ``UserEvent`` uses ``Table Header for User Events``.
    """
    event_path = Path(path).expanduser().resolve()
    try:
        lines = event_path.read_text(encoding="utf-8").splitlines()
    except UnicodeError as error:
        raise ValueError(
            f"{event_path}: cannot decode *_Events.txt file: {error}"
        ) from error

    # First collect the separate layouts declared for fixations, saccades,
    # blinks, user events, and trigger events. A section may be absent when the
    # file contains no rows of that event type
    # Example for blinks table header:
    """
    Table Header for Blinks:
    Event Type	Trial	Number	Start	End	Duration
    """

    headers: dict[str, tuple[str, ...]] = {}
    data_start = 0
    for index, line in enumerate(lines):
        match = re.match(r"Table Header for (.+):\s*$", line)
        if not match:
            continue
        if index + 1 >= len(lines):
            raise ValueError(
                f"{event_path}:{index + 1}: event table header has no column line"
            )
        section = match.group(1).strip()
        columns = tuple(lines[index + 1].rstrip("\r\n").split("\t"))
        if not columns or columns[0] != "Event Type":
            raise ValueError(
                f"{event_path}:{index + 2}: invalid column header for {section}"
            )
        headers[section] = columns
        data_start = max(data_start, index + 2)

    if not headers:
        raise ValueError(f"{event_path}: no event table headers were found")

    # Group the physical rows by their declared table. The whole file cannot be
    # passed to one read_csv call because, for example, fixations and saccades
    # have different columns.
    rows_by_section: dict[str, list[str]] = {section: [] for section in headers}
    line_numbers_by_section: dict[str, list[int]] = {
        section: [] for section in headers
    }
    for line_number, line in enumerate(lines, 1):
        if (
            not line
            or line.startswith("Table Header for ")
            or line.startswith("Event Type\t")
        ):
            continue
        parts = line.split("\t")
        section = _event_section(parts[0])
        if section is None:
            if line_number > data_start:
                raise ValueError(
                    f"{event_path}:{line_number}: unknown event row type {parts[0]!r}"
                )
            continue
        columns = headers.get(section)
        if columns is None:
            raise ValueError(
                f"{event_path}:{line_number}: event {parts[0]!r} has no matching "
                "table header"
            )
        if len(parts) > len(columns):
            raise ValueError(
                f"{event_path}:{line_number}: event has {len(parts)} cells but "
                f"{section} header defines "
                f"{len(columns)}",
            )
        rows_by_section[section].append(line)
        line_numbers_by_section[section].append(line_number)

    # Using similar code as the mne-bids-eyetracking-pipeline to parse events
    # using pandas, once table layouts are identified
    # however, original strings and physical source-line numbers are retained here
    tables: dict[str, pd.DataFrame] = {}
    for section, rows in rows_by_section.items():
        if not rows:
            continue
        try:
            table = pd.read_csv(
                StringIO("\n".join(rows)),
                sep="\t",
                names=headers[section],
                header=None,
                dtype="string",
                keep_default_na=False,
                engine="python",
            )
        except (pd.errors.ParserError, ValueError) as error:
            raise ValueError(
                f"{event_path}: cannot parse {section} event table: {error}"
            ) from error
        table = table.fillna("")
        table.index = pd.Index(
            line_numbers_by_section[section], name="source_line"
        )
        tables[section] = table

    events: list[EventRow] = []
    for section, table in tables.items():
        for line_number, row in table.iterrows():
            onset = row.get("Start", "")
            _decimal(onset, event_path, line_number, "event onset")
            event_type = row["Event Type"].strip()

            # Convert SMI's section-specific core fields to the four common BIDS
            # columns. For example, Fixation L Start=1000000 Duration=20000 becomes
            # onset=1000000, duration=0.02, trial_type=fixation, message=n/a.
            if section == "User Events":
                trial_type = "message"
                duration = "0"
                message = row.get("Description", "").strip() or "n/a"
            elif section == "Trigger Line Events":
                trial_type = "trigger"
                duration = _duration_seconds(
                    row.get("Duration", "0"), event_path, line_number
                )
                message = "n/a"
            else:
                trial_type = {
                    "Fixations": "fixation",
                    "Saccades": "saccade",
                    "Blinks": "blink",
                }[section]
                duration = _duration_seconds(
                    row.get("Duration", "0"), event_path, line_number
                )
                message = "n/a"

            # Retain all remaining SMI metrics, such as fixation location or
            # saccade peak speed, for later column-name mapping.
            extras = {
                key: value
                for key, value in row.items()
                if key not in {"Event Type", "Start", "Duration", "Description"}
            }
            events.append(
                EventRow(
                    line_number=line_number,
                    onset=onset,
                    duration=duration,
                    trial_type=trial_type,
                    message=message,
                    eye=_event_eye(event_type),
                    extras=extras,
                )
            )

    # DataFrames were parsed section by section, so restore the interleaved
    # order in which SMI wrote the events to the source file.
    return sorted(events, key=lambda event: event.line_number)


def _is_missing(value: str) -> bool:
    """Treat ``""`` and ``"n/a"`` as missing, but retain values such as ``"0"``."""
    return not value.strip() or value.strip().lower() == "n/a"


# Mapping SMI event fields and preparing per-eye BIDS physioevents tables


def prepare_event_recordings(
    events: list[EventRow], eyes: list[str]
) -> dict[str, PreparedEventRecording]:
    """Route parsed events and create one BIDS table per eye.

    For example, a ``Fixation L`` row retains its SMI timestamp as ``onset``,
    converts its duration from microseconds to seconds, and is routed only to
    the left eye. A shared User Event becomes ``trial_type=message`` and is
    copied into both eye recordings.
    """
    # each eye's .tsv should have the same layout
    # even if only one eye uses a particular metric.
    event_extra_sources: list[str] = []
    for event in events:
        for source, value in event.extras.items():
            if not _is_missing(value) and source not in event_extra_sources:
                event_extra_sources.append(source)
    # known metrics receive explicitly mapped names
    # for unknown metrics, derived snake_case names are used
    event_extra_names = [
        _KNOWN_EVENT_COLUMNS[source]
        if source in _KNOWN_EVENT_COLUMNS
        else snake_case_column(source)
        for source in event_extra_sources
    ]
    duplicate_event_names = [
        name for name, count in Counter(event_extra_names).items() if count > 1
    ]
    if duplicate_event_names:
        raise ValueError(
            f"BIDS event column-name collision: {sorted(duplicate_event_names)}"
        )

    event_columns = [
        "onset",
        "duration",
        "trial_type",
        "message",
        *event_extra_names,
    ]
    recordings: dict[str, PreparedEventRecording] = {}
    for eye in eyes:
        # Shared messages/triggers have eye=None and are copied to both files;
        # fixation/saccade/blink rows go only to their recorded eye.
        eye_events = [
            event for event in events if event.eye is None or event.eye == eye
        ]
        if not eye_events:
            logger.info("No physioevents rows for %s eye; omitting optional pair", eye)
            continue
        # Fixations, saccades, and blinks expose different extra measurements,
        # but one TSV must have one fixed set of columns. For example, if the
        # combined extras are location_x and peak_speed, a fixation row contains
        # its location_x and n/a for peak_speed, a saccade does the reverse.
        event_rows = [
            [
                event.onset,
                event.duration,
                event.trial_type,
                event.message,
                *[
                    event.extras.get(source, "").strip() or "n/a"
                    for source in event_extra_sources
                ],
            ]
            for event in eye_events
        ]
        recordings[eye] = PreparedEventRecording(
            eye=eye,
            rows=event_rows,
            output_columns=list(event_columns),
            extra_columns=list(
                zip(event_extra_sources, event_extra_names, strict=True)
            ),
        )
    return recordings
