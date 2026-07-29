"""Read SMI *_Samples.txt files and prepare per-eye BIDS physio tables."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from ._column_names_helpers import (
    canonical_sample_column,
    column_unit,
    snake_case_column,
)
from ._invalid_samples import mark_invalid_samples, missing_mask


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplesHeader:
    """For storing the metadata information preceding the sample rows
    in a SMI *_Samples.txt file.

    This object retains the raw key/value metadata for _metadata.py
    and the table layout needed below to read the SMP rows.
    """

    path: Path  # e.g. /data/sub-01_task-rest_Samples.txt
    entries: dict[str, tuple[str, ...]]  # e.g. {"Sample Rate": ("30",)}
    columns: tuple[str, ...]  # e.g. ("Time", "Type", "L POR X [px]", ...)
    header_line: int  # e.g. 37 when the line beginning with "Time" is line 37
    sampling_frequency: float  # e.g. 30.0 from "## Sample Rate:\t30"
    declared_samples: int | None  # e.g. 7716 from "## Number of Samples: 7716"

    def first(self, key: str) -> str | None:
        """Return the first non-empty value; ``Sample Rate`` may yield ``30``."""
        for value in self.entries.get(key, ()):
            value = value.strip()
            if value:
                return value
        return None


def _metadata_line(line: str) -> tuple[str, str] | None:
    """Parse one SMI header key/value line, or ignore a non-metadata line.

    Examples:
    ``"## Sample Rate:\t30"`` -> ``("Sample Rate", "30")``
    ``"Converted from: NA"`` -> ``("Converted from", "NA")``
    ``"## [Calibration]"`` and ``"[iView]"`` -> ``None`` (section labels)
    ``"## Time"`` -> ``None`` (no colon/value) and
    ``"Time\tType\t..."`` -> ``None`` (the table header)
    """
    cleaned = line.strip()
    if cleaned.startswith("##"):
        cleaned = cleaned[2:].strip()
    if not cleaned or cleaned.startswith("[") or ":" not in cleaned:
        return None
    key, value = cleaned.split(":", 1)
    return key.strip(), value.strip()


def read_samples_header(path: str | Path) -> SamplesHeader:
    """Read SMI header entries and exact table columns before the sample rows.

    The function only records source information. _metadata.py later
    decides which entries become BIDS metadata, while _read_samples uses
    `columns` and `header_line` to read the table itself.
    """
    # resolve() here so that error messages contain the full path.
    sample_path = Path(path).expanduser().resolve()
    entries: dict[str, list[str]] = {}  # e.g. {"Sample Rate": ["30"]}
    columns: tuple[str, ...] | None = None  # e.g. ("Time", "Type", ...)
    header_line = 0  # later becomes, for example, 37

    try:
        with sample_path.open("r", encoding="utf-8", newline="") as source:
            for line_number, line in enumerate(source, 1):
                if line.startswith("Time\t"):
                    columns = tuple(line.rstrip("\r\n").split("\t"))
                    header_line = line_number
                    break
                parsed = _metadata_line(line)
                if parsed is not None:
                    key, value = parsed
                    entries.setdefault(key, []).append(value)
    except UnicodeError as error:
        raise ValueError(
            f"{sample_path}: cannot decode *_Samples.txt file: {error}"
        ) from error

    # For HBN recordings, all table headers are intact
    # including 33-, 38-, 42-, and 46-column layouts.
    # These checks are only to catch damaged or manually edited exports
    if columns is None:
        raise ValueError(
            f"{sample_path}: *_Samples.txt table header beginning with 'Time' "
            "was not found"
        )
    if len(columns) != len(set(columns)):
        raise ValueError(
            f"{sample_path}:{header_line}: *_Samples.txt table contains duplicate "
            "column names"
        )
    if "Type" not in columns or "Time" not in columns:
        raise ValueError(
            f"{sample_path}:{header_line}: *_Samples.txt table requires Time and "
            "Type columns"
        )

    # if sample rate is absent or an empty value, it means that
    # metadata.yaml must provide the sample rate or that no count can be checked.
    sample_rate_text = entries.get("Sample Rate", [""])[0]
    if not sample_rate_text:
        sampling_frequency = float("nan")
    else:
        try:
            sampling_frequency = float(sample_rate_text)
        except ValueError as error:
            raise ValueError(
                f"{sample_path}:{header_line}: Sample Rate is not numeric"
            ) from error
        if sampling_frequency <= 0:
            raise ValueError(
                f"{sample_path}:{header_line}: Sample Rate must be greater than zero"
            )

    declared_text = entries.get("Number of Samples", [""])[0]
    if declared_text:
        try:
            declared_samples = int(declared_text)
        except ValueError as error:
            raise ValueError(
                f"{sample_path}:{header_line}: "
                f"Number of Samples must be an integer, got {declared_text!r}",
            ) from error
    else:
        declared_samples = None

    return SamplesHeader(
        path=sample_path,
        entries={key: tuple(values) for key, values in entries.items()},
        columns=columns,
        header_line=header_line,
        sampling_frequency=sampling_frequency,
        declared_samples=declared_samples,
    )


def _read_samples(header: SamplesHeader) -> pd.DataFrame:
    """Read samples Rows and output as Dataframe
    This ignores MSG rows, as *_Events.txt file will be used for User Events
    (For the HBN dataset, User events in *_Events and *_Samples are identical)

    The header-driven pandas read and the decision to ignore MSG rows are
    adapted from read_raw_iview() in the HBN-specific eye-tracking pipeline.
    This version preserves the source strings for lossless BIDS output.
    """
    try:
        frame = pd.read_csv(
            header.path,
            sep="\t",
            skiprows=header.header_line,
            names=header.columns,
            header=None,
            dtype="string",
            encoding="utf-8",
            encoding_errors="strict",
            keep_default_na=False,
            engine="python",
            skip_blank_lines=False,
        )
    except (UnicodeError, pd.errors.ParserError) as error:
        raise ValueError(
            f"{header.path}: cannot parse *_Samples.txt table: {error}"
        ) from error

    # Keeping blank lines during parsing makes this index equal the physical
    # source line, so source-data errors can identify the exact row.
    frame.index = pd.RangeIndex(
        start=header.header_line + 1,
        stop=header.header_line + 1 + len(frame),
        name="source_line",
    )
    stripped = frame.apply(lambda column: column.str.strip())
    blank_rows = stripped.eq("").all(axis=1)
    row_types = stripped["Type"]

    # Rows that are neither blank nor of type SMP or MSG indicate a damaged or
    # unsupported export. HBN does not include such rows
    unknown = ~blank_rows & ~row_types.isin({"SMP", "MSG"})
    if unknown.any():
        line_number = int(unknown[unknown].index[0])
        raise ValueError(
            f"{header.path}:{line_number}: "
            f"unknown Samples row type {row_types.loc[line_number]!r}",
        )

    # *_Events.txt is the authoritative source for HBN messages
    return frame.loc[row_types.eq("SMP")].copy()


def _find_column(columns: tuple[str, ...], canonical_name: str) -> str | None:
    """e.g. to find ``L POR X [px]`` using ``L POR X``; return ``None`` if absent."""
    matches = [
        column
        for column in columns
        if canonical_sample_column(column) == canonical_name
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple SMI columns match {canonical_name!r}: {matches}")
    return matches[0] if matches else None


# Mapping *_Samples.txt columns to BIDS physio columns


@dataclass(frozen=True)
class SampleColumn:
    """Map one SMI sample column to one BIDS physio column.

    In ``_KNOWN_SAMPLE_COLUMNS``, ``source`` is the canonical SMI name, such
    as ``POR X``. Once a recording is prepared, it is the exact source header,
    such as ``L POR X [px]``. The remaining fields are the same in both forms.
    """

    source: str  # e.g. "POR X", later bound to "L POR X [px]"
    applies_to: str  # "recorded_eye" for L/R data; "both_eyes" for shared data
    name: str  # e.g. "x_coordinate" in the BIDS TSV and JSON
    description: str  # e.g. "Gaze position x-coordinate of the recorded eye."
    units: str | None = None  # e.g. "pixel"; None means no unit is written
    measurement_group: str | None = None  # e.g. paired point-of-regard values
    numeric: bool = True  # non-missing source cells must be numeric by default

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError(
                f"SampleColumn for {self.name!r} requires a source SMI column; "
                f"got {self.source!r}"
            )

    def bind_to_source(self, source: str) -> SampleColumn:
        """Bind canonical ``POR X`` to exact source ``L POR X [px]``."""
        output_column = replace(self, source=source)
        return replace(
            output_column,
            units=column_unit(output_column.source) or output_column.units,
        )


# The columns in the *_Samples.txt table are either:
# 1. Just relevant for the left eye (e.g. L Raw X [px])
# 2. Just relevant for the right eye (e.g. R CR2 X [px])
# 3. Relevant for both eyes (e.g. Trial or Trigger)
# BIDS instead stores each eye in a separate physio file.
# - `applies_to` captures this category:
#    - "recorded_eye" cols have an L/R prefix and go only to that eye's file
#    - "both_eyes" cols occur once in SMI and are copied to both BIDS files
#
# The key "POR X" describes both "L POR X [px]" and "R POR X [px]"
# Analog with other directional columns
#
# The other attributes describe the complete *_Samples.txt-to-BIDS mapping:
# - "name" is the BIDS output column name.
# - "description" and "units" become that column's JSON metadata. A unit
#   written in the actual SMI header takes precedence over the default here.
# - "numeric" means every non-missing source value must be numeric.
# - "measurement_group" defines the columns that are zero-masked together.
#   These mostly follow SMI's measurement families: point of regard, raw pupil
#   position, pupil dimensions, each individual corneal reflection, eye/head
#   position, head rotation, and gaze vector. Mapped diameter is kept separate
#   from eye-camera Dia X/Y because it is an independently named, millimetre
#   representation used as BIDS pupil_size. None means that zero is not treated
#   as a missing-value sentinel for this column.
# - "numeric" controls a source-data check without changing the original text.
#
# BIDS prescribes timestamp/x_coordinate/y_coordinate and optionally
# pupil_size. Names and descriptions for additional columns are self-authored
# and not taken from a specification. The values are SampleColumn objects too,
# rather than a second nearly identical definition structure.
_KNOWN_SAMPLE_COLUMNS: dict[str, SampleColumn] = {
    column.source: column
    for column in (
        SampleColumn(
            "POR X",
            "recorded_eye",
            "x_coordinate",
            "Gaze position x-coordinate of the recorded eye.",
            "pixel",
            "point_of_regard",
        ),
        SampleColumn(
            "POR Y",
            "recorded_eye",
            "y_coordinate",
            "Gaze position y-coordinate of the recorded eye.",
            "pixel",
            "point_of_regard",
        ),
        SampleColumn(
            "Mapped Diameter",
            "recorded_eye",
            "pupil_size",
            "Pupil diameter mapped by the SMI eye tracker.",
            "mm",
            "mapped_pupil_diameter",
        ),
        SampleColumn(
            "Raw X",
            "recorded_eye",
            "raw_pupil_x",
            "Raw pupil position on the eye camera image, x-axis.",
            "pixel",
            "raw_pupil_position",
        ),
        SampleColumn(
            "Raw Y",
            "recorded_eye",
            "raw_pupil_y",
            "Raw pupil position on the eye camera image, y-axis.",
            "pixel",
            "raw_pupil_position",
        ),
        SampleColumn(
            "Dia X",
            "recorded_eye",
            "pupil_diameter_x",
            "Pupil diameter along the eye-camera x-axis.",
            "pixel",
            "pupil_dimensions",
        ),
        SampleColumn(
            "Dia Y",
            "recorded_eye",
            "pupil_diameter_y",
            "Pupil diameter along the eye-camera y-axis.",
            "pixel",
            "pupil_dimensions",
        ),
        SampleColumn(
            "Dia",
            "recorded_eye",
            "pupil_diameter",
            "Pupil diameter in the SMI eye-camera image.",
            "pixel",
            "pupil_dimensions",
        ),
        SampleColumn(
            "Area",
            "recorded_eye",
            "pupil_area",
            "Pupil area in the SMI eye-camera image.",
            "pixel2",
            "pupil_dimensions",
        ),
        SampleColumn(
            "CR1 X",
            "recorded_eye",
            "corneal_reflection_1_x",
            "First corneal-reflection position, x-axis.",
            "pixel",
            "corneal_reflection_1",
        ),
        SampleColumn(
            "CR1 Y",
            "recorded_eye",
            "corneal_reflection_1_y",
            "First corneal-reflection position, y-axis.",
            "pixel",
            "corneal_reflection_1",
        ),
        SampleColumn(
            "CR2 X",
            "recorded_eye",
            "corneal_reflection_2_x",
            "Second corneal-reflection position, x-axis.",
            "pixel",
            "corneal_reflection_2",
        ),
        SampleColumn(
            "CR2 Y",
            "recorded_eye",
            "corneal_reflection_2_y",
            "Second corneal-reflection position, y-axis.",
            "pixel",
            "corneal_reflection_2",
        ),
        SampleColumn(
            "Validity",
            "recorded_eye",
            "validity",
            "SMI per-eye quality value; coding depends on the device version.",
        ),
        SampleColumn(
            "Plane",
            "recorded_eye",
            "plane",
            "SMI plane number intersected by the gaze.",
        ),
        SampleColumn(
            "EPOS X",
            "recorded_eye",
            "eye_position_x",
            "Eye position in the SMI three-dimensional coordinate system, x-axis.",
            "mm",
            "eye_position",
        ),
        SampleColumn(
            "EPOS Y",
            "recorded_eye",
            "eye_position_y",
            "Eye position in the SMI three-dimensional coordinate system, y-axis.",
            "mm",
            "eye_position",
        ),
        SampleColumn(
            "EPOS Z",
            "recorded_eye",
            "eye_position_z",
            "Eye position in the SMI three-dimensional coordinate system, z-axis.",
            "mm",
            "eye_position",
        ),
        SampleColumn(
            "GVEC X",
            "recorded_eye",
            "gaze_vector_x",
            "Normalized SMI gaze vector, x component.",
            None,
            "gaze_vector",
        ),
        SampleColumn(
            "GVEC Y",
            "recorded_eye",
            "gaze_vector_y",
            "Normalized SMI gaze vector, y component.",
            None,
            "gaze_vector",
        ),
        SampleColumn(
            "GVEC Z",
            "recorded_eye",
            "gaze_vector_z",
            "Normalized SMI gaze vector, z component.",
            None,
            "gaze_vector",
        ),
        SampleColumn(
            "Time",
            "both_eyes",
            "timestamp",
            "Continuously increasing SMI iView X device timestamp.",
            "us",
        ),
        SampleColumn("Trial", "both_eyes", "trial", "SMI trial number."),
        SampleColumn(
            "Timing",
            "both_eyes",
            "timing",
            "SMI timing-violation flag (1 indicates delayed processing).",
        ),
        SampleColumn(
            "Latency",
            "both_eyes",
            "latency",
            "SMI sample-processing latency.",
            "us",
        ),
        SampleColumn(
            "Pupil Confidence",
            "both_eyes",
            "pupil_confidence",
            "SMI pupil-confidence value; coding depends on the device version.",
        ),
        SampleColumn(
            "H POS X",
            "both_eyes",
            "head_position_x",
            "Head position, x-axis.",
            "mm",
            "head_position",
        ),
        SampleColumn(
            "H POS Y",
            "both_eyes",
            "head_position_y",
            "Head position, y-axis.",
            "mm",
            "head_position",
        ),
        SampleColumn(
            "H POS Z",
            "both_eyes",
            "head_position_z",
            "Head position, z-axis.",
            "mm",
            "head_position",
        ),
        SampleColumn(
            "H ROT X",
            "both_eyes",
            "head_rotation_x",
            "Head rotation, x-axis.",
            "deg",
            "head_rotation",
        ),
        SampleColumn(
            "H ROT Y",
            "both_eyes",
            "head_rotation_y",
            "Head rotation, y-axis.",
            "deg",
            "head_rotation",
        ),
        SampleColumn(
            "H ROT Z",
            "both_eyes",
            "head_rotation_z",
            "Head rotation, z-axis.",
            "deg",
            "head_rotation",
        ),
        SampleColumn(
            "Trigger",
            "both_eyes",
            "trigger",
            "Status of the SMI trigger input.",
        ),
        SampleColumn(
            "Frame",
            "both_eyes",
            "frame",
            "SMI frame counter.",
            numeric=False,
        ),
        SampleColumn(
            "Aux1",
            "both_eyes",
            "aux1",
            "SMI auxiliary data channel.",
            numeric=False,
        ),
    )
}


@dataclass
class PreparedSampleRecording:
    """Final in-memory form immediately before writing one physio file pair."""

    eye: str  # e.g. "left", written as RecordedEye in the JSON
    table: pd.DataFrame  # exact rows and columns written to _physio.tsv.gz
    output_columns: list[SampleColumn]  # same column order plus JSON details


def _sample_output_column_for_source(source: str, eye: str) -> SampleColumn:
    """Map a source such as ``L Raw X [px]`` to BIDS ``raw_pupil_x``."""
    canonical = canonical_sample_column(source)
    if canonical.startswith(("L ", "R ")):
        _, key = canonical.split(" ", 1)
        applies_to = "recorded_eye"
    else:
        key = canonical
        applies_to = "both_eyes"

    definition = _KNOWN_SAMPLE_COLUMNS.get(key)
    if definition is not None:
        return definition.bind_to_source(source)

    output_name = snake_case_column(key)
    if applies_to == "recorded_eye":
        return SampleColumn(
            source=source,
            applies_to=applies_to,
            name=output_name,
            description=f"SMI source column {source!r} for the {eye} eye.",
            units=column_unit(source),
        )
    return SampleColumn(
        source=source,
        applies_to=applies_to,
        name=output_name,
        description=f"SMI source column {source!r}, copied to both eye recordings.",
        units=column_unit(source),
    )


def _sample_output_columns(
    sample_header: SamplesHeader, eye: str, side: str
) -> list[SampleColumn]:
    """Build one eye's ordered BIDS columns from the *_Samples.txt header."""
    if not any(
        canonical_sample_column(column).startswith(f"{side} ")
        for column in sample_header.columns
    ):
        return []

    x_source = _find_column(sample_header.columns, f"{side} POR X")
    y_source = _find_column(sample_header.columns, f"{side} POR Y")
    output_columns = [
        _KNOWN_SAMPLE_COLUMNS["Time"],
        _KNOWN_SAMPLE_COLUMNS["POR X"].bind_to_source(x_source),
        _KNOWN_SAMPLE_COLUMNS["POR Y"].bind_to_source(y_source),
    ]
    mapped_source = _find_column(sample_header.columns, f"{side} Mapped Diameter")
    if mapped_source is not None:
        output_columns.append(
            _KNOWN_SAMPLE_COLUMNS["Mapped Diameter"].bind_to_source(mapped_source)
        )

    handled_sample_sources = {"Time", "Type", x_source, y_source}
    if mapped_source is not None:
        handled_sample_sources.add(mapped_source)

    opposite_side = "R" if side == "L" else "L"
    for source in sample_header.columns:
        if source in handled_sample_sources:
            continue
        canonical = canonical_sample_column(source)
        # For a left-eye output, skip R-prefixed fields (and vice versa).
        # Shared fields have neither prefix and therefore remain in both files.
        if canonical.startswith(f"{opposite_side} "):
            continue
        output_columns.append(_sample_output_column_for_source(source, eye))

    sample_output_names = [column.name for column in output_columns]
    duplicate_sample_output_names = sorted(
        name for name, count in Counter(sample_output_names).items() if count > 1
    )
    if duplicate_sample_output_names:
        raise ValueError(
            f"{sample_header.path}: BIDS output column-name collision for {eye}: "
            f"{duplicate_sample_output_names}"
        )
    return output_columns


def _sample_output_columns_by_eye(
    sample_header: SamplesHeader,
) -> dict[str, list[SampleColumn]]:
    """Build the left and right output layouts present in the source file."""
    output_columns_by_eye: dict[str, list[SampleColumn]] = {}
    left_output_columns = _sample_output_columns(sample_header, "left", "L")
    right_output_columns = _sample_output_columns(sample_header, "right", "R")
    if left_output_columns:
        output_columns_by_eye["left"] = left_output_columns
    if right_output_columns:
        output_columns_by_eye["right"] = right_output_columns
    if not output_columns_by_eye:
        raise ValueError(
            f"{sample_header.path}: no eye has both POR X and POR Y columns"
        )
    return output_columns_by_eye


# *_Samples.txt source-data checks and numeric view


def _numeric_sample_columns(
    samples: pd.DataFrame,
    output_columns_by_eye: dict[str, list[SampleColumn]],
    path: Path,
) -> dict[str, pd.Series]:
    """Check mapped numeric fields and create numbers for timing / n/a tests.

    For example, ``L POR X='431.2'`` is accepted and represented numerically;
    ``L POR X='lost'`` raises a source filename-and-line-number error. Original
    strings remain untouched in the output table.
    """
    # HBN does not contain numeric columns with non-numeric values
    # this is another protection from corruped files
    numeric_sources = {
        column.source
        for output_columns in output_columns_by_eye.values()
        for column in output_columns
        if column.numeric
    }

    numeric_sample_columns: dict[str, pd.Series] = {}
    for source in numeric_sources:
        text = samples[source].fillna("").astype("string").str.strip()
        missing = missing_mask(text)
        numbers = pd.to_numeric(text.mask(missing), errors="coerce")
        invalid = ~missing & numbers.isna()
        if invalid.any():
            line_number = int(invalid[invalid].index[0])
            raise ValueError(
                f"{path}:{line_number}: column {source!r} must be numeric, "
                f"got {text.loc[line_number]!r}"
            )
        numeric_sample_columns[source] = numbers
    return numeric_sample_columns


def _check_sample_timestamps(
    samples: pd.DataFrame,
    timestamps: pd.Series,
    sampling_frequency: float,
    path: Path,
) -> None:
    """Reject missing/repeated timestamps and report irregular sample spacing.

    Real HBN recordings contain increasing gaps up to about 218 ms, presumably
    from dropped or delayed samples, so irregular spacing is not malformed but
    warrants a warning
    """
    missing_timestamp = timestamps.isna()
    if missing_timestamp.any():
        line_number = int(missing_timestamp[missing_timestamp].index[0])
        raise ValueError(f"{path}:{line_number}: sample timestamp is missing")

    intervals = timestamps.diff()
    non_increasing = intervals.le(0).fillna(False)
    if non_increasing.any():
        line_number = int(non_increasing[non_increasing].index[0])
        raise ValueError(
            f"{path}:{line_number}: sample timestamp "
            f"{samples.loc[line_number, 'Time']} is not strictly increasing"
        )

    expected_interval = 1_000_000 / sampling_frequency
    interval_tolerance = max(2, expected_interval * 0.05)
    irregular_intervals = int(
        intervals.sub(expected_interval).abs().gt(interval_tolerance).sum()
    )
    if irregular_intervals:
        logger.warning(
            "%s: %d of %d sample intervals differ from %.6f us by more than 5%%",
            path,
            irregular_intervals,
            max(len(samples) - 1, 0),
            float(expected_interval),
        )


def _prepare_eye_recording(
    samples: pd.DataFrame,
    output_columns: list[SampleColumn],
    numeric_columns: dict[str, pd.Series],
    eye: str,
) -> PreparedSampleRecording:
    """Mark unavailable values and assemble one final per-eye recording."""
    table, kept_columns, absent_count, missing_counts = mark_invalid_samples(
        samples, output_columns, numeric_columns
    )
    omitted_columns = [
        column.name for column in output_columns if column not in kept_columns
    ]
    if omitted_columns:
        logger.info(
            "%s eye: omitted wholly empty additional columns: %s",
            eye,
            ", ".join(omitted_columns),
        )

    missing_categories = {"eye_absent": absent_count, **missing_counts}
    for category, count in sorted(missing_categories.items()):
        logger.info(
            "%s eye %s: %d/%d samples (%.2f%%) written as n/a",
            eye,
            category,
            count,
            len(samples),
            100 * count / len(samples),
        )
    return PreparedSampleRecording(eye, table, kept_columns)


def _check_sample_count(sample_header: SamplesHeader, sample_count: int) -> None:
    """Reject an empty table and report disagreement with the declared count."""
    if sample_count == 0:
        raise ValueError(
            f"{sample_header.path}: *_Samples.txt file contains no SMP rows"
        )
    # A count mismatch is suspicious but does not make otherwise readable rows
    # unusable, so report it and continue with the rows actually parsed
    if (
        sample_header.declared_samples is not None
        and sample_count != sample_header.declared_samples
    ):
        logger.warning(
            "%s: declared %d samples but parsed %d; using parsed rows",
            sample_header.path,
            sample_header.declared_samples,
            sample_count,
        )


# Reading, checking, marking unavailable data, and splitting by eye


def prepare_sample_recordings(
    sample_header: SamplesHeader, sampling_frequency: float
) -> dict[str, PreparedSampleRecording]:
    """Check source rows and prepare one BIDS physio table per recorded eye.

    These checks catch malformed SMI input before conversion: absent required
    POR columns, unknown row types, nonnumeric required measurements, missing
    or non-increasing timestamps, and implausible sample counts or intervals.
    They do not repair malformed input. Later in this same function,
    ``mark_invalid_samples`` separately translates SMI zero-filled unavailable
    measurements to BIDS ``n/a``.
    """
    # Define the exact source-to-output column mapping for each recorded eye.
    output_columns_by_eye = _sample_output_columns_by_eye(sample_header)

    # Read only SMP rows, retaining the source line number as the DataFrame index.
    samples = _read_samples(sample_header)
    _check_sample_count(sample_header, len(samples))

    # Parse numeric copies for source checks and missing-measurement detection;
    # the original strings remain in ``samples`` for lossless output.
    numeric_columns = _numeric_sample_columns(
        samples, output_columns_by_eye, sample_header.path
    )
    # Timestamps must increase; uneven but increasing spacing is only reported
    # because dropped samples can legitimately produce a longer interval.
    _check_sample_timestamps(
        samples,
        numeric_columns["Time"],
        sampling_frequency,
        sample_header.path,
    )

    # Mark unavailable values and create the final per-eye tables.
    recordings: dict[str, PreparedSampleRecording] = {}
    for eye, output_columns in output_columns_by_eye.items():
        recordings[eye] = _prepare_eye_recording(
            samples, output_columns, numeric_columns, eye
        )
    return recordings
