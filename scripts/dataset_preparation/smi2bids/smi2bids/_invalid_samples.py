"""Detect unavailable eye measurements and mark them as BIDS n/a values."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ._samples import SampleColumn


# These names are measurement_group values from the *_Samples.txt column mapping.
# If one sample from a group is present, it provides direct evidence that
# an eye was tracked in a sample.
_DIRECT_EYE_MEASUREMENT_GROUPS = {
    "raw_pupil_position",
    "pupil_dimensions",
    "mapped_pupil_diameter",
    "corneal_reflection_1",
    "corneal_reflection_2",
    "eye_position",
    "gaze_vector",
}


def missing_mask(values: pd.Series) -> pd.Series:
    """Identify missing text: ``""`` and ``"n/a"`` are true; ``"0"`` is false."""
    text = values.fillna("").astype("string").str.strip()
    return text.eq("") | text.str.lower().eq("n/a")


def _group_is_zero(
    numeric_columns: dict[str, pd.Series],
    output_columns: list[SampleColumn],
    measurement_group: str,
) -> pd.Series:
    """Mark rows where e.g. both POR coordinates are observed zeros."""
    group_columns = [
        column
        for column in output_columns
        if column.measurement_group == measurement_group
    ]
    if not group_columns:
        raise RuntimeError(f"unknown sample measurement group {measurement_group!r}")
    if any(column.source not in numeric_columns for column in group_columns):
        return pd.Series(False, index=next(iter(numeric_columns.values())).index)

    group_values = pd.concat(
        [numeric_columns[column.source] for column in group_columns], axis=1
    )
    observed = group_values.notna().any(axis=1)
    return observed & group_values.fillna(0).eq(0).all(axis=1)


def mark_invalid_samples(
    samples: pd.DataFrame,
    output_columns: list[SampleColumn],
    numeric_columns: dict[str, pd.Series],
) -> tuple[pd.DataFrame, list[SampleColumn], int, dict[str, int]]:
    """Replace unavailable measurements with n/a without dropping timestamps.

    For example, an SMI row with POR ``0, 0`` but pupil diameter ``10, 9``
    becomes gaze ``n/a, n/a`` while preserving pupil diameter ``10, 9``.

    This generalizes read_raw_iview()'s original fixed-column all-zero
    heuristic into named, eye-specific measurement groups.
    """
    output = samples[[column.source for column in output_columns]].copy()
    output.columns = [column.name for column in output_columns]
    output = output.apply(lambda column: column.fillna("").str.strip())

    measurement_groups = {
        column.measurement_group
        for column in output_columns
        if column.measurement_group is not None
    }
    direct_columns = [
        column
        for column in output_columns
        if column.measurement_group in _DIRECT_EYE_MEASUREMENT_GROUPS
    ]
    if direct_columns:
        direct_present = pd.Series(False, index=samples.index)
        for column in direct_columns:
            values = numeric_columns.get(column.source)
            if values is not None:
                direct_present |= values.notna() & values.ne(0)
        validity_column = next(
            (column for column in output_columns if column.name == "validity"), None
        )
        validity_present = pd.Series(False, index=samples.index)
        if validity_column is not None:
            validity = numeric_columns.get(validity_column.source)
            if validity is not None:
                validity_present = validity.eq(1).fillna(False)
        eye_present = direct_present | validity_present
    else:
        eye_present = ~_group_is_zero(
            numeric_columns, output_columns, "point_of_regard"
        )

    missing_groups: dict[str, pd.Series] = {}
    for group in measurement_groups:
        group_columns = [
            column for column in output_columns if column.measurement_group == group
        ]
        group_is_zero = _group_is_zero(numeric_columns, output_columns, group)
        # differentiate between eye-specific group and complete eye absence
        # head measurement groups are independent of eye presence
        if all(column.applies_to == "recorded_eye" for column in group_columns):
            group_is_zero &= eye_present
        missing_groups[group] = group_is_zero

    for column in output_columns:
        if column.measurement_group is not None:
            invalid = missing_groups[column.measurement_group]
            if column.applies_to == "recorded_eye":
                invalid |= ~eye_present
            output.loc[invalid, column.name] = "n/a"
        output.loc[missing_mask(output[column.name]), column.name] = "n/a"

    required_columns = {"timestamp", "x_coordinate", "y_coordinate"}
    empty_columns = [
        column
        for column in output.columns
        if column not in required_columns and missing_mask(output[column]).all()
    ]
    if empty_columns:
        output = output.drop(columns=empty_columns)
    kept_columns = [
        column for column in output_columns if column.name not in empty_columns
    ]
    missing_counts = {group: int(mask.sum()) for group, mask in missing_groups.items()}
    return output, kept_columns, int((~eye_present).sum()), missing_counts
