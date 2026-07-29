"""Shared handling of SMI and BIDS column names."""

import re


def ascii_sanitize(text: str) -> str:
    """Make SMI labels ASCII: ``[°]`` -> ``[deg]`` and ``[µs]`` -> ``[us]``."""
    text = text.replace("°", "deg").replace("µ", "u").replace("²", "2")
    return text.encode("ascii", "ignore").decode("ascii")


def canonical_sample_column(name: str) -> str:
    # Compare Samples headers independently of units and repeated whitespace:
    # "L POR X   [px]" becomes "L POR X".
    without_unit = re.sub(r"\s*\[[^\]]+\]\s*$", "", name.strip())
    return re.sub(r"\s+", " ", without_unit).strip()


def column_unit(name: str) -> str | None:
    """Map ``L POR X [px]`` to ``pixel`` and a unitless ``Trial`` to ``None``."""
    match = re.search(r"\[([^\]]+)\]\s*$", name.strip())
    if not match:
        return None
    unit = ascii_sanitize(match.group(1).strip())
    return {
        "px": "pixel",
        "mm": "mm",
        "deg": "deg",
        "ms": "ms",
        "us": "us",
    }.get(unit, unit)


def snake_case_column(name: str) -> str:
    """Map ``Peak Speed At`` to ``peak_speed_at`` and ``3D X`` to ``value_3d_x``."""
    normalized = ascii_sanitize(name).replace(".", " ")
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
    if not normalized:
        raise ValueError(f"cannot derive a BIDS column name from {name!r}")
    if normalized[0].isdigit():
        normalized = f"value_{normalized}"
    return normalized
