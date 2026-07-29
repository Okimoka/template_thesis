"""Serialize and atomically install converter outputs."""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


def write_json(path: Path, content: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(content, output, indent=2, ensure_ascii=False)
        output.write("\n")


def write_gzip_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("wb") as binary:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=binary, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, delimiter="\t", lineterminator="\n")
                writer.writerows(rows)


def write_gzip_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Write a headerless string DataFrame as a reproducible TSV.GZ file."""
    with path.open("wb") as binary:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=binary, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(
                    text,
                    sep="\t",
                    header=False,
                    index=False,
                    na_rep="n/a",
                    lineterminator="\n",
                )


def commit_files(files: dict[Path, Path], temp_root: Path) -> None:
    """Replace the complete output bundle, restoring old files on failure."""
    backup_root = temp_root / "backups"
    backup_root.mkdir()
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    existed = {
        destination: destination.exists() or destination.is_symlink()
        for destination in files.values()
    }
    try:
        for destination in files.values():
            if existed[destination]:
                backup = backup_root / f"{len(backups):04d}-{destination.name}"
                os.replace(destination, backup)
                backups[destination] = backup
        for temporary, destination in files.items():
            os.replace(temporary, destination)
            installed.append(destination)
    except OSError as error:
        for destination in installed:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise RuntimeError(
            f"failed to commit output bundle atomically: {error}"
        ) from error

    for destination in files.values():
        action = "Replaced" if existed[destination] else "Generated"
        logger.info("%s %s", action, destination)
