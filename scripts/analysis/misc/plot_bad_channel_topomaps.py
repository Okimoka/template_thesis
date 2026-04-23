#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import mne

"""
Generates the heatmap of the bad channels distribution using the bad channels summary
The summary only contains bad channel detection for allTasks
Fully written by LLM
"""


DEFAULT_OUTPUT_DIR = Path("/home/oki/Desktop/EHLERS/unfold_bene/sync_quality/bads_heatmap")
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "bad_channels_summary.csv"
DEFAULT_TASKS = ("allTasks", "freeView")
DEFAULT_MONTAGE = "GSN-HydroCel-129"
CHANNELS = [f"E{i}" for i in range(1, 129)] + ["Cz"]


@dataclass
class TaskStats:
    task: str
    usable_rows: int
    bad_counts: dict[str, int]
    bad_fractions: dict[str, float]

    @property
    def top_channels(self) -> list[tuple[str, float]]:
        ranked = sorted(
            self.bad_fractions.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create MNE topomap heatmaps showing how often channels were marked "
            "bad for selected tasks."
        )
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=f"Summary CSV from summarize_bad_channels.py (default: {DEFAULT_SUMMARY_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output figures and CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help=f"Tasks to plot (default: {' '.join(DEFAULT_TASKS)})",
    )
    parser.add_argument(
        "--montage",
        default=DEFAULT_MONTAGE,
        help=f"MNE montage name to use (default: {DEFAULT_MONTAGE})",
    )
    return parser.parse_args()


def load_task_stats(summary_csv: Path, tasks: list[str]) -> list[TaskStats]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    task_stats: list[TaskStats] = []
    for task in tasks:
        usable_rows = [
            row
            for row in rows
            if row.get("task") == task and row.get("status_available") == "true"
        ]
        if not usable_rows:
            raise ValueError(f"No usable rows found for task {task!r} in {summary_csv}.")

        counts = Counter()
        for row in usable_rows:
            for channel in filter(None, row.get("bad_channels", "").split(";")):
                counts[channel] += 1

        missing_channels = [channel for channel in CHANNELS if channel not in counts]
        for channel in missing_channels:
            counts[channel] += 0

        fractions = {
            channel: counts[channel] / len(usable_rows)
            for channel in CHANNELS
        }
        task_stats.append(
            TaskStats(
                task=task,
                usable_rows=len(usable_rows),
                bad_counts=dict(counts),
                bad_fractions=fractions,
            )
        )

    return task_stats


def make_info(montage_name: str) -> mne.Info:
    info = mne.create_info(CHANNELS, sfreq=1000.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage)
    return info


def write_stats_csv(task_stats: list[TaskStats], output_path: Path) -> None:
    rows = []
    for stats in task_stats:
        for channel in CHANNELS:
            rows.append(
                {
                    "task": stats.task,
                    "usable_row_count": stats.usable_rows,
                    "channel": channel,
                    "bad_count": stats.bad_counts[channel],
                    "bad_fraction": f"{stats.bad_fractions[channel]:.6f}",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "usable_row_count",
                "channel",
                "bad_count",
                "bad_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def annotation_text(stats: TaskStats) -> str:
    top = ", ".join(
        f"{channel} {fraction * 100:.1f}%"
        for channel, fraction in stats.top_channels
    )
    return f"Rows with status: {stats.usable_rows}\nTop channels: {top}"


def plot_task_topomap(
    stats: TaskStats,
    info: mne.Info,
    output_path: Path,
    vmax: float,
) -> None:
    data = [stats.bad_fractions[channel] for channel in CHANNELS]

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image, _ = mne.viz.plot_topomap(
        data,
        info,
        axes=ax,
        show=False,
        cmap="YlOrRd",
        sensors=True,
        contours=8,
        res=256,
        vlim=(0.0, vmax),
    )
    ax.set_title(f"{stats.task}: bad-channel rate", pad=12)
    ax.text(
        0.02,
        0.02,
        annotation_text(stats),
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    cbar = fig.colorbar(image, ax=ax, shrink=0.82, format=PercentFormatter(xmax=1.0))
    cbar.set_label("Fraction of subject-task rows marked bad")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_combined_topomaps(
    task_stats: list[TaskStats],
    info: mne.Info,
    output_path: Path,
    vmax: float,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(task_stats),
        figsize=(7.0 * len(task_stats), 6.3),
        constrained_layout=True,
    )
    if len(task_stats) == 1:
        axes = [axes]

    last_image = None
    for ax, stats in zip(axes, task_stats):
        data = [stats.bad_fractions[channel] for channel in CHANNELS]
        last_image, _ = mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            cmap="YlOrRd",
            sensors=True,
            contours=8,
            res=256,
            vlim=(0.0, vmax),
        )
        ax.set_title(f"{stats.task}: bad-channel rate", pad=12)
        ax.text(
            0.02,
            0.02,
            annotation_text(stats),
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )

    cbar = fig.colorbar(
        last_image,
        ax=axes,
        shrink=0.86,
        format=PercentFormatter(xmax=1.0),
    )
    cbar.set_label("Fraction of subject-task rows marked bad")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary_csv = args.summary_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    info = make_info(args.montage)
    task_stats = load_task_stats(summary_csv, args.tasks)
    vmax = max(
        fraction
        for stats in task_stats
        for fraction in stats.bad_fractions.values()
    )

    stats_csv_path = output_dir / "bad_channel_topomap_stats.csv"
    write_stats_csv(task_stats, stats_csv_path)

    for stats in task_stats:
        figure_path = output_dir / f"bad_channel_topomap_{stats.task}.png"
        plot_task_topomap(stats, info, figure_path, vmax=vmax)

    combined_path = output_dir / "bad_channel_topomap_combined.png"
    plot_combined_topomaps(task_stats, info, combined_path, vmax=vmax)

    print(f"Wrote stats CSV: {stats_csv_path}")
    for stats in task_stats:
        print(
            f"Wrote figure:   {output_dir / f'bad_channel_topomap_{stats.task}.png'} "
            f"(usable rows: {stats.usable_rows})"
        )
    print(f"Wrote figure:   {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
