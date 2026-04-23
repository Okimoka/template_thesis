#!/usr/bin/env python3
"""
Run the full sync_quality grouping workflow with a configurable ROI.

This script covers:
- age quartiles
- sex groups
- p_factor quartiles
- attention quartiles
- internalizing quartiles
- externalizing quartiles
- ehq_total halves
- random male halves

For each group folder it runs `new_analyze_group_original.jl` once, then creates
fixed-amplitude comparison overlays for both wins_filtered and median_filtered.

Written by LLM
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComparisonConfig:
    name: str
    summary_file: str
    output_dir_name: str
    output_prefix: str


COMPARISON_CONFIGS = (
    ComparisonConfig(
        name="age",
        summary_file="unsynced_clean_age_quartile_groups.tsv",
        output_dir_name="age_quartile_amplitude_comparisons",
        output_prefix="age_quartile_comparison",
    ),
    ComparisonConfig(
        name="sex",
        summary_file="unsynced_clean_sex_groups.tsv",
        output_dir_name="sex_amplitude_comparisons",
        output_prefix="sex_comparison",
    ),
    ComparisonConfig(
        name="p_factor",
        summary_file="unsynced_clean_p_factor_quartile_groups.tsv",
        output_dir_name="p_factor_quartile_amplitude_comparisons",
        output_prefix="p_factor_quartile_comparison",
    ),
    ComparisonConfig(
        name="attention",
        summary_file="unsynced_clean_attention_quartile_groups.tsv",
        output_dir_name="attention_quartile_amplitude_comparisons",
        output_prefix="attention_quartile_comparison",
    ),
    ComparisonConfig(
        name="internalizing",
        summary_file="unsynced_clean_internalizing_quartile_groups.tsv",
        output_dir_name="internalizing_quartile_amplitude_comparisons",
        output_prefix="internalizing_quartile_comparison",
    ),
    ComparisonConfig(
        name="externalizing",
        summary_file="unsynced_clean_externalizing_quartile_groups.tsv",
        output_dir_name="externalizing_quartile_amplitude_comparisons",
        output_prefix="externalizing_quartile_comparison",
    ),
    ComparisonConfig(
        name="ehq_total",
        summary_file="unsynced_clean_ehq_total_half_groups.tsv",
        output_dir_name="ehq_total_half_amplitude_comparisons",
        output_prefix="ehq_total_half_comparison",
    ),
    ComparisonConfig(
        name="male_random_half",
        summary_file="unsynced_clean_male_random_half_groups.tsv",
        output_dir_name="male_random_half_amplitude_comparisons",
        output_prefix="male_random_half_comparison",
    ),
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Run the full sync_quality grouping workflow into a fresh, tagged "
            "output tree using a configurable ROI."
        )
    )
    parser.add_argument(
        "--run-tag",
        required=True,
        help="Name of the fresh output directory inside sync_quality/visual_outputs.",
    )
    parser.add_argument(
        "--roi-channels",
        default="E76,E75,E71,E83,E72,E70,E77,E67,E82",
        help="Comma-separated ROI channels, e.g. E76,E75,E71,...",
    )
    parser.add_argument(
        "--male-random-seed",
        type=int,
        default=20260422,
        help="Seed for the reproducible male-half split.",
    )
    parser.add_argument(
        "--filename-suffix",
        default="proc-clean_raw.jld2",
        help="Filename suffix passed to new_analyze_group_original.jl.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root used as the working directory for Julia commands.",
    )
    parser.add_argument(
        "--sync-quality-dir",
        type=Path,
        default=script_dir,
        help="Path to the sync_quality directory.",
    )
    parser.add_argument(
        "--julia",
        default="julia",
        help="Julia executable to use.",
    )
    parser.add_argument(
        "--only",
        choices=[config.name for config in COMPARISON_CONFIGS],
        action="append",
        help="Restrict the run to one or more grouping sets.",
    )
    parser.add_argument(
        "--skip-group-creation",
        action="store_true",
        help="Assume all symlink group folders already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_args()


def shell_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def selected_configs(args: argparse.Namespace) -> tuple[ComparisonConfig, ...]:
    if not args.only:
        return COMPARISON_CONFIGS
    wanted = set(args.only)
    return tuple(config for config in COMPARISON_CONFIGS if config.name in wanted)


def read_group_names(summary_file: Path) -> list[str]:
    with summary_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = {"group_name"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required column(s) in {summary_file}: {', '.join(sorted(missing))}"
            )
        return [
            row["group_name"].strip()
            for row in reader
            if (row.get("group_name") or "").strip()
        ]


def run_command(
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    env_updates: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    prefix = "[dry-run]" if dry_run else "[run]"
    print(f"{prefix} {shell_text(command)}")

    if env_updates:
        env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_updates.items())
        print(f"{prefix} env -> {env_text}")
    if log_path is not None:
        print(f"{prefix} log -> {log_path}")

    if dry_run:
        return

    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)

    if log_path is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    sync_quality_dir = args.sync_quality_dir.resolve()

    if not repo_root.is_dir():
        raise NotADirectoryError(repo_root)
    if not sync_quality_dir.is_dir():
        raise NotADirectoryError(sync_quality_dir)

    output_root = sync_quality_dir / "visual_outputs" / args.run_tag
    logs_root = sync_quality_dir / "run_logs" / args.run_tag

    if output_root.exists():
        raise FileExistsError(
            f"Refusing to reuse existing output directory: {output_root}"
        )
    if logs_root.exists():
        raise FileExistsError(
            f"Refusing to reuse existing log directory: {logs_root}"
        )

    configs = selected_configs(args)
    if not configs:
        raise ValueError("No grouping configurations were selected.")

    create_age_script = sync_quality_dir / "create_unsynced_clean_age_quartiles.py"
    create_metric_script = sync_quality_dir / "create_unsynced_clean_metric_groups.py"
    create_male_half_script = sync_quality_dir / "create_unsynced_clean_random_male_halves.py"
    analyze_script = sync_quality_dir / "new_analyze_group_original.jl"
    compare_script = sync_quality_dir / "plot_group_amplitude_comparisons.jl"

    if not args.skip_group_creation:
        run_command(
            [sys.executable, str(create_age_script)],
            cwd=repo_root,
            dry_run=args.dry_run,
        )
        run_command(
            [sys.executable, str(create_metric_script)],
            cwd=repo_root,
            dry_run=args.dry_run,
        )
        run_command(
            [
                sys.executable,
                str(create_male_half_script),
                "--seed",
                str(args.male_random_seed),
            ],
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    output_root.mkdir(parents=True, exist_ok=True) if not args.dry_run else None
    logs_root.mkdir(parents=True, exist_ok=True) if not args.dry_run else None

    analysis_env = {
        "UNFOLD_ROI_CHANNELS": args.roi_channels,
        "UNFOLD_OUTPUT_BASE_DIR": str(output_root),
    }
    comparison_env = {
        "UNFOLD_GROUP_OUTPUTS_ROOT": str(output_root),
    }

    for config in configs:
        summary_file = sync_quality_dir / config.summary_file
        if not summary_file.exists():
            raise FileNotFoundError(summary_file)

        group_names = read_group_names(summary_file)
        if not group_names:
            raise ValueError(f"No group names were found in {summary_file}")

        print()
        print(f"[{config.name}] {len(group_names)} group(s)")

        for group_name in group_names:
            group_dir = sync_quality_dir / group_name
            if not group_dir.is_dir():
                raise NotADirectoryError(group_dir)

            log_path = logs_root / f"{group_name}.log"
            run_command(
                [
                    args.julia,
                    "--project=.",
                    str(analyze_script),
                    str(group_dir),
                    args.filename_suffix,
                ],
                cwd=repo_root,
                dry_run=args.dry_run,
                env_updates=analysis_env,
                log_path=log_path,
            )

        comparison_output_dir = output_root / config.output_dir_name
        for statistic_filter in ("wins_filtered", "median_filtered"):
            run_command(
                [
                    args.julia,
                    "--project=.",
                    str(compare_script),
                    str(summary_file),
                    args.filename_suffix.removesuffix(".jld2"),
                    statistic_filter,
                    str(comparison_output_dir),
                    config.output_prefix,
                ],
                cwd=repo_root,
                dry_run=args.dry_run,
                env_updates=comparison_env,
            )


if __name__ == "__main__":
    main()
