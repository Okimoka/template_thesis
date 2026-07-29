"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from .smi2bids import smi2bids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smi2bids",
        description="Convert SMI iView X IDF files that were exported to .txt files into BIDS compliant eye tracking files",
    )
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--events-file", required=True)
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bids-prefix", required=True)
    parser.add_argument("--start-time", required=True, type=float)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (repeat for debug logging).",
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and call :func:`smi2bids`."""
    args = _parser().parse_args(argv)
    level = logging.DEBUG if args.verbose > 1 else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    smi2bids(
        samples_file=args.samples_file,
        events_file=args.events_file,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
        bids_prefix=args.bids_prefix,
        start_time=args.start_time,
    )


if __name__ == "__main__":
    cli(sys.argv[1:])
