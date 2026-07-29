"""Regenerate the retained BIDS examples from their SMI input files."""

from pathlib import Path

from smi2bids import smi2bids


TESTS_DIR = Path(__file__).resolve().parent
INPUTS_DIR = TESTS_DIR / "inputs"
OUTPUT_DIR = TESTS_DIR / "bids_output"

EXAMPLES = (
    "sub-NDARAA306NT2_task-DiaryOfAWimpyKid_run-1",
    "sub-NDARAB674LNB_task-DiaryOfAWimpyKid_run-1",
    "sub-NDARAB793GL3_task-symbolSearch_run-1",
    "sub-NDARAC904DMU_task-symbolSearch_run-1",
)


def main() -> None:
    for bids_prefix in EXAMPLES:
        input_dir = INPUTS_DIR / bids_prefix
        subject = bids_prefix.split("_", 1)[0]
        output_dir = OUTPUT_DIR / subject / "beh"
        (samples_file,) = input_dir.glob("*_Samples.txt")
        (events_file,) = input_dir.glob("*_Events.txt")

        smi2bids(
            samples_file=samples_file,
            events_file=events_file,
            metadata_file=input_dir / "metadata.yaml",
            output_dir=output_dir,
            bids_prefix=bids_prefix,
            start_time=0,
        )
        print(f"Converted {bids_prefix}")


if __name__ == "__main__":
    main()
