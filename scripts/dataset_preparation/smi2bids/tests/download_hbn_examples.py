"""Download the HBN recordings used by the representative conversions."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


TESTS_DIR = Path(__file__).resolve().parent
INPUTS_DIR = TESTS_DIR / "inputs"
HBN_EEG_URL = "https://fcp-indi.s3.amazonaws.com/data/Projects/HBN/EEG"

# The keys are paths below tests/inputs. Checksums make it apparent if an HBN
# object changes and prevent a partial download from being used as test input.
HBN_FILES = {
    "sub-NDARAA306NT2_task-DiaryOfAWimpyKid_run-1/NDARAA306NT2_Video-WK_Samples.txt": "1237183fd6263010cf45a4e589032949eac1a06a32c18e8d3266451f9e9b9bbc",
    "sub-NDARAA306NT2_task-DiaryOfAWimpyKid_run-1/NDARAA306NT2_Video-WK_Events.txt": "d7724475646a538d13b9e93c17fe39b833e838c14628b8777937c764a7d7c132",
    "sub-NDARAB674LNB_task-DiaryOfAWimpyKid_run-1/NDARAB674LNB_Video-WK_Samples.txt": "793bfc4bc26182f54406db438fda6d4bad2ac46cca7419be565e880bdfd246c9",
    "sub-NDARAB674LNB_task-DiaryOfAWimpyKid_run-1/NDARAB674LNB_Video-WK_Events.txt": "92684b3b37d8024b87c5520a5451da1b957496b20997f822ee728c2fd165874b",
    "sub-NDARAB793GL3_task-symbolSearch_run-1/NDARAB793GL3_WISC_ProcSpeed_Samples.txt": "e3df9ac8bf266e4c2d1266f378c1221a72ce484f6e1811daea0209008220dce4",
    "sub-NDARAB793GL3_task-symbolSearch_run-1/NDARAB793GL3_WISC_ProcSpeed_Events.txt": "ce039de2c369b6ee4c1db4c2c86d0765cca258265a749e3bbe6d9ce1b1fd2c93",
    "sub-NDARAC904DMU_task-symbolSearch_run-1/NDARAC904DMU_WISC_ProcSpeed_Samples.txt": "ee05e82cff8613f9c655922049b0ddfc0f3315d193fb33b278cee249b8c2226e",
    "sub-NDARAC904DMU_task-symbolSearch_run-1/NDARAC904DMU_WISC_ProcSpeed_Events.txt": "dd10b32275d1fe12bb17f7e876dee27716fafa546b2203cedd6fcf1dec1f5f79",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_url(destination: Path) -> str:
    """Map e.g. ``NDAR..._Samples.txt`` to its public HBN S3 object."""
    subject = destination.name.split("_", 1)[0]
    return f"{HBN_EEG_URL}/{subject}/Eyetracking/txt/{quote(destination.name)}"


def _download(relative_path: str, expected_sha256: str) -> None:
    destination = INPUTS_DIR / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and _sha256(destination) == expected_sha256:
        print(f"Already present and verified: {destination.relative_to(INPUTS_DIR)}")
        return

    request = Request(_source_url(destination), headers={"User-Agent": "smi2bids-tests"})
    temporary_path: Path | None = None
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 (fixed HBN URL)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".download",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                digest = hashlib.sha256()
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    temporary.write(chunk)
                    digest.update(chunk)

        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {destination.name}: expected "
                f"{expected_sha256}, got {digest.hexdigest()}"
            )
        temporary_path.replace(destination)
        print(f"Downloaded and verified: {destination.relative_to(INPUTS_DIR)}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    for relative_path, expected_sha256 in HBN_FILES.items():
        _download(relative_path, expected_sha256)


if __name__ == "__main__":
    main()
