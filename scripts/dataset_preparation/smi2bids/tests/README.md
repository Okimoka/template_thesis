# HBN test recordings

`smi2bids` was tested using the eye-tracking recordings provided by the Healthy Brain Network (HBN) dataset. To demonstrate the functionality of `smi2bids`, this folder contains scripts to download and run these example recordings locally.

From the `smi2bids` project directory run:

```console
python -m pip install -e .
python tests/download_hbn_examples.py
python tests/regenerate_outputs.py
```

The downloader retrieves the eight exact HBN files from the public HBN S3
bucket and verifies their SHA-256 checksums. Repeated conversion runs replace
the existing generated files.

| Input directory | SMI columns | Sample rows | Output columns per eye | Task |
| --- | ---: | ---: | ---: | --- |
| `sub-NDARAA306NT2_task-DiaryOfAWimpyKid_run-1` | 46 | 7,444 | 24 | Video-WK |
| `sub-NDARAB674LNB_task-DiaryOfAWimpyKid_run-1` | 46 | 3,615 | 24 | Video-WK |
| `sub-NDARAB793GL3_task-symbolSearch_run-1` | 38 | 13,779 | 23 | WISC ProcSpeed |
| `sub-NDARAC904DMU_task-symbolSearch_run-1` | 33 | 7,716 | 18 | WISC ProcSpeed |

The locally generated files are written to `bids_output`. They use
`StartTime: 0` because synchronization is deliberately outside `smi2bids`.



## Validate the output

With the current official BIDS Validator installed:

```console
bids-validator-deno tests/bids_output
```

BIDS Validator 3.0.1 reports zero errors for this retained
dataset, meaning it is in adherence with the specification. Its remaining warnings concern optional or recommended metadata that is intentionally absent from these minimal examples.

## Attribution


HBN datasets are distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with a subset of
participants distributed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).


- Alexander, L. M. et al. (2017). *An open resource for transdiagnostic
  research in pediatric mental health and learning disorders*. Scientific
  Data, 4, 170181. https://doi.org/10.1038/sdata.2017.181
- Langer, N. et al. (2017). *A resource for assessing information processing
  in the developing brain using EEG and eye tracking*. Scientific Data, 4,
  170040. https://doi.org/10.1038/sdata.2017.40

The HBN project and access instructions are described in the
[official HBN FAQ](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/FAQ.html).
