
from argparse import Namespace
from pathlib import Path
from merge_bids.main import run

"""
Expects the following folder structure:

DATASET/
├─ ET/                                # Eye-Tracking
│  ├─ NDARAB977GFB/
│  │  ├─ Eyetracking/
│  │  │  ├─ txt/
│  │  │  │  ├─ NDARAB977GFB_Video-DM_Events.txt
│  │  │  │  ├─ NDARAB977GFB_Video-DM_Samples.txt
│  │  │  │  └─ ...                   # all ET .txt files for participant
│  │  │  └─ ...                      # optionally, tsv or idf folders
│  │  ├─ Behavioral/
│  │  │  └─ ...                      # any behavioral files for participant
│  │  └─ ...
│  ├─ <id>/
│  │  ├─ Eyetracking/
│  │  │  └─ txt/
│  │  │     └─ ...
│  │  ├─ Behavioral/
│  │  │  └─ ...
│  │  └─ ...
│  ├─ <id>/
│  │  ├─ Eyetracking/
│  │  │  └─ txt/
│  │  │     └─ ...
│  │  ├─ Behavioral/
│  │  │  └─ ...
│  │  └─ ...
│  └─ ...                            # for all subjects
│
└─ EEG/                               # 11 BIDS releases (randomly named)
    ├─ release1_any_name/
    │  ├─ dataset_description.json
    │  ├─ participants.tsv
    │  ├─ sub-NDARAB977GFB/
    │  │  └─ eeg/
    │  │     ├─ sub-NDARAB977GFB_task-ThePresent_eeg.set
    │  │     ├─ sub-NDARAB977GFB_task-ThePresent_eeg.json
    │  │     ├─ sub-NDARAB977GFB_task-ThePresent_channels.tsv
    │  │     ├─ sub-NDARAB977GFB_task-ThePresent_events.tsv
    │  │     └─ ...
    │  └─ ...                         # for all subjects in this release
    └─ release2/
    └─ ...

Some parts in merge_bids were LLM written
"""

#overwrites existing symlinks + root-level files on re-execution

args = Namespace(
    dataset=str(Path("/data/work/st156392/DATASET")),
    output=str(Path("/data/work/st156392/mergedDataset"))
)
run(args)
