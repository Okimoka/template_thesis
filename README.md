# **MSc-Thesis:** Insights from a very large EEG+Eyetracking dataset
**Author:** *Okan Mazlum*

**Supervisor(s):** *Jevri Hanna*, *Benedikt Ehinger*

**Year:** *2026*

## Project Description
This project has two main parts:
1. It turns the Healthy Brain Network (HBN) EEG and eye-tracking recordings into a unified and practically usable combined resource by merging the existing HBN-EEG releases, linking the original eye-tracking files, synchronizing both data streams, and introducing quality metrics for curating usable recordings.

2. It uses the resulting dataset for an exploratory large-scale fixation-related potential (FRP) analysis in the free-viewing paradigms


## Instruction for a new student

### Setting up the dataset

Scripts for setting up the dataset are contained in `scripts/dataset_preparation`.
First, the dataset needs to be downloaded. All info for this can be found in `download_dataset/Readme.md` (+ some more info in the thesis appendix).

Once the `DATASET` folder with all source data is ready, it can be merged into a unified BIDS dataset (`mergedDataset`). This only requires adjusting the input and output paths in `hbn-merge/create_merged_dataset.py` and running it (may appear stuck when running, but takes up to 20 minutes).

In theory, this is enough for a workable BIDS dataset. However, it is useful to also

1. Add the mock "freeView" and "allTasks" tasks. Instructions on how to do this are found in `augmentation/add_freeView_final.py`

2. Mark bad channels in the dataset. This can be done using the `augmentation/add_bad_channels.py` script. First, adjust the list of tasks to add bad channels to (`TASKS_DEFAULT`). Run using `python3 add_bad_channels.py --root /path/to/mergedDataset` (use `--remove` parameter to restore to original channels files). Since bad channel detection takes a while, it is recommended to run this script via `sbatch start_bads.sh` instead, after adjusting the paths to the venv and mergedDataset there.



To now produce `.fif` outputs with merged ET+EEG data, the correct version of the mne-bids-eyetracking-pipeline has to be used. 

This repository contains three versions of the pipeline:
1. **mne-bids-eyetracking-pipeline-hbn-specific** (minimal changes to make it work with HBN + metrics)
2. **mne-bids-eyetracking-pipeline-used-ehlers** (the actual pipeline that was used for the analyses in the thesis)
3. **mne-bids-eyetracking-pipeline-smi-integration** (like 2., but slimmed down to remove unncecessary changes and split into commits to more easily be able to comprehend all changes. I recommend using this version, if it doesn't work, try 2.)

To setup the python venv, first create a new git repository and initialize it with the chosen modified pipeline (or just fork https://github.com/Okimoka/mne-bids-eyetracking-pipeline). Then modify the `requirements.txt` to point to your repository, i.e. change the line

```
mne-bids-pipeline @ git+https://github.com/Okimoka/mne-bids-eyetracking-pipeline.git@ec71625e61d72ec10de04c991a384fb6810c6bee
```

to

```
mne-bids-pipeline @ git+https://github.com/<username>/mne-bids-eyetracking-pipeline.git@<branch name>
```

(adapt branch name to smi-integration if forked)


Then setup the python venv (name of the venv is "smi" for all start scripts, so it's easiest to use it here too)

```bash
python3 -m venv smi
source smi/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

There are two main ways to make changes to the used pipeline, depending on whether you plan to make frequent prototyping changes to the pipeline, or only rarely:

1. **Rarely**: Every time you make the changes to your local version, push them to github (i.e. `git add . && git commit -m "message" && git push -u origin main`), then update via `python -m pip install --upgrade --force-reinstall -r requirements.txt`

2. **Frequently**: Symlink your local version into the venv (i.e. `ln -s /path/to/local_pipeline /path/to/smi/lib/python<version>/site-packages/mne_bids_pipeline`)

The pipeline can now be run. Adjust `config.py` to point to the correct dataset and make custom adjustments (e.g. only process one subject for a first test).

1. To run the pipeline locally, activate your venv and run `mne_bids_pipeline --config=/path/to/config.py`
2. To run on a cluster, use the `start_pipeline.sh` script. For this, first adjust the list of subjects you want to process in `subjects_list.txt`. Then go into `start_pipeline.sh` and adjust the number 2230 in `#SBATCH --array=0-2230%24` to instead correspond to the number of subjects (minus one) in `subjects_list.txt`. You may also adjust the number 24 for higher/lower parallelity. Then set the three paths for `SMI_DIR`, `SUBJECT_LIST` and `CONFIG_PATH` at the start of the file. Finally, the pipeline can be run with `sbatch start_pipeline.sh`.


Useful slurm commands:
```bash
squeue -u "$USER" # see queued and running jobs
scontrol show job <job_id> # see details for one job
sbatch start_pipeline.sh # start a job
scancel <job_id> # cancel one job or full job array
scancel <job_id>_<array_task_id> # cancel one array task

# inspect stdout/stderr logs while a job is running
tail -f pipeline_logs/<log_file>.out
tail -f pipeline_logs/<log_file>.err
```

### Creating effects plot from outputted .fif files

The general workflow is like this:
1. Define groups of models, e.g. one group could be "all cleaned ThePresent recordings of male subjects", and the comparison group could be "all cleaned ThePresent recorings of female subjects". This would then be defined in a file like this (see `scripts/analysis/create_models/group_definitions.jl` for a full example):
    ```
    male_thepresent_clean = [
        ("/path/to/sub-NDARAA306NT2_task-freeView_run-3_proc-clean_raw.fif"),
        ("/path/to/sub-NDARAB055BPR_task-freeView_run-3_proc-clean_raw.fif"),
        ("/path/to/sub-NDARAB348EWR_task-freeView_run-3_proc-clean_raw.fif"),
        ...
    ]

    male_thepresent_clean = [
        ("/path/to/sub-NDARAB674LNB_task-freeView_run-3_proc-clean_raw.fif"),
        ("/path/to/sub-NDARAC462DZH_task-freeView_run-3_proc-clean_raw.fif"),
        ("/path/to/sub-NDARAC589YMB_task-freeView_run-3_proc-clean_raw.fif"),
        ...
    ]
    ```

    Each of the entries is in a tuple, because you may want to compute the effects plot of the concatenated sum of multiple recordings, e.g. all recordings of each subject. In this case, the tuple will contain each of the recordings of the given subject.
    You can use the script `create_models/generate_group_definition.py` to create such a list, but depending on your use case you may have to write script that e.g. connects this to the `participants.tsv`.

TODO Make more elaborate

2. Once group_definitions.jl is done, run `sbatch start_chunked_array.sh` on a cluster to mass-fit all models. The result will be a folder for each list defined in group_definitions.jl containing the `.jld2` model files

3. Generate an average FRP by running `src/new_analyze_group_original.jl` (after entering the name of the folder containing the `.jld2` files). All other scripts in the analysis folder are suitable to be executed on such a folder of models



## Overview of Folder Structure 

```
│projectdir          <- Project's main folder. It is initialized as a Git
│                       repository with a reasonable .gitignore file.
│
├── report           <- **Immutable and add-only!**
│   ├── proposal     <- Proposal PDF
│   ├── thesis       <- Final Thesis PDF
│   ├── talks        <- PDFs (and optionally pptx etc) of the Intro,
|   |                   Midterm & Final-Talk
|
├── _research        <- WIP scripts, code, notes, comments,
│   |                   to-dos and anything in an alpha state.
│
├── plots            <- All exported plots go here, best in date folders.
|   |                   Note that to ensure reproducibility it is required that all plots can be
|   |                   recreated using the plotting scripts in the scripts folder.
|
├── notebooks        <- Pluto, Jupyter, Weave or any other mixed media notebooks.*
│
├── scripts          <- Various scripts, e.g. simulations, plotting, analysis,
│   │                   The scripts use the `src` folder for their base code.
│
├── src              <- Source code for use in this project. Contains functions,
│                       structures and modules that are used throughout
│                       the project and in multiple scripts.
│
├── test             <- Folder containing tests for `src`.
│   └── runtests.jl  <- Main test file
│   └── setup.jl     <- Setup test environment
│
├── README.md        <- Top-level README. A fellow student needs to be able to
|   |                   continue your project. Think about her!!
|
├── .gitignore       <- focused on Julia, but some Matlab things as well
│
├── (Manifest.toml)  <- Contains full list of exact package versions used currently.
|── (Project.toml)   <- Main project file, allows activation and installation.
└── (Requirements.txt)<- in case of python project - can also be an anaconda file, MakeFile etc.
                        
```

\*Instead of having a separate *notebooks* folder, you can also delete it and integrate your notebooks in the scripts folder. However, notebooks should always be marked by adding `nb_` in front of the file name.
