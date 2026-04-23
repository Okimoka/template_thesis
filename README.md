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
First, the dataset needs to be downloaded. All info for this can be found in `download_dataset/Readme.md`.

Once the `DATASET` folder with all source data is ready, it can be merged into a unified BIDS dataset (`mergedDataset`). This only requires adjusting the input and output paths in `hbn-merge/create_merged_dataset.py` and running it (may appear stuck when running, but takes up to 20 minutes).

In theory, this is enough for a workable BIDS dataset. However, it is useful to also
1. Mark bad channels in the dataset. This can be done by running `augmentation/start_bads.sh` using sbatch or simply add_bad_channels.py manually after adjusting all file paths

2. Add the mock "freeView" and "allTasks" tasks. Instructions on how to do this are found in `augmentation/add_freeView_final.py`

To now produce `.fif` outputs with merged ET+EEG data, the correct version of the mne-bids-eyetracking-pipeline has to be used.
After the python venv is set up, replace the `mne_bids_pipeline` folder in the venv directory with the mne-bids-eyetracking-pipeline-smi-integration folder in `dataset_preparation` (rename it back to `mne_bids_pipeline`).

The pipeline can be run with `sbatch start_pipeline.sh`, after adjusting the config and required slurm parameters to the specific use case.

### Creating effects plot from outputted .fif files

1. Decide what kind of effects plots should be computed. These should be formatted like in `scripts/analysis/create_models/group_definitions.jl`, i.e.:

- A tuple of paths to the fifs will concatenate all recordings and compute one effects model
- A tuple containing a single fif path will fit the effects model on that recording 

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
