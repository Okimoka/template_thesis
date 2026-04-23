#!/bin/bash
#SBATCH --job-name=pipeline_full
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=12G
#SBATCH --time=1-12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# don't use this version! use start_pipeline.sh instead
# just for documentation purposes

source smi/bin/activate
mne_bids_pipeline --config=config.py
