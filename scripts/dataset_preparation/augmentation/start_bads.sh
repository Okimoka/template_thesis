#!/bin/bash
#SBATCH --job-name=badch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=6G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# these parameters were in practice sufficient
# to add bads to all freeView recordings

set -euo pipefail
source smi/bin/activate
python3 add_bad_channels.py