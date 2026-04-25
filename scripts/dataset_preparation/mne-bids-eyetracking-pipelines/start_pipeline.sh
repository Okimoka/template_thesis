#!/bin/bash
#SBATCH --job-name=et_pipeline
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=2:00:00
# Keep this aligned with the number of subjects in subjects_list.txt.
#SBATCH --array=0-2230%24
#SBATCH --output=pipeline_logs/%x_%A_%a.out
#SBATCH --error=pipeline_logs/%x_%A_%a.err

# use absolute paths
SMI_DIR="/path/to/smi"
SUBJECT_LIST="/path/to/subjects_list.txt"
CONFIG_PATH="/path/to/config.py"

# activate the right python version
if [[ ! -d "$SMI_DIR" ]]; then
    echo "SMI directory not found: $SMI_DIR" >&2
    exit 1
fi

source "$SMI_DIR/bin/activate"
export PATH="$SMI_DIR/bin:$PATH"

# limit to 1 thread for each array job
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# subjects_list.txt needs list one subject per line (without "sub-")

if [[ ! -f "$SUBJECT_LIST" ]]; then
    echo "Subject list not found: $SUBJECT_LIST" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
fi

mapfile -t SUBJECTS < <(grep -v '^[[:space:]]*$' "$SUBJECT_LIST")

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set." >&2
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SUBJECTS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range for ${#SUBJECTS[@]} subjects." >&2
    exit 1
fi

SUBJ="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
#SUBJ="${SUBJECTS[$((SLURM_ARRAY_TASK_ID+2001))]}" # to continue from some point

mne_bids_pipeline --config="$CONFIG_PATH" --subject "$SUBJ"
