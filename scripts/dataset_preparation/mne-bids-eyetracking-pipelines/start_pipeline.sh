#!/bin/bash
#SBATCH --job-name=subj_synced
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=2:00:00
# Keep this aligned with the number of subjects in subjects_list.txt.
#SBATCH --array=0-2230%24
#SBATCH --output=brain_logs/%x_%A_%a.out
#SBATCH --error=brain_logs/%x_%A_%a.err

# safety checks LLM written

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# activate the right python version
source "$SUBMIT_DIR/smi/bin/activate"
export PATH=/pfs/work9/workspace/scratch/st_st156392-mydata/smi/bin:$PATH

# limit to 1 thread for each array job
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# subjects_list.txt needs list one subject per line (without "sub-")
SUBJECT_LIST="$SUBMIT_DIR/subjects_list.txt"
CONFIG_PATH="$SUBMIT_DIR/config.py"

if [[ ! -f "$SUBJECT_LIST" ]]; then
    echo "Subject list not found: $SUBJECT_LIST" >&2
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
