#!/bin/bash
#SBATCH --job-name=naive_frp_age_q
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0-03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="/home/st156392/topoplots"
PROJECT_DIR="/home/st156392/topoplots"
FRP_SCRIPT="${FRP_SCRIPT:-$SCRIPT_DIR/generate_naive_frp_age_quartiles.jl}"
JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_CONDAPKG_BACKEND="${JULIA_CONDAPKG_BACKEND:-MicroMamba}"
PYTHONCALL_EXE="/home/st156392/topoplots/.CondaPkg/env/bin/python"
PYTHONCALL_LIB="/home/st156392/topoplots/.CondaPkg/env/lib/libpython3.so"

if [[ ! -f "$FRP_SCRIPT" ]]; then
    echo "Naive FRP age-quartile script not found: $FRP_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/Project.toml" ]]; then
    echo "Julia project not found in: $PROJECT_DIR" >&2
    exit 1
fi

if ! command -v "$JULIA_BIN" >/dev/null 2>&1; then
    echo "Julia executable not found: $JULIA_BIN" >&2
    exit 1
fi

unset PYTHONHOME
unset PYTHONPATH

export JULIA_CONDAPKG_BACKEND
export JULIA_PYTHONCALL_EXE="$PYTHONCALL_EXE"
export JULIA_PYTHONCALL_LIB="$PYTHONCALL_LIB"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NAIVE_FRP_FIXED_THRESHOLD_UV="${NAIVE_FRP_FIXED_THRESHOLD_UV:-100}"
export NAIVE_FRP_GROUPS_FILE="${NAIVE_FRP_GROUPS_FILE:-$SCRIPT_DIR/more/generated_recording_groups2.jl}"
export NAIVE_FRP_MATCH_DATASET_SUBJECTS="${NAIVE_FRP_MATCH_DATASET_SUBJECTS:-1}"

CMD=(
    "$JULIA_BIN"
    "--project=$PROJECT_DIR"
    "$FRP_SCRIPT"
)

echo "Start time: $(date --iso-8601=seconds)"
echo "Submit directory: $SUBMIT_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Project directory: $PROJECT_DIR"
echo "FRP script: $FRP_SCRIPT"
echo "NAIVE_FRP_FIXED_THRESHOLD_UV: $NAIVE_FRP_FIXED_THRESHOLD_UV"
echo "NAIVE_FRP_GROUPS_FILE: $NAIVE_FRP_GROUPS_FILE"
echo "NAIVE_FRP_MATCH_DATASET_SUBJECTS: ${NAIVE_FRP_MATCH_DATASET_SUBJECTS}"
echo "NAIVE_FRP_MAX_SUBJECTS: ${NAIVE_FRP_MAX_SUBJECTS:-<unset>}"
echo "JULIA_PYTHONCALL_EXE: $JULIA_PYTHONCALL_EXE"
echo "JULIA_PYTHONCALL_LIB: $JULIA_PYTHONCALL_LIB"
echo "Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

cd "$PROJECT_DIR"
"${CMD[@]}"

echo "Finished at: $(date --iso-8601=seconds)"
