#!/bin/bash
#SBATCH --job-name=topoplot_run3
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0-5:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="${SCRIPT_DIR:-$SUBMIT_DIR}"

TOPOPLOT_SCRIPT="${TOPOPLOT_SCRIPT:-$SCRIPT_DIR/create_topoplot2.jl}"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
JULIA_BIN="${JULIA_BIN:-julia}"
RUN3_INPUT_KIND="${RUN3_INPUT_KIND:-both}"
JULIA_CONDAPKG_BACKEND="${JULIA_CONDAPKG_BACKEND:-MicroMamba}"
PYTHONCALL_EXE="${PYTHONCALL_EXE:-$SCRIPT_DIR/.CondaPkg/env/bin/python}"
PYTHONCALL_LIB="${PYTHONCALL_LIB:-$SCRIPT_DIR/.CondaPkg/env/lib/libpython3.so}"

RUN_TAG="${RUN_TAG:-${RUN3_INPUT_KIND}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$SCRIPT_DIR/topoplot_outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE_DIR/$RUN_TAG}"

if [[ ! -f "$TOPOPLOT_SCRIPT" ]]; then
    echo "Topoplot script not found: $TOPOPLOT_SCRIPT" >&2
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
export RUN3_INPUT_KIND
export JULIA_PYTHONCALL_EXE="$PYTHONCALL_EXE"
export JULIA_PYTHONCALL_LIB="$PYTHONCALL_LIB"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

CMD=(
    "$JULIA_BIN"
    "--project=$PROJECT_DIR"
    "$TOPOPLOT_SCRIPT"
)

echo "Submit directory: $SUBMIT_DIR"
echo "Project directory: $PROJECT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "RUN3_INPUT_KIND: $RUN3_INPUT_KIND"
echo "JULIA_PYTHONCALL_EXE: $JULIA_PYTHONCALL_EXE"
echo "JULIA_PYTHONCALL_LIB: $JULIA_PYTHONCALL_LIB"
echo "Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
