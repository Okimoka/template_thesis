#!/bin/bash
# Cluster usage from the internal get_groups directory:
#   bash start_chunked_array.sh
#   GROUP_DEFINITIONS_MODE=test bash start_chunked_array.sh
#
# Resume an existing chunked run:
#   RUN_DIR=/pfs/work9/workspace/scratch/st_st156392-mydata/get_groups/chunked_runs/<run_name> \
#   bash start_chunked_array.sh
#
# Local dry-run preparation from the mounted mirror:
#   GROUP_DEFINITIONS_MODE=test \
#   DRY_RUN=1 \
#   SUBMIT_ARRAY=0 \
#   PATH_PREFIX_FROM=/pfs/work9/workspace/scratch/st_st156392-mydata \
#   PATH_PREFIX_TO=/home/oki/bwuni-ws2 \
#   bash start_chunked_array.sh

set -euo pipefail

resolve_script_dir() {
    local source_path source_dir
    local -a candidates=()

    source_path="${BASH_SOURCE[0]}"
    source_dir="$(cd "$(dirname "${source_path}")" && pwd)"
    candidates+=("${source_dir}")

    if [[ -n "${SCRIPT_DIR:-}" ]]; then
        candidates=("${SCRIPT_DIR}" "${candidates[@]}")
    fi

    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
        local job_info command_path command_dir
        job_info="$(scontrol show job "${SLURM_JOB_ID}" -o 2>/dev/null || true)"
        if [[ "${job_info}" =~ Command=([^[:space:]]+) ]]; then
            command_path="${BASH_REMATCH[1]}"
            if [[ -f "${command_path}" ]]; then
                command_dir="$(cd "$(dirname "${command_path}")" && pwd)"
                candidates=("${command_dir}" "${candidates[@]}")
            fi
        fi
    fi

    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        candidates+=("$(cd "${SLURM_SUBMIT_DIR}" && pwd)")
    fi

    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}/run_new_uf2_group_definitions_parallel.jl" && -f "${candidate}/group_definitions.jl" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    echo "Could not locate the get_groups script directory. Checked: ${candidates[*]}" >&2
    return 1
}

SCRIPT_DIR="$(resolve_script_dir)"
BUILD_MANIFEST_SCRIPT="${SCRIPT_DIR}/build_chunked_manifest.jl"
SUMMARY_SCRIPT="${SCRIPT_DIR}/refresh_chunked_run_summary.jl"
LAUNCHER_SCRIPT="${SCRIPT_DIR}/run_new_uf2_group_definitions_parallel.jl"

GROUP_DEFINITIONS_MODE="${GROUP_DEFINITIONS_MODE:-full}"
GROUP_DEFINITIONS_FILE="${GROUP_DEFINITIONS_FILE:-}"
MODELS_PER_JOB="${MODELS_PER_JOB:-${CHUNK_SIZE:-400}}"
GPU_WORKERS_PER_CHUNK="${GPU_WORKERS_PER_CHUNK:-1}"
CPU_WORKERS_PER_CHUNK="${CPU_WORKERS_PER_CHUNK:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_ARRAY="${SUBMIT_ARRAY:-1}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-20}"
MAX_ARRAY_TASKS_PER_SUBMISSION="${MAX_ARRAY_TASKS_PER_SUBMISSION:-}"
SOLVER_FILTER="${SOLVER_FILTER:-all}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/chunked_runs}"
RUN_DIR="${RUN_DIR:-}"
PATH_PREFIX_FROM="${PATH_PREFIX_FROM:-}"
PATH_PREFIX_TO="${PATH_PREFIX_TO:-}"
GPU_PARTITION="${PARTITION:-gpu_a100_short}"
GPU_TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
GPU_CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
GPU_MEMORY_PER_TASK="${MEMORY_PER_TASK:-12G}"
GPU_RESOURCES="${GPU_RESOURCES:-gpu:1}"
CPU_PARTITION="${CPU_PARTITION:-cpu_il}"
CPU_TIME_LIMIT="${CPU_TIME_LIMIT:-${GPU_TIME_LIMIT}}"
CPU_CPUS_PER_TASK="${CPU_CPUS_PER_TASK:-${GPU_CPUS_PER_TASK}}"
CPU_MEMORY_PER_TASK="${CPU_MEMORY_PER_TASK:-${GPU_MEMORY_PER_TASK}}"
JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_CONDAPKG_BACKEND_VALUE="${JULIA_CONDAPKG_BACKEND_VALUE:-MicroMamba}"
TASK_ID_FILE="${TASK_ID_FILE:-}"

if [[ -d /pfs/work9/workspace/scratch/st_st156392-mydata/smi/bin ]]; then
    export PATH=/pfs/work9/workspace/scratch/st_st156392-mydata/smi/bin:$PATH
fi

if [[ -z "${GROUP_DEFINITIONS_FILE}" ]]; then
    case "${GROUP_DEFINITIONS_MODE}" in
        full)
            GROUP_DEFINITIONS_FILE="${SCRIPT_DIR}/group_definitions.jl"
            ;;
        test)
            GROUP_DEFINITIONS_FILE="${SCRIPT_DIR}/group_definitions_final_test.jl"
            ;;
        *)
            echo "Unsupported GROUP_DEFINITIONS_MODE=${GROUP_DEFINITIONS_MODE}. Use 'full' or 'test'." >&2
            exit 1
            ;;
    esac
fi

GROUP_DEFINITIONS_FILE="$(realpath "${GROUP_DEFINITIONS_FILE}")"
CHUNK_SIZE="${MODELS_PER_JOB}"

if [[ -n "${PATH_PREFIX_FROM}" || -n "${PATH_PREFIX_TO}" ]]; then
    if [[ -z "${PATH_PREFIX_FROM}" || -z "${PATH_PREFIX_TO}" ]]; then
        echo "Both PATH_PREFIX_FROM and PATH_PREFIX_TO must be set together." >&2
        exit 1
    fi
fi

if ! [[ "${CHUNK_SIZE}" =~ ^[0-9]+$ ]] || (( CHUNK_SIZE <= 0 )); then
    echo "MODELS_PER_JOB/CHUNK_SIZE must be a positive integer." >&2
    exit 1
fi

if ! [[ "${GPU_WORKERS_PER_CHUNK}" =~ ^[0-9]+$ ]] || (( GPU_WORKERS_PER_CHUNK <= 0 )); then
    echo "GPU_WORKERS_PER_CHUNK must be a positive integer." >&2
    exit 1
fi

if ! [[ "${CPU_WORKERS_PER_CHUNK}" =~ ^[0-9]+$ ]] || (( CPU_WORKERS_PER_CHUNK <= 0 )); then
    echo "CPU_WORKERS_PER_CHUNK must be a positive integer." >&2
    exit 1
fi

if ! [[ "${ARRAY_CONCURRENCY}" =~ ^[0-9]+$ ]] || (( ARRAY_CONCURRENCY <= 0 )); then
    echo "ARRAY_CONCURRENCY must be a positive integer." >&2
    exit 1
fi

case "${SOLVER_FILTER}" in
    all|cpu|gpu)
        ;;
    *)
        echo "SOLVER_FILTER must be one of: all, cpu, gpu." >&2
        exit 1
        ;;
esac

detect_max_array_tasks_per_submission() {
    local raw_max_array_size

    if command -v scontrol >/dev/null 2>&1; then
        raw_max_array_size="$(
            scontrol show config 2>/dev/null | awk -F '=' '
                /^[[:space:]]*MaxArraySize[[:space:]]*=/ {
                    gsub(/[[:space:]]/, "", $2)
                    print $2
                    exit
                }
            '
        )"

        if [[ "${raw_max_array_size}" =~ ^[0-9]+$ ]] && (( raw_max_array_size > 1 )); then
            printf '%s\n' "$(( raw_max_array_size - 1 ))"
            return 0
        fi
    fi

    printf '%s\n' "1000"
}

if [[ -z "${MAX_ARRAY_TASKS_PER_SUBMISSION}" ]]; then
    MAX_ARRAY_TASKS_PER_SUBMISSION="$(detect_max_array_tasks_per_submission)"
fi

if ! [[ "${MAX_ARRAY_TASKS_PER_SUBMISSION}" =~ ^[0-9]+$ ]] || (( MAX_ARRAY_TASKS_PER_SUBMISSION <= 0 )); then
    echo "MAX_ARRAY_TASKS_PER_SUBMISSION must be a positive integer." >&2
    exit 1
fi

summary_lock_dir() {
    printf '%s\n' "${RUN_DIR}/summary.lock"
}

refresh_run_summary_locked() {
    local lock_dir
    lock_dir="$(summary_lock_dir)"

    while ! mkdir "${lock_dir}" 2>/dev/null; do
        sleep 1
    done

    (
        trap 'rmdir "${lock_dir}" 2>/dev/null || true' EXIT
        JULIA_CONDAPKG_BACKEND="${JULIA_CONDAPKG_BACKEND_VALUE}" \
            "${JULIA_BIN}" \
            "${SUMMARY_SCRIPT}" \
            "${RUN_DIR}"
    )
}

escape_state_value() {
    local value="$1"
    value="${value//$'\n'/\\n}"
    printf '%s' "${value}"
}

write_state_file() {
    local status_stem="$1"
    local status="$2"
    local message="$3"
    local status_file="${status_stem}.state"

    mkdir -p "$(dirname "${status_file}")"
    {
        printf 'status=%s\n' "${status}"
        printf 'updated_at=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-}"
        printf 'array_task_id=%s\n' "${CURRENT_ARRAY_SLOT:-${SLURM_ARRAY_TASK_ID:-${ARRAY_TASK_ID:-}}}"
        printf 'task_id=%s\n' "${CURRENT_TASK_ID:-}"
        printf 'task_id_file=%s\n' "${TASK_ID_FILE:-}"
        printf 'host=%s\n' "$(hostname)"
        printf 'message=%s\n' "$(escape_state_value "${message}")"
    } > "${status_file}"
}

write_id_list_file() {
    local path="$1"
    shift
    local value

    mkdir -p "$(dirname "${path}")"
    : > "${path}"
    for value in "$@"; do
        printf '%s\n' "${value}" >> "${path}"
    done
}

build_array_spec() {
    local -a ids=("$@")
    local spec=""
    local range_start="" previous=""
    local id

    for id in "${ids[@]}"; do
        if [[ -z "${range_start}" ]]; then
            range_start="${id}"
            previous="${id}"
            continue
        fi

        if (( id == previous + 1 )); then
            previous="${id}"
            continue
        fi

        if [[ -n "${spec}" ]]; then
            spec+=","
        fi
        if (( range_start == previous )); then
            spec+="${range_start}"
        else
            spec+="${range_start}-${previous}"
        fi

        range_start="${id}"
        previous="${id}"
    done

    if [[ -n "${range_start}" ]]; then
        if [[ -n "${spec}" ]]; then
            spec+=","
        fi
        if (( range_start == previous )); then
            spec+="${range_start}"
        else
            spec+="${range_start}-${previous}"
        fi
    fi

    printf '%s\n' "${spec}"
}

write_run_documentation() {
    local run_readme="${RUN_DIR}/README.txt"
    local run_config="${RUN_DIR}/run_config.env"

    cat > "${run_readme}" <<EOF
Chunked UF2 run directory

Run directory: ${RUN_DIR}
Definitions file: ${GROUP_DEFINITIONS_FILE}
Definitions mode: ${GROUP_DEFINITIONS_MODE}
Models per chunk job: ${CHUNK_SIZE}
GPU workers per chunk: ${GPU_WORKERS_PER_CHUNK}
CPU workers per chunk: ${CPU_WORKERS_PER_CHUNK}
GPU partition: ${GPU_PARTITION}
CPU partition: ${CPU_PARTITION}
Skip existing: ${SKIP_EXISTING}
Dry run: ${DRY_RUN}
Solver filter: ${SOLVER_FILTER}

Important files:
- manifest.tsv: one row per chunk job
- planned_fits.tsv: one row per planned fit
- task_summary.tsv: chunk-level progress summary
- fit_status.tsv: fit-level progress summary
- progress_overview.txt: human-readable progress snapshot
- remaining_task_ids.txt: chunk ids still left to run
- array_batches/: task-id files used for each submitted SLURM array batch
- submission_history.tsv: actual array submissions and their task-id files
- results/task_XXXX.tsv: completed fits for each chunk
- status/task_XXXX.state: latest chunk state marker

Resume the same run:
RUN_DIR=${RUN_DIR} bash start_chunked_array.sh
EOF

    cat > "${run_config}" <<EOF
SCRIPT_DIR=${SCRIPT_DIR}
GROUP_DEFINITIONS_MODE=${GROUP_DEFINITIONS_MODE}
GROUP_DEFINITIONS_FILE=${GROUP_DEFINITIONS_FILE}
MODELS_PER_JOB=${CHUNK_SIZE}
CHUNK_SIZE=${CHUNK_SIZE}
GPU_WORKERS_PER_CHUNK=${GPU_WORKERS_PER_CHUNK}
CPU_WORKERS_PER_CHUNK=${CPU_WORKERS_PER_CHUNK}
SKIP_EXISTING=${SKIP_EXISTING}
DRY_RUN=${DRY_RUN}
SOLVER_FILTER=${SOLVER_FILTER}
PATH_PREFIX_FROM=${PATH_PREFIX_FROM}
PATH_PREFIX_TO=${PATH_PREFIX_TO}
GPU_PARTITION=${GPU_PARTITION}
GPU_TIME_LIMIT=${GPU_TIME_LIMIT}
GPU_CPUS_PER_TASK=${GPU_CPUS_PER_TASK}
GPU_MEMORY_PER_TASK=${GPU_MEMORY_PER_TASK}
GPU_RESOURCES=${GPU_RESOURCES}
CPU_PARTITION=${CPU_PARTITION}
CPU_TIME_LIMIT=${CPU_TIME_LIMIT}
CPU_CPUS_PER_TASK=${CPU_CPUS_PER_TASK}
CPU_MEMORY_PER_TASK=${CPU_MEMORY_PER_TASK}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY}
MAX_ARRAY_TASKS_PER_SUBMISSION=${MAX_ARRAY_TASKS_PER_SUBMISSION}
JULIA_BIN=${JULIA_BIN}
JULIA_CONDAPKG_BACKEND_VALUE=${JULIA_CONDAPKG_BACKEND_VALUE}
EOF
}

manifest_solver_backend_for_task() {
    local task_id="$1"
    awk -F '\t' -v target="${task_id}" '
        NR == 1 { next }
        $1 == target { print $4; exit }
    ' "${RUN_DIR}/manifest.tsv"
}

submit_batches_for_task_group() {
    local task_group="$1"
    local partition="$2"
    local cpus_per_task="$3"
    local memory_per_task="$4"
    local time_limit="$5"
    local gpu_resources="$6"
    shift 6

    local batch_index batch_start batch_size batch_file batch_label batch_first_task_id batch_last_task_id
    local array_argument sbatch_output job_id dependency_job_id dependency_value
    local submission_prefix history_file
    local -a task_ids sbatch_args batch_task_ids

    task_ids=("$@")
    if (( ${#task_ids[@]} == 0 )); then
        return 0
    fi

    mkdir -p "${RUN_DIR}/array_batches"
    history_file="${RUN_DIR}/submission_history.tsv"
    if [[ ! -f "${history_file}" ]]; then
        printf 'submitted_at\tjob_id\tdependency\tarray_argument\tbatch_file\tbatch_size\tfirst_task_id\tlast_task_id\n' > "${history_file}"
    fi

    echo "${task_group} chunk tasks: ${#task_ids[@]}"
    echo "  Partition: ${partition}"
    echo "  CPUs per task: ${cpus_per_task}"
    echo "  Memory per task: ${memory_per_task}"
    echo "  Time limit: ${time_limit}"
    if [[ -n "${gpu_resources}" ]]; then
        echo "  GPU resources: ${gpu_resources}"
    else
        echo "  GPU resources: none"
    fi

    submission_prefix="$(date '+%Y%m%d_%H%M%S')_$$_${task_group}"
    batch_index=0
    dependency_job_id=""

    for (( batch_start = 0; batch_start < ${#task_ids[@]}; batch_start += MAX_ARRAY_TASKS_PER_SUBMISSION )); do
        batch_index=$((batch_index + 1))
        batch_task_ids=( "${task_ids[@]:batch_start:MAX_ARRAY_TASKS_PER_SUBMISSION}" )
        batch_size="${#batch_task_ids[@]}"
        batch_first_task_id="${batch_task_ids[0]}"
        batch_last_task_id="${batch_task_ids[$((batch_size - 1))]}"
        batch_file="${RUN_DIR}/array_batches/${submission_prefix}_part_$(printf '%03d' "${batch_index}").task_ids"
        write_id_list_file "${batch_file}" "${batch_task_ids[@]}"

        if (( batch_size == 1 )); then
            array_argument="1"
        else
            array_argument="1-${batch_size}%${ARRAY_CONCURRENCY}"
        fi

        batch_label="${task_group} submission batch ${batch_index}: manifest tasks ${batch_first_task_id}-${batch_last_task_id} (${batch_size} chunks)"
        echo "${batch_label}"
        echo "  Task id file: ${batch_file}"
        echo "  Array spec: ${array_argument}"

        if [[ "${SUBMIT_ARRAY}" != "1" ]]; then
            continue
        fi

        sbatch_args=(
            --job-name "fit_models_chunked_${task_group}"
            --partition "${partition}"
            --nodes 1
            --ntasks 1
            --cpus-per-task "${cpus_per_task}"
            --mem "${memory_per_task}"
            --time "${time_limit}"
            --output "${RUN_DIR}/logs/%x_%A_%a.out"
            --error "${RUN_DIR}/logs/%x_%A_%a.err"
            --array "${array_argument}"
        )

        if [[ -n "${gpu_resources}" ]]; then
            sbatch_args+=(--gres "${gpu_resources}")
        fi

        dependency_value=""
        if [[ -n "${dependency_job_id}" ]]; then
            dependency_value="afterany:${dependency_job_id}"
            sbatch_args+=(--dependency "${dependency_value}")
        fi

        sbatch_args+=(
            --export "ALL,SCRIPT_DIR=${SCRIPT_DIR},RUN_DIR=${RUN_DIR},GROUP_DEFINITIONS_MODE=${GROUP_DEFINITIONS_MODE},GROUP_DEFINITIONS_FILE=${GROUP_DEFINITIONS_FILE},MODELS_PER_JOB=${CHUNK_SIZE},CHUNK_SIZE=${CHUNK_SIZE},GPU_WORKERS_PER_CHUNK=${GPU_WORKERS_PER_CHUNK},CPU_WORKERS_PER_CHUNK=${CPU_WORKERS_PER_CHUNK},SKIP_EXISTING=${SKIP_EXISTING},DRY_RUN=${DRY_RUN},SOLVER_FILTER=${SOLVER_FILTER},PATH_PREFIX_FROM=${PATH_PREFIX_FROM},PATH_PREFIX_TO=${PATH_PREFIX_TO},GPU_PARTITION=${GPU_PARTITION},GPU_TIME_LIMIT=${GPU_TIME_LIMIT},GPU_CPUS_PER_TASK=${GPU_CPUS_PER_TASK},GPU_MEMORY_PER_TASK=${GPU_MEMORY_PER_TASK},GPU_RESOURCES=${GPU_RESOURCES},CPU_PARTITION=${CPU_PARTITION},CPU_TIME_LIMIT=${CPU_TIME_LIMIT},CPU_CPUS_PER_TASK=${CPU_CPUS_PER_TASK},CPU_MEMORY_PER_TASK=${CPU_MEMORY_PER_TASK},ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY},MAX_ARRAY_TASKS_PER_SUBMISSION=${MAX_ARRAY_TASKS_PER_SUBMISSION},JULIA_BIN=${JULIA_BIN},JULIA_CONDAPKG_BACKEND_VALUE=${JULIA_CONDAPKG_BACKEND_VALUE},SUBMIT_ARRAY=1,TASK_ID_FILE=${batch_file}"
            "$0"
            --worker
        )

        sbatch_output="$(sbatch "${sbatch_args[@]}")"
        echo "${sbatch_output}"

        job_id="$(printf '%s\n' "${sbatch_output}" | awk '{print $NF}')"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" \
            "${job_id}" \
            "${dependency_value}" \
            "${array_argument}" \
            "${batch_file}" \
            "${batch_size}" \
            "${batch_first_task_id}" \
            "${batch_last_task_id}" >> "${history_file}"

        dependency_job_id="${job_id}"
        echo "Submitted array job ${job_id} for ${batch_label}."
    done
}

resolve_task_id_from_array_slot() {
    local array_slot="$1"
    local resolved_task_id

    if [[ -n "${TASK_ID_FILE}" ]]; then
        [[ -f "${TASK_ID_FILE}" ]] || {
            echo "TASK_ID_FILE does not exist: ${TASK_ID_FILE}" >&2
            exit 1
        }

        resolved_task_id="$(sed -n "${array_slot}p" "${TASK_ID_FILE}" | tr -d '[:space:]')"
        [[ -n "${resolved_task_id}" ]] || {
            echo "No manifest task id found for array slot ${array_slot} in ${TASK_ID_FILE}" >&2
            exit 1
        }
        printf '%s\n' "${resolved_task_id}"
        return 0
    fi

    printf '%s\n' "${array_slot}"
}

worker_mode() {
    local task_id manifest_line launcher_exit_code array_slot
    local task_id_field label group_name solver_backend formula_mode event_eye event_onset workers
    local chunk_index chunk_count start_group_index end_group_index fit_count
    local chunk_definition_file results_file status_stem
    local manifest_header
    local -a launcher_args

    [[ -n "${RUN_DIR}" ]] || {
        echo "RUN_DIR must be set in worker mode." >&2
        exit 1
    }

    array_slot="${SLURM_ARRAY_TASK_ID:-${ARRAY_TASK_ID:-}}"
    [[ -n "${array_slot}" ]] || {
        echo "No array task id found. Use SLURM_ARRAY_TASK_ID or ARRAY_TASK_ID." >&2
        exit 1
    }
    [[ "${array_slot}" =~ ^[0-9]+$ ]] || {
        echo "Invalid array task id: ${array_slot}" >&2
        exit 1
    }
    task_id="$(resolve_task_id_from_array_slot "${array_slot}")"
    export CURRENT_ARRAY_SLOT="${array_slot}"
    export CURRENT_TASK_ID="${task_id}"

    manifest_line="$(
        awk -F '\t' -v target="${task_id}" '
            NR == 1 { next }
            $1 == target { print; exit }
        ' "${RUN_DIR}/manifest.tsv"
    )"
    [[ -n "${manifest_line}" ]] || {
        echo "Task id ${task_id} was not found in ${RUN_DIR}/manifest.tsv" >&2
        exit 1
    }

    manifest_header="$(head -n 1 "${RUN_DIR}/manifest.tsv")"
    if [[ "${manifest_header}" == *$'\tevent_onset\t'* || "${manifest_header}" == event_onset$'\t'* || "${manifest_header}" == *$'\tevent_onset' ]]; then
        IFS=$'\t' read -r \
            task_id_field \
            label \
            group_name \
            solver_backend \
            formula_mode \
            event_eye \
            event_onset \
            workers \
            chunk_index \
            chunk_count \
            start_group_index \
            end_group_index \
            fit_count \
            chunk_definition_file \
            results_file \
            status_stem <<< "${manifest_line}"
    else
        event_onset="saccade"
        IFS=$'\t' read -r \
            task_id_field \
            label \
            group_name \
            solver_backend \
            formula_mode \
            event_eye \
            workers \
            chunk_index \
            chunk_count \
            start_group_index \
            end_group_index \
            fit_count \
            chunk_definition_file \
            results_file \
            status_stem <<< "${manifest_line}"
    fi

    write_state_file \
        "${status_stem}" \
        "running" \
        "Starting ${label} chunk ${chunk_index}/${chunk_count} with ${fit_count} fits."
    refresh_run_summary_locked

    on_interrupt() {
        write_state_file \
            "${status_stem}" \
            "failed" \
            "Interrupted while running ${label} chunk ${chunk_index}/${chunk_count}."
        refresh_run_summary_locked
        exit 143
    }

    trap on_interrupt INT TERM

    launcher_args=(
        "${LAUNCHER_SCRIPT}"
        "--group-definitions-file" "${chunk_definition_file}"
        "--results-file" "${results_file}"
        "--workers" "${workers}"
        "--solver-backend" "${solver_backend}"
        "--formula" "${formula_mode}"
        "--event-eye" "${event_eye}"
        "--event-onset" "${event_onset}"
    )

    if [[ -n "${PATH_PREFIX_FROM}" ]]; then
        launcher_args+=("--path-prefix-from" "${PATH_PREFIX_FROM}" "--path-prefix-to" "${PATH_PREFIX_TO}")
    fi

    if [[ "${SKIP_EXISTING}" == "1" ]]; then
        launcher_args+=("--skip-existing")
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        launcher_args+=("--dry-run")
    fi

    launcher_args+=("${group_name}")

    echo
    echo "=== Chunk task ${task_id} ==="
    echo "Array slot: ${array_slot}"
    echo "Label: ${label}"
    echo "Run directory: ${RUN_DIR}"
    echo "Chunk: ${chunk_index}/${chunk_count}"
    echo "Fits: ${fit_count} (${start_group_index}-${end_group_index})"
    echo "Definitions: ${chunk_definition_file}"
    echo "Solver: ${solver_backend}"
    echo "Formula: ${formula_mode}"
    echo "Event eye: ${event_eye}"
    echo "Event onset: ${event_onset}"
    echo "Workers: ${workers}"
    if [[ -n "${PATH_PREFIX_FROM}" ]]; then
        echo "Path remap: ${PATH_PREFIX_FROM} -> ${PATH_PREFIX_TO}"
    fi

    set +e
    JULIA_CONDAPKG_BACKEND="${JULIA_CONDAPKG_BACKEND_VALUE}" \
        "${JULIA_BIN}" \
        "${launcher_args[@]}"
    launcher_exit_code=$?
    set -e

    if (( launcher_exit_code == 0 )); then
        if [[ "${DRY_RUN}" == "1" ]]; then
            write_state_file \
                "${status_stem}" \
                "pending" \
                "Dry run finished for ${label} chunk ${chunk_index}/${chunk_count}; no models were written."
        else
            write_state_file \
                "${status_stem}" \
                "success" \
                "Finished ${label} chunk ${chunk_index}/${chunk_count}."
        fi
    else
        write_state_file \
            "${status_stem}" \
            "failed" \
            "Chunk ${chunk_index}/${chunk_count} exited with code ${launcher_exit_code}."
    fi

    refresh_run_summary_locked
    trap - INT TERM

    return "${launcher_exit_code}"
}

submitter_mode() {
    local timestamp run_name task_count
    local task_id solver_backend
    local -a remaining_task_ids manifest_args gpu_task_ids cpu_task_ids

    mkdir -p "${RUN_ROOT}"

    if [[ -z "${RUN_DIR}" ]]; then
        timestamp="$(date '+%Y%m%d_%H%M%S')"
        if [[ "${SOLVER_FILTER}" == "all" ]]; then
            run_name="${timestamp}_${GROUP_DEFINITIONS_MODE}_models${CHUNK_SIZE}"
        else
            run_name="${timestamp}_${GROUP_DEFINITIONS_MODE}_${SOLVER_FILTER}_models${CHUNK_SIZE}"
        fi
        RUN_DIR="${RUN_ROOT}/${run_name}"
    fi

    mkdir -p "${RUN_DIR}"
    mkdir -p "${RUN_DIR}/logs"

    if [[ ! -f "${RUN_DIR}/manifest.tsv" ]]; then
        echo "Creating chunk manifest in ${RUN_DIR}"
        manifest_args=(
            "${BUILD_MANIFEST_SCRIPT}"
            "${GROUP_DEFINITIONS_FILE}"
            "${RUN_DIR}"
            "${CHUNK_SIZE}"
            "${GPU_WORKERS_PER_CHUNK}"
            "${CPU_WORKERS_PER_CHUNK}"
        )
        if [[ -n "${PATH_PREFIX_FROM}" ]]; then
            manifest_args+=("${PATH_PREFIX_FROM}" "${PATH_PREFIX_TO}")
        fi
        task_count="$(
            JULIA_CONDAPKG_BACKEND="${JULIA_CONDAPKG_BACKEND_VALUE}" \
                "${JULIA_BIN}" \
                "${manifest_args[@]}"
        )"
        echo "Prepared ${task_count} chunk tasks."
    else
        echo "Reusing existing run directory: ${RUN_DIR}"
    fi

    write_run_documentation
    refresh_run_summary_locked

    mapfile -t remaining_task_ids < "${RUN_DIR}/remaining_task_ids.txt"

    if (( ${#remaining_task_ids[@]} == 0 )); then
        echo "Nothing left to submit. All chunk tasks are already complete."
        echo "Run directory: ${RUN_DIR}"
        echo "Progress: ${RUN_DIR}/progress_overview.txt"
        return 0
    fi

    echo "Run directory: ${RUN_DIR}"
    echo "Remaining chunk tasks: ${#remaining_task_ids[@]}"
    echo "Models per job: ${CHUNK_SIZE}"
    echo "Array concurrency: ${ARRAY_CONCURRENCY}"
    echo "Max tasks per submitted array: ${MAX_ARRAY_TASKS_PER_SUBMISSION}"
    echo "Solver filter: ${SOLVER_FILTER}"
    echo "Progress summary: ${RUN_DIR}/progress_overview.txt"

    if [[ "${SUBMIT_ARRAY}" != "1" ]]; then
        echo "SUBMIT_ARRAY=${SUBMIT_ARRAY}, so no sbatch submission was performed."
    fi

    gpu_task_ids=()
    cpu_task_ids=()
    for task_id in "${remaining_task_ids[@]}"; do
        solver_backend="$(manifest_solver_backend_for_task "${task_id}")"
        if [[ "${SOLVER_FILTER}" != "all" && "${solver_backend}" != "${SOLVER_FILTER}" ]]; then
            continue
        fi
        case "${solver_backend}" in
            gpu)
                gpu_task_ids+=("${task_id}")
                ;;
            cpu)
                cpu_task_ids+=("${task_id}")
                ;;
            *)
                echo "Unsupported or missing solver backend '${solver_backend}' for task ${task_id}." >&2
                exit 1
                ;;
        esac
    done

    if (( ${#gpu_task_ids[@]} == 0 && ${#cpu_task_ids[@]} == 0 )); then
        echo "No remaining chunk tasks matched SOLVER_FILTER=${SOLVER_FILTER}."
        echo "Run directory: ${RUN_DIR}"
        echo "Progress: ${RUN_DIR}/progress_overview.txt"
        return 0
    fi

    submit_batches_for_task_group \
        "gpu" \
        "${GPU_PARTITION}" \
        "${GPU_CPUS_PER_TASK}" \
        "${GPU_MEMORY_PER_TASK}" \
        "${GPU_TIME_LIMIT}" \
        "${GPU_RESOURCES}" \
        "${gpu_task_ids[@]}"

    submit_batches_for_task_group \
        "cpu" \
        "${CPU_PARTITION}" \
        "${CPU_CPUS_PER_TASK}" \
        "${CPU_MEMORY_PER_TASK}" \
        "${CPU_TIME_LIMIT}" \
        "" \
        "${cpu_task_ids[@]}"

    echo "Track progress with: ${RUN_DIR}/progress_overview.txt"
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    worker_mode "$@"
else
    submitter_mode "$@"
fi
