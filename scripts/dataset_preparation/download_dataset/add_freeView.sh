#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./add_freeView.sh mergedDataset
#   ./add_freeView.sh --remove mergedDataset

TASKS=(symbolSearch FunwithFractals ThePresent DiaryOfAWimpyKid DespicableMe)
SUFFIXES=(channels.tsv eeg.json eeg.set events.tsv)

REMOVE=0
if [[ "${1:-}" == "--remove" ]]; then
  REMOVE=1
  shift
fi

DATASET="${1:-}"
if [[ -z "${DATASET}" || ! -d "${DATASET}" ]]; then
  echo "Usage: $0 [--remove] <bids_dataset_dir>"
  exit 1
fi

relpath() {
  local from="$1"
  local to="$2"
  if command -v realpath >/dev/null 2>&1; then
    # GNU coreutils realpath supports --relative-to
    realpath --relative-to="$from" "$to"
  else
    python3 - "$from" "$to" <<'PY'
import os, sys
from_dir = sys.argv[1]
to_path = sys.argv[2]
print(os.path.relpath(to_path, start=from_dir))
PY
  fi
}

mapfile -t EEG_DIRS < <(find "${DATASET}" -type d -name eeg -path "*/sub-*/*" | sort)

if [[ ${#EEG_DIRS[@]} -eq 0 ]]; then
  echo "No eeg directories found under: ${DATASET}"
  exit 0
fi

for eeg_dir in "${EEG_DIRS[@]}"; do
  sub_id="$(basename "$(dirname "${eeg_dir}")")"
  if [[ "${sub_id}" != sub-* ]]; then
    sub_id="$(basename "$(dirname "$(dirname "${eeg_dir}")")")"
  fi
  [[ "${sub_id}" != sub-* ]] && continue

  if [[ "${REMOVE}" -eq 1 ]]; then
    shopt -s nullglob
    for suf in "${SUFFIXES[@]}"; do
      for f in "${eeg_dir}/${sub_id}_task-freeView_run-"*"_${suf}"; do
        [[ -L "${f}" ]] && rm -f "${f}"
      done
    done
    shopt -u nullglob
    continue
  fi

  run_idx=1
  for task in "${TASKS[@]}"; do
    # Require all 4 files for the source task/run-1
    src_files=()
    ok=1
    for suf in "${SUFFIXES[@]}"; do
      src="${eeg_dir}/${sub_id}_task-${task}_run-1_${suf}"
      if [[ ! -e "${src}" ]]; then
        ok=0
        break
      fi
      src_files+=("${src}")
    done
    [[ "${ok}" -eq 0 ]] && continue

    for i in "${!SUFFIXES[@]}"; do
      suf="${SUFFIXES[$i]}"
      src="${src_files[$i]}"
      dst="${eeg_dir}/${sub_id}_task-freeView_run-${run_idx}_${suf}"

      target_rel="$(relpath "${eeg_dir}" "${src}")"

      if [[ -L "${dst}" ]]; then
        cur_target="$(readlink "${dst}")"
        [[ "${cur_target}" == "${target_rel}" ]] && continue
        rm -f "${dst}"
      elif [[ -e "${dst}" ]]; then
        rm -f "${dst}"
      fi

      ln -s "${target_rel}" "${dst}"
    done

    run_idx=$((run_idx + 1))
  done
done

echo "Done."
