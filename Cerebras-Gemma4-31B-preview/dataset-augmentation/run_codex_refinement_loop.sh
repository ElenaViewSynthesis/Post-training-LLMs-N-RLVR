#!/usr/bin/env bash
set -euo pipefail

# Resume one immutable Codex refinement run until its accepted-row target is met.
# The Python worker owns source identity, micro-batching, validation, atomic
# Parquet promotion, and the per-output lock. This wrapper only relaunches a
# clean worker invocation if its explicit call budget is exhausted.

umask 077

augmentation_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${CODEX_REFINEMENT_PYTHON:-${augmentation_dir}/.venv/bin/python}"
pipeline_data_dir="${PIPELINE_DATA_DIR:-/mnt/c/Users/proxi/pipeline/data}"
source_snapshot_dir="${CODEX_SOURCE_SNAPSHOT_DIR:-${pipeline_data_dir}/source-snapshot}"
refinement_output_dir="${CODEX_REFINEMENT_OUTPUT_DIR:-${pipeline_data_dir}/codex-refined}"
refinement_model="${CODEX_REFINEMENT_MODEL:-gpt-5.6-sol}"
target_rows="${CODEX_REFINEMENT_TARGET_ROWS:-150000}"
batch_size="${CODEX_REFINEMENT_BATCH_SIZE:-4}"
concurrency="${CODEX_REFINEMENT_CONCURRENCY:-1}"
calls_per_invocation="${CODEX_MAX_AGENT_CALLS_PER_INVOCATION:-100000}"
attempts_per_run="${CODEX_MAX_ATTEMPTS_PER_RUN:-3}"
timeout_seconds="${CODEX_REFINEMENT_TIMEOUT_SECONDS:-600}"

if [[ ! -x "${python_bin}" ]]; then
    echo "ERROR: Python environment is not executable: ${python_bin}" >&2
    exit 1
fi
if [[ ! -d "${source_snapshot_dir}" ]]; then
    echo "ERROR: sealed source snapshot is missing: ${source_snapshot_dir}" >&2
    exit 1
fi

while true; do
    set +e
    "${python_bin}" "${augmentation_dir}/codex_refinement_worker.py" \
        --data-dir "${pipeline_data_dir}" \
        --source-snapshot-dir "${source_snapshot_dir}" \
        --output-dir "${refinement_output_dir}" \
        --target-rows "${target_rows}" \
        --model "${refinement_model}" \
        --batch-size "${batch_size}" \
        --concurrency "${concurrency}" \
        --max-agent-calls "${calls_per_invocation}" \
        --max-attempts-per-run "${attempts_per_run}" \
        --timeout-seconds "${timeout_seconds}" \
        --execute
    worker_status=$?
    set -e

    if (( worker_status != 0 )); then
        echo "ERROR: Codex worker exited ${worker_status}; sealed state was preserved." >&2
        exit "${worker_status}"
    fi

    progress_path="${refinement_output_dir}/progress.json"
    if [[ ! -f "${progress_path}" ]]; then
        echo "ERROR: successful worker invocation produced no progress file." >&2
        exit 1
    fi
    progress_reader='import json, sys; '
    progress_reader+='value=json.load(open(sys.argv[1], encoding="utf-8")); '
    progress_reader+='print(value["completed_rows"], value["target_rows"])'
    read -r completed target < <(
        "${python_bin}" -c "${progress_reader}" "${progress_path}"
    )
    if (( completed >= target )); then
        echo "Codex refinement target reached: ${completed}/${target}."
        exit 0
    fi
    echo "Codex call budget exhausted at ${completed}/${target}; resuming."
done
