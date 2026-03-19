#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl

JOB_CONFIG="${1:-configs/latency/qwen3_8b_gptq/g2b02_q8b_bs1.json}"
DETACH_MODE="${MODAL_DETACH:-1}"
POLL_SECONDS="${MODAL_POLL_SECONDS:-60}"
RESULTS_VOLUME_NAME="${MODAL_RESULTS_VOLUME:-llm-decomposition-results}"

cd "${REPO_ROOT}"

readarray -t JOB_VALUES < <(
python - "${JOB_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

job_path = Path(sys.argv[1])
job = json.loads(job_path.read_text(encoding="utf-8"))
for key in (
    "job_id",
    "source_run_id",
    "source_manifest",
    "model_subpath",
    "results_prefix",
    "gpu",
    "batch_size",
    "prompt_length",
    "decode_length",
    "warmup_iterations",
    "timed_iterations",
):
    print(job[key])
print(job.get("prompt_template", ""))
PY
)

JOB_ID="${JOB_VALUES[0]}"
RUN_ID="${JOB_VALUES[1]}"
MANIFEST="${JOB_VALUES[2]}"
MODEL_SUBPATH="${JOB_VALUES[3]}"
RESULTS_PREFIX="${JOB_VALUES[4]}"
JOB_GPU="${JOB_VALUES[5]}"
BATCH_SIZE="${JOB_VALUES[6]}"
PROMPT_LENGTH="${JOB_VALUES[7]}"
DECODE_LENGTH="${JOB_VALUES[8]}"
WARMUP_ITERATIONS="${JOB_VALUES[9]}"
TIMED_ITERATIONS="${JOB_VALUES[10]}"
PROMPT_TEMPLATE="${JOB_VALUES[11]}"

RESULTS_DIR="$(
python - "${MANIFEST}" "${RUN_ID}" "${RESULTS_PREFIX}" "${BATCH_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
run_id = sys.argv[2]
results_prefix = Path(sys.argv[3])
batch_size = sys.argv[4]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for rel in manifest["runs"]:
    candidate = Path(rel)
    cfg_path = candidate if candidate.exists() else manifest_path.parent / rel
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg["run_id"] != run_id:
        continue
    phase = cfg.get("phase", "adhoc")
    print(results_prefix / phase / f"{run_id}__bs{batch_size}")
    break
else:
    raise SystemExit(f"run id {run_id!r} not found in {manifest_path}")
PY
)"

echo "Repo root: ${REPO_ROOT}"
echo "Latency job: ${JOB_ID}"
echo "Conda env: rl"
echo "Run id: ${RUN_ID}"
echo "Manifest: ${MANIFEST}"
echo "Modal GPU: ${JOB_GPU}"
echo "Modal model volume: ${MODAL_MODEL_VOLUME:-llm-decomposition-models}"
echo "Modal results volume: ${RESULTS_VOLUME_NAME}"
echo "Modal model subpath: ${MODEL_SUBPATH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Prompt length: ${PROMPT_LENGTH}"
echo "Decode length: ${DECODE_LENGTH}"
echo "Warmup iterations: ${WARMUP_ITERATIONS}"
echo "Timed iterations: ${TIMED_ITERATIONS}"
echo "Results dir: ${RESULTS_DIR}"
echo "Detach mode: ${DETACH_MODE}"
echo "Command target: scripts/modal_experiment_gptq.py::run_latency_remote"

fetch_results() {
  local parent_dir
  local run_dir_name
  local staging_root
  local staging_run_dir

  parent_dir="$(dirname "${RESULTS_DIR}")"
  run_dir_name="$(basename "${RESULTS_DIR}")"
  staging_root=".modal_fetch/${JOB_ID}"
  staging_run_dir="${staging_root}/${run_dir_name}"

  echo "Fetching results from ${RESULTS_VOLUME_NAME}:/$(printf '%s' "${RESULTS_DIR}")"
  mkdir -p "${staging_root}"
  mkdir -p "${parent_dir}"

  modal volume get "${RESULTS_VOLUME_NAME}" "/${RESULTS_DIR}" "${staging_root}" --force || return 0

  if [ -d "${staging_run_dir}" ]; then
    rm -rf "${RESULTS_DIR}"
    mv "${staging_run_dir}" "${RESULTS_DIR}"
    echo "Wrote results into ${RESULTS_DIR}"
  else
    echo "No fetched result directory found for ${JOB_ID}"
  fi
}

run_command=(
  modal run
)

if [ "${DETACH_MODE}" = "1" ]; then
  run_command+=(-d)
fi

run_command+=(
  scripts/modal_experiment_gptq.py::run_latency_remote
  --run-id "${RUN_ID}"
  --manifest "${MANIFEST}"
  --batch-size "${BATCH_SIZE}"
  --model-subpath "${MODEL_SUBPATH}"
  --prompt-length "${PROMPT_LENGTH}"
  --decode-length "${DECODE_LENGTH}"
  --warmup-iterations "${WARMUP_ITERATIONS}"
  --timed-iterations "${TIMED_ITERATIONS}"
  --results-prefix "${RESULTS_PREFIX}"
)

if [ -n "${PROMPT_TEMPLATE}" ]; then
  run_command+=(--prompt-template "${PROMPT_TEMPLATE}")
fi

if [ "${DETACH_MODE}" = "1" ]; then
  RUN_OUTPUT="$(
    MODAL_GPU="${JOB_GPU}" "${run_command[@]}" 2>&1
  )"
  MODAL_EXIT_CODE=$?
  printf '%s\n' "${RUN_OUTPUT}"
  APP_ID="$(printf '%s\n' "${RUN_OUTPUT}" | grep -o 'ap-[A-Za-z0-9]\+' | tail -n 1 || true)"
  if [ -z "${APP_ID}" ]; then
    echo "Failed to determine Modal app id for detached latency run"
    exit 1
  fi

  echo "Detached app id: ${APP_ID}"
  echo "Polling every ${POLL_SECONDS}s until the app stops"
  while true; do
    APP_LINE="$(modal app list | grep "${APP_ID}" || true)"
    if [ -z "${APP_LINE}" ]; then
      echo "App ${APP_ID} not yet visible in app list"
      sleep "${POLL_SECONDS}"
      continue
    fi
    echo "${APP_LINE}"
    if printf '%s\n' "${APP_LINE}" | grep -q 'stopped'; then
      break
    fi
    sleep "${POLL_SECONDS}"
  done

  fetch_results
else
  MODAL_GPU="${JOB_GPU}" "${run_command[@]}"
  MODAL_EXIT_CODE=$?
  fetch_results
fi

if [ -f "${RESULTS_DIR}/latency_benchmark.json" ]; then
  echo "Wrote results into ${RESULTS_DIR}"
else
  echo "No latency benchmark artifact written for ${JOB_ID}"
fi

exit "${MODAL_EXIT_CODE}"
