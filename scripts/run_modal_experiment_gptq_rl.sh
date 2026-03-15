#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

RUN_ID="${1:-R3_Q17B}"
MANIFEST="${2:-configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_transfer_manifest.json}"
MODEL_SUBPATH="${3:-qwen3-1.7b-base}"
DETACH_MODE="${MODAL_DETACH:-1}"
POLL_SECONDS="${MODAL_POLL_SECONDS:-60}"
RESULTS_VOLUME_NAME="${MODAL_RESULTS_VOLUME:-llm-decomposition-results}"

cd "${REPO_ROOT}"

RESULTS_DIR="$(
python - "${MANIFEST}" "${RUN_ID}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
run_id = sys.argv[2]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for rel in manifest["runs"]:
    candidate = Path(rel)
    cfg_path = candidate if candidate.exists() else manifest_path.parent / rel
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg["run_id"] != run_id:
        continue
    phase = cfg.get("phase", "adhoc")
    print(Path("results/modal") / phase / run_id)
    break
else:
    raise SystemExit(f"run id {run_id!r} not found in {manifest_path}")
PY
)"

echo "Repo root: ${REPO_ROOT}"
echo "Conda env: rl"
echo "Run id: ${RUN_ID}"
echo "Manifest: ${MANIFEST}"
echo "Modal GPU: ${MODAL_GPU:-A100}"
echo "Modal model volume: ${MODAL_MODEL_VOLUME:-llm-decomposition-models}"
echo "Modal results volume: ${RESULTS_VOLUME_NAME}"
echo "Modal model subpath: ${MODEL_SUBPATH}"
echo "Detach mode: ${DETACH_MODE}"
echo "Command target: scripts/modal_experiment_gptq.py::run_config_remote"

fetch_results() {
  local parent_dir
  local run_dir_name
  local staging_root
  local staging_run_dir

  parent_dir="$(dirname "${RESULTS_DIR}")"
  run_dir_name="$(basename "${RESULTS_DIR}")"
  staging_root=".modal_fetch/${RUN_ID}"
  staging_run_dir="${staging_root}/${run_dir_name}"

  echo "Fetching results from ${RESULTS_VOLUME_NAME}:/$(printf '%s' "${RESULTS_DIR}")"
  rm -rf "${staging_root}"
  mkdir -p "${staging_root}"

  if [ -f "${parent_dir}" ]; then
    rm -f "${parent_dir}"
  fi
  mkdir -p "${parent_dir}"

  modal volume get "${RESULTS_VOLUME_NAME}" "/${RESULTS_DIR}" "${staging_root}" --force || return 0

  if [ -d "${staging_run_dir}" ]; then
    rm -rf "${RESULTS_DIR}"
    mv "${staging_run_dir}" "${RESULTS_DIR}"
    echo "Wrote results into ${RESULTS_DIR}"
  else
    echo "No fetched result directory found for ${RUN_ID}"
  fi
}

if [ "${DETACH_MODE}" = "1" ]; then
  RUN_OUTPUT="$(
    modal run --detach scripts/modal_experiment_gptq.py::run_config_remote \
      --run-id "${RUN_ID}" \
      --manifest "${MANIFEST}" \
      --model-subpath "${MODEL_SUBPATH}" \
      --results-prefix "results/modal" 2>&1
  )"
  MODAL_EXIT_CODE=$?
  printf '%s\n' "${RUN_OUTPUT}"
  APP_ID="$(printf '%s\n' "${RUN_OUTPUT}" | grep -o 'ap-[A-Za-z0-9]\+' | tail -n 1 || true)"
  if [ -z "${APP_ID}" ]; then
    echo "Failed to determine Modal app id for detached run"
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
  modal run scripts/modal_experiment_gptq.py::run_config_remote \
    --run-id "${RUN_ID}" \
    --manifest "${MANIFEST}" \
    --model-subpath "${MODEL_SUBPATH}" \
    --results-prefix "results/modal"
  MODAL_EXIT_CODE=$?
  fetch_results
fi

if [ -f "${RESULTS_DIR}/metrics.json" ]; then
  echo "Wrote results into ${RESULTS_DIR}"
else
  echo "No results written for ${RUN_ID}"
fi

exit "${MODAL_EXIT_CODE}"
