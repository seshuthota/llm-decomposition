#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

RUN_ID="${1:?run id required}"
MANIFEST="${2:?manifest required}"
MODEL_SUBPATH="${3:?model subpath required}"
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
echo "Modal GPU: ${MODAL_GPU:-T4}"
echo "Modal CPU: ${MODAL_CPU:-8}"
echo "Modal model volume: ${MODAL_MODEL_VOLUME:-llm-decomposition-models}"
echo "Modal results volume: ${RESULTS_VOLUME_NAME}"
echo "Modal model subpath: ${MODEL_SUBPATH}"
echo "Detach mode: ${DETACH_MODE}"
echo "Command target: scripts/modal_experiment_detached.py::run_config_remote"

if [ "${DETACH_MODE}" = "1" ]; then
  RUN_OUTPUT="$(
    modal run --detach scripts/modal_experiment_detached.py::run_config_remote \
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

  echo "Fetching results from ${RESULTS_VOLUME_NAME}:/$(printf '%s' "${RESULTS_DIR}")"
  modal volume get "${RESULTS_VOLUME_NAME}" "/${RESULTS_DIR}" "$(dirname "${RESULTS_DIR}")" --force || true
else
  modal run scripts/modal_experiment_detached.py::run_config_remote \
    --run-id "${RUN_ID}" \
    --manifest "${MANIFEST}" \
    --model-subpath "${MODEL_SUBPATH}" \
    --results-prefix "results/modal"
  MODAL_EXIT_CODE=$?
  echo "Fetching results from ${RESULTS_VOLUME_NAME}:/$(printf '%s' "${RESULTS_DIR}")"
  modal volume get "${RESULTS_VOLUME_NAME}" "/${RESULTS_DIR}" "$(dirname "${RESULTS_DIR}")" --force || true
fi

if [ -f "${RESULTS_DIR}/metrics.json" ]; then
  echo "Wrote results into ${RESULTS_DIR}"
else
  echo "No results written for ${RUN_ID}"
fi

exit "${MODAL_EXIT_CODE}"
