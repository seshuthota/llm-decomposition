#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/results/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/phase1_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

source "${SCRIPT_DIR}/common_env.sh"
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl

echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_FILE}"
echo "Conda env: ${CONDA_DEFAULT_ENV}" | tee -a "${LOG_FILE}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF token: loaded from .env" | tee -a "${LOG_FILE}"
fi
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Started at: $(date)" | tee -a "${LOG_FILE}"

cd "${REPO_ROOT}"
PYTHON_ENTRYPOINT="scripts/run_phase1.py"
for arg in "$@"; do
  if [[ "${arg}" == "--manifest" ]]; then
    PYTHON_ENTRYPOINT="scripts/run_manifest.py"
    break
  fi
done

echo "Command: python ${PYTHON_ENTRYPOINT} $*" | tee -a "${LOG_FILE}"
python "${PYTHON_ENTRYPOINT}" "$@" 2>&1 | tee -a "${LOG_FILE}"

echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
echo "Saved log to ${LOG_FILE}"
