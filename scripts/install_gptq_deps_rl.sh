#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/results/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/install_gptq_deps_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl

echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_FILE}"
echo "Conda env: ${CONDA_DEFAULT_ENV}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Started at: $(date)" | tee -a "${LOG_FILE}"

echo | tee -a "${LOG_FILE}"
echo "Installing GPTQ-related dependencies into env 'rl'..." | tee -a "${LOG_FILE}"

python -m pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${LOG_FILE}"
python -m pip install --upgrade accelerate optimum transformers 2>&1 | tee -a "${LOG_FILE}"
python -m pip install gptqmodel --no-build-isolation 2>&1 | tee -a "${LOG_FILE}"

echo | tee -a "${LOG_FILE}"
echo "Verifying imports..." | tee -a "${LOG_FILE}"
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import importlib.util

mods = ["torch", "transformers", "datasets", "accelerate", "optimum", "gptqmodel"]
for name in mods:
    print(f"{name}: {bool(importlib.util.find_spec(name))}")
PY

echo | tee -a "${LOG_FILE}"
echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
echo "Saved log to ${LOG_FILE}"
