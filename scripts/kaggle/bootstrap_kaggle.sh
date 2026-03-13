#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-rtn}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export HF_HOME="${HF_HOME:-/kaggle/working/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_NO_INPUT=1

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" /kaggle/working/results

python -m pip install --upgrade pip setuptools wheel

case "${MODE}" in
  rtn)
    python -m pip install -r "${REPO_ROOT}/requirements-kaggle-rtn.txt"
    ;;
  gptq)
    python -m pip install -r "${REPO_ROOT}/requirements-kaggle-rtn.txt"
    python -m pip install --no-build-isolation optimum==2.1.0 gptqmodel
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use 'rtn' or 'gptq'."
    exit 1
    ;;
esac

echo "Kaggle bootstrap complete."
echo "Mode: ${MODE}"
echo "HF_HOME: ${HF_HOME}"

