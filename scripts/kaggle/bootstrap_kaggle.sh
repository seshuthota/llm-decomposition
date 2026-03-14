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
export PIP_PREFER_BINARY=1

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" /kaggle/working/results

python -m pip install --upgrade pip setuptools wheel

case "${MODE}" in
  rtn)
    python -m pip install -r "${REPO_ROOT}/requirements-kaggle-rtn.txt"
    ;;
  gptq)
    python -m pip install -r "${REPO_ROOT}/requirements-kaggle-gptq.txt"
    GPTQMODEL_WHEEL_URL="${GPTQMODEL_WHEEL_URL:-https://github.com/ModelCloud/GPTQModel/releases/download/v5.7.0/gptqmodel-5.7.0+cu126torch2.9-cp312-cp312-linux_x86_64.whl}"
    python -m pip install --no-deps "${GPTQMODEL_WHEEL_URL}"
    python - <<'PY'
import sys
import torch
import transformers
import optimum
import gptqmodel
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("optimum:", optimum.__version__)
print("gptqmodel:", getattr(gptqmodel, "__version__", "unknown"))
PY
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use 'rtn' or 'gptq'."
    exit 1
    ;;
esac

echo "Kaggle bootstrap complete."
echo "Mode: ${MODE}"
echo "HF_HOME: ${HF_HOME}"
