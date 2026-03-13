#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <gpu_id> <manifest> <run_id> [extra args ...]"
  exit 1
fi

GPU_ID="$1"
MANIFEST="$2"
RUN_ID="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HOME="${HF_HOME:-/kaggle/working/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

cd "${REPO_ROOT}"
python scripts/run_manifest.py --manifest "${MANIFEST}" --run-id "${RUN_ID}" "$@"

