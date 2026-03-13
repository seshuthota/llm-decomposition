#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl

VOLUME_NAME="${MODAL_MODEL_VOLUME:-llm-decomposition-models}"
LOCAL_MODEL_DIR="${1:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd}"
REMOTE_SUBPATH="${2:-/qwen3-0.6b-base}"

cd "${REPO_ROOT}"

echo "Repo root: ${REPO_ROOT}"
echo "Conda env: rl"
echo "Modal model volume: ${VOLUME_NAME}"
echo "Local model dir: ${LOCAL_MODEL_DIR}"
echo "Remote model path inside volume root ${REMOTE_SUBPATH}"

modal volume create "${VOLUME_NAME}" || true
modal volume put "${VOLUME_NAME}" "${LOCAL_MODEL_DIR}" "${REMOTE_SUBPATH}"

echo "Upload complete."
echo "Use model_subpath='${REMOTE_SUBPATH#/}' when running scripts/modal_experiment.py."
