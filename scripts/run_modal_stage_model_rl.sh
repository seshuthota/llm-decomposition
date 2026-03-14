#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

REPO_ID="${1:?repo id required}"
MODEL_SUBPATH="${2:?model subpath required}"

cd "${REPO_ROOT}"

echo "Repo root: ${REPO_ROOT}"
echo "Conda env: rl"
echo "Modal model volume: ${MODAL_MODEL_VOLUME:-llm-decomposition-models}"
echo "Repo id: ${REPO_ID}"
echo "Model subpath: ${MODEL_SUBPATH}"
echo "Command: modal run scripts/modal_stage_model.py --repo-id ${REPO_ID} --model-subpath ${MODEL_SUBPATH}"

modal run scripts/modal_stage_model.py --repo-id "${REPO_ID}" --model-subpath "${MODEL_SUBPATH}"
