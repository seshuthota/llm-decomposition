#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

RUN_ID="${1:-P2R03}"
MANIFEST="${2:-configs/phase2/phase2_matched_frontier_manifest.json}"
MODEL_SUBPATH="${3:-qwen3-0.6b-base}"

cd "${REPO_ROOT}"

echo "Repo root: ${REPO_ROOT}"
echo "Conda env: rl"
echo "Run id: ${RUN_ID}"
echo "Manifest: ${MANIFEST}"
echo "Modal GPU: ${MODAL_GPU:-T4}"
echo "Modal model volume: ${MODAL_MODEL_VOLUME:-llm-decomposition-models}"
echo "Modal model subpath: ${MODEL_SUBPATH}"
echo "Command: modal run scripts/modal_experiment.py --run-id ${RUN_ID} --manifest ${MANIFEST} --model-subpath ${MODEL_SUBPATH}"

modal run scripts/modal_experiment.py --run-id "${RUN_ID}" --manifest "${MANIFEST}" --model-subpath "${MODEL_SUBPATH}"
