#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

VOLUME_NAME="${MODAL_MODEL_VOLUME:-llm-decomposition-models}"
REMOTE_SUBPATH="${2:-/smollm3-3b-base}"

if [[ $# -ge 1 ]]; then
  LOCAL_MODEL_DIR="$1"
else
  LOCAL_MODEL_DIR="$(python - <<'PY'
from pathlib import Path
root = Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B-Base"
ref_path = root / "refs/main"
if not ref_path.exists():
    raise SystemExit(1)
snapshot = ref_path.read_text(encoding="utf-8").strip()
print((root / "snapshots" / snapshot).as_posix())
PY
)"
fi

if [[ ! -d "${LOCAL_MODEL_DIR}" ]]; then
  echo "Local model dir not found: ${LOCAL_MODEL_DIR}"
  echo "Run ./scripts/cache_smollm3_3b_base_rl.sh first, or pass the snapshot path explicitly."
  exit 1
fi

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
