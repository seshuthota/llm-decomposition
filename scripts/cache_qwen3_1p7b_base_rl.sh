#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

cd "${REPO_ROOT}"

python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="Qwen/Qwen3-1.7B-Base")
print(path)
PY
