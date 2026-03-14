#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${SCRIPT_DIR}/run_manifest_on_gpu.sh" \
  "${GPU_ID}" \
  "configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_smoke_manifest.json" \
  "R3S_Q17B"

