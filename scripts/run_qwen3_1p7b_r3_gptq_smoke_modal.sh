#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_modal_experiment_gptq_rl.sh" \
  R3S_Q17B \
  configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_smoke_manifest.json \
  qwen3-1.7b-base
