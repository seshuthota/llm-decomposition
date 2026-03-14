#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_modal_experiment_gptq_rl.sh" \
  G2R02_Q8B \
  configs/scaleup_qwen3_8b_gptq/qwen3_8b_gptq_transfer_manifest.json \
  qwen3-8b-base
