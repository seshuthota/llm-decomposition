#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_modal_experiment_gptq_rl.sh" \
  R3_S3B \
  configs/scaleup_smollm3_3b_gptq/smollm3_3b_gptq_baselines_manifest.json \
  smollm3-3b-base
