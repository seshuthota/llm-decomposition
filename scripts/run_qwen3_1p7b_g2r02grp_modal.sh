#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_modal_experiment_gptq_rl.sh" \
  G2R02GRP_Q17B \
  configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_grouped_rank_manifest.json \
  qwen3-1.7b-base
