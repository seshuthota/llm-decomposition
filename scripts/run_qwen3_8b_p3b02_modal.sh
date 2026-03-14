#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_modal_experiment_detached_rl.sh" \
  P3B02_Q8B \
  configs/scaleup_qwen3_8b/qwen3_8b_transfer_manifest.json \
  qwen3-8b-base
