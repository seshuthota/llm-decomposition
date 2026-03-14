#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_modal_experiment_detached_rl.sh" \
  P3B03_S3B \
  configs/scaleup_smollm3_3b/smollm3_3b_transfer_manifest.json \
  smollm3-3b-base
