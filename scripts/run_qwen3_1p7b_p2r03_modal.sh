#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

exec "${REPO_ROOT}/scripts/run_modal_experiment_rl.sh" \
  P2R03_Q17B \
  configs/scaleup_1p7b/qwen3_1p7b_transfer_manifest.json \
  qwen3-1.7b-base
