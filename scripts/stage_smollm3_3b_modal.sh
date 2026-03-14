#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_modal_stage_model_rl.sh" \
  HuggingFaceTB/SmolLM3-3B-Base \
  smollm3-3b-base
