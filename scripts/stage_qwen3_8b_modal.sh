#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_modal_stage_model_rl.sh" \
  Qwen/Qwen3-8B-Base \
  qwen3-8b-base
