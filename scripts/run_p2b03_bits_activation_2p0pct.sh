#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/results/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/phase2_${TIMESTAMP}.log"

source "$REPO_ROOT/scripts/common_env.sh"
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl

cd "$REPO_ROOT"

{
  echo "Repo root: $REPO_ROOT"
  echo "Conda env: rl"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF token: loaded from .env"
  fi
  echo "Log file: $LOG_FILE"
  echo "Started at: $(date)"
  echo "Command: python scripts/run_phase2.py --manifest configs/phase2/phase2_matched_frontier_manifest.json --run-id P2B03"
  python scripts/run_phase2.py --manifest configs/phase2/phase2_matched_frontier_manifest.json --run-id P2B03
  echo "Finished at: $(date)"
  echo "Saved log to $LOG_FILE"
} | tee "$LOG_FILE"
