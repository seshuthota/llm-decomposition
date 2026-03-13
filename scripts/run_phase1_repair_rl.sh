#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_phase1_rl.sh" --manifest configs/phase1/phase1_repair_manifest.json "$@"
