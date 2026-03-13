#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_PATH="${1:-/kaggle/working/llm-decomposition-results.zip}"

cd "${REPO_ROOT}"
zip -r "${OUT_PATH}" results docs README.md >/dev/null
echo "Wrote ${OUT_PATH}"

