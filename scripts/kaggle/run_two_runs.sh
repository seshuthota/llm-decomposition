#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <manifest_a> <run_id_a> <manifest_b> <run_id_b>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MANIFEST_A="$1"
RUN_A="$2"
MANIFEST_B="$3"
RUN_B="$4"

mkdir -p "${REPO_ROOT}/results/logs"

bash "${SCRIPT_DIR}/run_manifest_on_gpu.sh" 0 "${MANIFEST_A}" "${RUN_A}" \
  > "${REPO_ROOT}/results/logs/${RUN_A}_gpu0.log" 2>&1 &
PID_A=$!

bash "${SCRIPT_DIR}/run_manifest_on_gpu.sh" 1 "${MANIFEST_B}" "${RUN_B}" \
  > "${REPO_ROOT}/results/logs/${RUN_B}_gpu1.log" 2>&1 &
PID_B=$!

wait "${PID_A}"
wait "${PID_B}"

echo "Completed ${RUN_A} on GPU0 and ${RUN_B} on GPU1"

