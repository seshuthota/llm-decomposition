#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/results/diagnostics"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/torch_cuda_rl_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

run_shell_section() {
  local title="$1"
  local command="$2"

  echo
  echo "===== ${title} =====" | tee -a "${LOG_FILE}"
  echo "COMMAND: ${command}" | tee -a "${LOG_FILE}"
  bash -lc "${command}" 2>&1 | tee -a "${LOG_FILE}"
}

echo "Writing torch CUDA checks to ${LOG_FILE}"
echo "Torch CUDA checks started at $(date)" | tee -a "${LOG_FILE}"

run_shell_section "Conda Run Torch Probe" "conda run -n rl python - <<'PY'
import torch
print('torch_version:', torch.__version__)
print('cuda_compiled_version:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
print('current_device:', torch.cuda.current_device() if torch.cuda.is_available() else None)
print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if torch.cuda.is_available():
    x = torch.randn(4, device='cuda')
    y = x * 2
    print('cuda_tensor_ok:', y.tolist())
else:
    print('cuda_tensor_ok:', False)
PY"

run_shell_section "Conda Run Torch+Transformers Probe" "conda run -n rl python - <<'PY'
import torch
import transformers
print('transformers_version:', transformers.__version__)
print('cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY"

run_shell_section "Activated Env Torch Probe" "source /home/seshu/anaconda3/etc/profile.d/conda.sh && conda activate rl && python - <<'PY'
import torch
print('torch_version:', torch.__version__)
print('cuda_compiled_version:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
print('current_device:', torch.cuda.current_device() if torch.cuda.is_available() else None)
print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if torch.cuda.is_available():
    x = torch.randn(4, device='cuda')
    y = x * 2
    print('cuda_tensor_ok:', y.tolist())
else:
    print('cuda_tensor_ok:', False)
PY"

echo
echo "Torch CUDA checks completed at $(date)" | tee -a "${LOG_FILE}"
echo "Log saved to ${LOG_FILE}"
