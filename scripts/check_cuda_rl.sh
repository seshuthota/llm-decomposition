#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/results/diagnostics"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/cuda_rl_check_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

run_section() {
  local title="$1"
  shift

  echo
  echo "===== ${title} =====" | tee -a "${LOG_FILE}"
  echo "COMMAND: $*" | tee -a "${LOG_FILE}"
  "$@" 2>&1 | tee -a "${LOG_FILE}"
}

run_shell_section() {
  local title="$1"
  local command="$2"

  echo
  echo "===== ${title} =====" | tee -a "${LOG_FILE}"
  echo "COMMAND: ${command}" | tee -a "${LOG_FILE}"
  bash -lc "${command}" 2>&1 | tee -a "${LOG_FILE}"
}

echo "Writing CUDA diagnostics to ${LOG_FILE}"
echo "CUDA diagnostics started at $(date)" | tee -a "${LOG_FILE}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_FILE}"

run_section "System Info" uname -a
run_shell_section "Environment Variables" "echo CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>} && echo CONDA_DEFAULT_ENV=\${CONDA_DEFAULT_ENV:-<unset>} && echo PATH=\$PATH"
run_shell_section "NVIDIA Device Nodes" "ls -l /dev/nvidia* 2>/dev/null || echo no_nvidia_devices"
run_shell_section "NVIDIA Driver Version" "cat /proc/driver/nvidia/version 2>/dev/null || echo no_nvidia_driver_proc"
run_shell_section "Loaded NVIDIA Kernel Modules" "lsmod | grep -i nvidia || echo no_nvidia_modules_listed"
run_shell_section "nvidia-smi -L" "nvidia-smi -L || true"
run_shell_section "nvidia-smi" "nvidia-smi || true"
run_shell_section "nvcc --version" "command -v nvcc >/dev/null && nvcc --version || echo no_nvcc"
run_shell_section "Conda Envs" "conda env list || true"
run_shell_section "Python In rl Env" "conda run -n rl python -V"
run_shell_section "Torch/CUDA Probe In rl Env" "conda run -n rl python - <<'PY'
import importlib.util

mods = ['torch', 'transformers', 'datasets', 'accelerate', 'bitsandbytes', 'optimum', 'auto_gptq']
for name in mods:
    print(f'{name}:', bool(importlib.util.find_spec(name)))

import torch
print('torch_version:', torch.__version__)
print('cuda_compiled_version:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
print('cudnn_available:', torch.backends.cudnn.is_available())

if torch.cuda.is_available():
    print('current_device:', torch.cuda.current_device())
    print('device_name:', torch.cuda.get_device_name(0))
    x = torch.randn(4, device='cuda')
    y = x * 2
    print('cuda_tensor_ok:', y.tolist())
else:
    print('current_device:', None)
    print('device_name:', None)
PY"
run_shell_section "Torch CUDA Collect Env In rl Env" "conda run -n rl python -m torch.utils.collect_env || true"

echo
echo "CUDA diagnostics completed at $(date)" | tee -a "${LOG_FILE}"
echo "Log saved to ${LOG_FILE}"
