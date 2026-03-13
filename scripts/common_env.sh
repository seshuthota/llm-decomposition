#!/usr/bin/env bash

# Load repo-local environment variables for manual experiment runs.
# Intended to be sourced by other scripts.

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _COMMON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _COMMON_ENV_DIR="$(pwd)"
fi
_COMMON_ENV_REPO_ROOT="$(cd "${_COMMON_ENV_DIR}/.." && pwd)"
_COMMON_ENV_FILE="${_COMMON_ENV_REPO_ROOT}/.env"

if [[ -f "${_COMMON_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${_COMMON_ENV_FILE}"
  set +a
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi
