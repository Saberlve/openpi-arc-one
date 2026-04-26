#!/usr/bin/env bash
set -euo pipefail

# Launch JAX training for the pi05_arx config while reusing an existing OpenPI environment.
#
# Defaults:
# - Environment root: /VLA/openpi
# - Config: pi05_arx
# - Exp name: pick_X_times
# - XLA mem fraction: 0.9
# - overwrite enabled, resume disabled
#
# Example:
#   bash scripts/train_pi05_arx_jax.sh --exp-name my_run --overwrite
#
# Override env root if needed:
#   OPENPI_ENV_ROOT=/path/to/openpi bash scripts/train_pi05_arx_jax.sh --exp-name my_run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPENPI_ENV_ROOT="${OPENPI_ENV_ROOT:-/VLA/openpi}"
PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ENV_ROOT}/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-pi05_arx}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${HOME}/autodl-tmp/datasets/pick_X_times_filterd_twice}"
EXP_NAME_DEFAULT="${EXP_NAME_DEFAULT:-pick_X_times}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python not found or not executable: ${PYTHON_BIN}" >&2
    echo "        You can override with PYTHON_BIN=/absolute/path/to/python" >&2
    exit 1
fi

# Make sure this repo's code is imported first while still using /VLA/openpi environment packages.
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
else
    export PYTHONPATH="${PROJECT_ROOT}/src"
fi

# Recommended for JAX memory usage; can be overridden by caller.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# If user did not explicitly override data repo path, inject a sensible default for pi05_arx.
USER_SET_DATA_REPO_ID=0
for arg in "$@"; do
    case "${arg}" in
        --data.repo-id|--data.repo-id=*|--data.repo_id|--data.repo_id=*)
            USER_SET_DATA_REPO_ID=1
            break
            ;;
    esac
done

TRAIN_ARGS=("$@")
if [[ ${USER_SET_DATA_REPO_ID} -eq 0 ]]; then
    TRAIN_ARGS+=("--data.repo-id=${DATASET_REPO_ID}")
fi

USER_SET_EXP_NAME=0
for arg in "$@"; do
    case "${arg}" in
        --exp-name|--exp-name=*|--exp_name|--exp_name=*)
            USER_SET_EXP_NAME=1
            break
            ;;
    esac
done

if [[ ${USER_SET_EXP_NAME} -eq 0 ]]; then
    TRAIN_ARGS+=("--exp-name=${EXP_NAME_DEFAULT}")
fi

echo "[INFO] project root: ${PROJECT_ROOT}"
echo "[INFO] python:       ${PYTHON_BIN}"
echo "[INFO] config:       ${CONFIG_NAME}"
echo "[INFO] exp-name:     ${EXP_NAME_DEFAULT} (default if not overridden)"
echo "[INFO] dataset:      ${DATASET_REPO_ID}"
echo "[INFO] PYTHONPATH:   ${PYTHONPATH}"
echo "[INFO] XLA mem frac: ${XLA_PYTHON_CLIENT_MEM_FRACTION}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train.py" "${CONFIG_NAME}" "${TRAIN_ARGS[@]}" --overwrite 