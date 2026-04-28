#!/usr/bin/env bash
set -euo pipefail

# OpenPi Policy Server for ARX-ONE real robot
# Usage: bash scripts/serve_arx.sh [checkpoint_dir] [port]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_DIR="/home/ubuntu/openpi-arc-one/ckpt/pi05_arx/pick"
PORT="${2:-8000}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Pick up the black pouch three times, then touch the green grommet}"
CONFIG_NAME="${CONFIG_NAME:-pi05_arx}"
OPENPI_DEBUG_IO="${OPENPI_DEBUG_IO:-1}"
OPENPI_DEBUG_DIR="${OPENPI_DEBUG_DIR:-${PROJECT_ROOT}/openpi-arx-debug}"
export OPENPI_DEBUG_IO OPENPI_DEBUG_DIR

echo "============================================"
echo "  OpenPi Policy Server (ARX-ONE)"
echo "============================================"
echo "  Config:      ${CONFIG_NAME}"
echo "  Checkpoint:  ${CHECKPOINT_DIR}"
echo "  Port:        ${PORT}"
echo "  Prompt:      ${DEFAULT_PROMPT}"
echo "  Debug IO:    ${OPENPI_DEBUG_IO}"
echo "  Debug dir:   ${OPENPI_DEBUG_DIR}"
echo "============================================"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "[ERROR] Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

exec uv run scripts/serve_policy.py \
    --port="${PORT}" \
    --default-prompt="${DEFAULT_PROMPT}" \
    policy:checkpoint \
    --policy.config="${CONFIG_NAME}" \
    --policy.dir="${CHECKPOINT_DIR}"
