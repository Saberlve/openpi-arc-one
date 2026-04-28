#!/usr/bin/env bash
set -euo pipefail

# OpenPi Inference Client for ARX-ONE real robot
# Connects to a remote OpenPi policy server and runs inference on the robot.
#
# Usage:
#   OPENPI_HOST=192.168.1.10 TASK_PROMPT="your task" bash scripts/infer_arx_client.sh
#
# Environment variables:
#   OPENPI_HOST         - Policy server IP (default: localhost)
#   OPENPI_PORT         - Policy server port (default: 8000)
#   TASK_PROMPT         - Task description (default: see below)
#   ROBOT_TYPE          - Robot type (default: ACone)
#   FRAME_RATE          - Control loop frequency (default: 20)
#   HEAD_CAMERA         - Head camera device path (default: /dev/video10)
#   LEFT_WRIST_CAMERA   - Left wrist camera device path (default: /dev/video11)
#   RIGHT_WRIST_CAMERA  - Right wrist camera device path (default: /dev/video12)
#   ARX_MAX_SAFE_JOINT_STEP   - Max allowed per-step joint jump in radians (default: 0.03)
#   ARX_MAX_SAFE_GRIPPER_STEP - Max allowed per-step gripper jump (default: 0.8)
#   OPENPI_DEBUG_DIR    - Directory for saved observation images (default: ./openpi-arx-debug)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARX_SCRIPTS="${PROJECT_ROOT}/third_party/ARX-ONE/src/edlsrobot/scripts"
INFERENCE_PY="${ARX_SCRIPTS}/inference.py"
ARX_CONFIG="${PROJECT_ROOT}/third_party/ARX-ONE/act/data/config.yaml"
ARX_ROOT="${PROJECT_ROOT}/third_party/ARX-ONE"
ARX_X5_ROOT="${ARX_X5_ROOT:-${ARX_ROOT}/../ARX_X5}"

if [[ ! -d "${ARX_X5_ROOT}" ]] && [[ -d "/home/ubuntu/edlsrobot/repos/ARX_X5" ]]; then
    ARX_X5_ROOT="/home/ubuntu/edlsrobot/repos/ARX_X5"
fi

ROS2_ROOT="${ARX_X5_ROOT}/ROS2/X5_ws"

OPENPI_HOST="${OPENPI_HOST:-localhost}"
OPENPI_PORT="${OPENPI_PORT:-8000}"
ROBOT_TYPE="${ROBOT_TYPE:-ACone}"
ROBOT_ID="${ROBOT_ID:-my_ACone_arm}"
FRAME_RATE="${FRAME_RATE:-20}"
DATASET_REPO_ID="${DATASET_REPO_ID:-eval/eval_ACone_openpi_pi05}"
TASK_PROMPT="${TASK_PROMPT:-Pick up the black pouch three times, then touch the green grommet}"

HEAD_CAMERA="${HEAD_CAMERA:-/dev/video10}"
LEFT_WRIST_CAMERA="${LEFT_WRIST_CAMERA:-/dev/video11}"
RIGHT_WRIST_CAMERA="${RIGHT_WRIST_CAMERA:-/dev/video12}"
ARX_MAX_SAFE_JOINT_STEP="${ARX_MAX_SAFE_JOINT_STEP:-0.03}"
ARX_MAX_SAFE_GRIPPER_STEP="${ARX_MAX_SAFE_GRIPPER_STEP:-0.8}"
OPENPI_DEBUG_IO="${OPENPI_DEBUG_IO:-1}"
OPENPI_DEBUG_DIR="${OPENPI_DEBUG_DIR:-${PROJECT_ROOT}/openpi-arx-debug}"
ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/openpi-arx-ros-log}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/openpi-arx-matplotlib}"
export ROS_LOG_DIR MPLCONFIGDIR
export ARX_MAX_SAFE_JOINT_STEP ARX_MAX_SAFE_GRIPPER_STEP
export OPENPI_DEBUG_IO OPENPI_DEBUG_DIR
mkdir -p "${ROS_LOG_DIR}" "${MPLCONFIGDIR}" "${OPENPI_DEBUG_DIR}"

CONDA_ACT_PYTHON="/home/ubuntu/Package/miniconda/envs/act/bin/python"

if [[ ! -f "${ROS2_ROOT}/install/setup.bash" ]]; then
    echo "[ERROR] ROS2 workspace setup not found: ${ROS2_ROOT}/install/setup.bash" >&2
    exit 1
fi

ARX_MSGS_LIB="${ROS2_ROOT}/install/arx5_arm_msg/lib"
ARX_MSGS_PY="${ROS2_ROOT}/install/arx5_arm_msg/local/lib/python3.10/dist-packages"
OPENPI_CLIENT_PY="${PROJECT_ROOT}/packages/openpi-client/src"
# ROS/ament setup scripts read optional variables that may be unset.
set +u
source /opt/ros/humble/setup.bash
source "${ROS2_ROOT}/install/setup.bash"
set -u
export LD_LIBRARY_PATH="${ARX_MSGS_LIB}:/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${OPENPI_CLIENT_PY}:${ARX_MSGS_PY}:${PYTHONPATH:-}"

echo "============================================"
echo "  OpenPi Inference Client (ARX-ONE)"
echo "============================================"
echo "  Server:      ${OPENPI_HOST}:${OPENPI_PORT}"
echo "  Robot:       ${ROBOT_TYPE}"
echo "  Frame rate:  ${FRAME_RATE}"
echo "  Joint step:  ${ARX_MAX_SAFE_JOINT_STEP}"
echo "  Grip step:   ${ARX_MAX_SAFE_GRIPPER_STEP}"
echo "  Debug IO:    ${OPENPI_DEBUG_IO}"
echo "  Debug dir:   ${OPENPI_DEBUG_DIR}"
echo "  ROS config:  ${ARX_CONFIG}"
echo "  Task:        ${TASK_PROMPT}"
echo "============================================"

if [[ ! -f "${INFERENCE_PY}" ]]; then
    echo "[ERROR] inference.py not found: ${INFERENCE_PY}" >&2
    exit 1
fi
if [[ ! -f "${ARX_CONFIG}" ]]; then
    echo "[ERROR] ARX config not found: ${ARX_CONFIG}" >&2
    exit 1
fi

cd "${ARX_SCRIPTS}"

exec "${CONDA_ACT_PYTHON}" "${INFERENCE_PY}" \
    --robot.type="${ROBOT_TYPE}" \
    --robot.id="${ROBOT_ID}" \
    --robot.data="${ARX_CONFIG}" \
    --robot.frame_rate="${FRAME_RATE}" \
    --robot.cameras="{\"head\":{\"type\":\"opencv\",\"index_or_path\":\"${HEAD_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30},\"left_wrist\":{\"type\":\"opencv\",\"index_or_path\":\"${LEFT_WRIST_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30},\"right_wrist\":{\"type\":\"opencv\",\"index_or_path\":\"${RIGHT_WRIST_CAMERA}\",\"width\":640,\"height\":480,\"fps\":30}}" \
    --display_data=false \
    --use_openpi_server=true \
    --openpi_server_host="${OPENPI_HOST}" \
    --openpi_server_port="${OPENPI_PORT}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.single_task="${TASK_PROMPT}"
