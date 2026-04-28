#!/usr/bin/env bash
set -euo pipefail

# Start ARX-ONE robot hardware (CAN, ROS2 inference controller, joy, RealSense)
# Usage: bash scripts/start_arx_arm.sh
#
# This script launches the following in separate gnome-terminal windows:
#   - CAN buses (can1, can3, can6)
#   - ROS2 joint-control controller
#   - ROS2 joy node
#   - RealSense cameras
#
# After starting, press "2" on the joystick to reset to collection pose.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARX_ROOT="${PROJECT_ROOT}/third_party/ARX-ONE"

# Verify ARX-ONE directory exists
if [[ ! -d "${ARX_ROOT}" ]]; then
    echo "[ERROR] ARX-ONE directory not found: ${ARX_ROOT}" >&2
    exit 1
fi

# Paths (adjust these if your ARX_X5 repo is elsewhere)
ARX_X5_ROOT="${ARX_X5_ROOT:-${ARX_ROOT}/../ARX_X5}"

# Fallback to known location if default doesn't exist
if [[ ! -d "${ARX_X5_ROOT}" ]] && [[ -d "/home/ubuntu/edlsrobot/repos/ARX_X5" ]]; then
    ARX_X5_ROOT="/home/ubuntu/edlsrobot/repos/ARX_X5"
fi
CAN_ROOT="${ARX_X5_ROOT}/ARX_CAN/arx_can"
ROS2_ROOT="${ARX_X5_ROOT}/ROS2/X5_ws"
JOY_ROOT="${ARX_X5_ROOT}/arx_joy"
REALSENSE_ROOT="${ARX_ROOT}/realsense"

shell_type=${SHELL##*/}
if [[ -z "${shell_type}" ]]; then
    shell_type="bash"
fi

require_dir() {
    local path="$1"
    if [[ ! -d "${path}" ]]; then
        echo "[ERROR] Missing required directory: ${path}" >&2
        exit 1
    fi
}

launch_terminal() {
    local title="$1"
    local command="$2"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[DRY_RUN] ${title}: ${command}"
        return 0
    fi

    if [[ -z "${DISPLAY:-}" ]]; then
        echo "[ERROR] DISPLAY is not set. Run inside a graphical desktop terminal." >&2
        exit 1
    fi

    if ! xdpyinfo >/dev/null 2>&1; then
        echo "[ERROR] DISPLAY=${DISPLAY} is not reachable." >&2
        exit 1
    fi

    gnome-terminal --title="${title}" -- "${shell_type}" -lc "${command}; exec ${shell_type} -i"
}

# Validate required directories
require_dir "${CAN_ROOT}"
require_dir "${ROS2_ROOT}"
require_dir "${JOY_ROOT}"
require_dir "${REALSENSE_ROOT}"

echo "============================================"
echo "  Starting ARX-ONE Robot Hardware"
echo "============================================"
echo "  ARX_X5_ROOT: ${ARX_X5_ROOT}"
echo "  CAN_ROOT:    ${CAN_ROOT}"
echo "  ROS2_ROOT:   ${ROS2_ROOT}"
echo "  JOY_ROOT:    ${JOY_ROOT}"
echo "============================================"
echo ""
echo "  Launching CAN buses..."
launch_terminal "can1" "cd '${CAN_ROOT}' && ./arx_can1.sh"
sleep 0.3
launch_terminal "can3" "cd '${CAN_ROOT}' && ./arx_can3.sh"
sleep 0.3
launch_terminal "can6" "cd '${CAN_ROOT}' && ./arx_can6.sh"
sleep 0.3

echo "  Launching ROS2 joint-control controller..."
launch_terminal "lift" "cd '${ROS2_ROOT}' && source install/setup.bash && ros2 launch arx_x5_controller v2_joint_control.launch.py"
sleep 0.3

echo "  Launching joy node..."
launch_terminal "joy" "cd '${JOY_ROOT}' && source install/setup.bash && ros2 run arx_joy arx_joy"
sleep 1

echo "  Launching RealSense cameras..."
launch_terminal "realsense" "cd '${REALSENSE_ROOT}' && ./realsense.sh"
sleep 3

echo ""
echo "============================================"
echo "  All robot components launched!"
echo "============================================"
echo "  Next steps:"
echo "    1. Wait for all terminals to initialize"
echo "    2. Wait for /arm_slave_l_status and /arm_slave_r_status to publish"
echo "    3. Run: bash scripts/infer_arx_client.sh"
echo "============================================"
