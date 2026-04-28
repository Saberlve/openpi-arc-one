from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_start_arm_uses_joint_control_launch():
    script = (ROOT / "scripts" / "start_arx_arm.sh").read_text()

    assert "v2_joint_control.launch.py" in script
    assert "v2_collect.launch.py" not in script


def test_infer_client_sources_ros_workspace_setup():
    script = (ROOT / "scripts" / "infer_arx_client.sh").read_text()

    assert "source /opt/ros/humble/setup.bash" in script
    assert 'source "${ROS2_ROOT}/install/setup.bash"' in script


def test_infer_client_disables_nounset_while_sourcing_ros_setup():
    script = (ROOT / "scripts" / "infer_arx_client.sh").read_text()

    source_ros = script.index("source /opt/ros/humble/setup.bash")
    source_workspace = script.index('source "${ROS2_ROOT}/install/setup.bash"')
    restore_nounset = script.index("set -u", source_workspace)

    assert script.rindex("set +u", 0, source_ros) < source_ros
    assert source_ros < source_workspace < restore_nounset


def test_infer_client_uses_writable_runtime_cache_dirs():
    script = (ROOT / "scripts" / "infer_arx_client.sh").read_text()

    assert 'ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/openpi-arx-ros-log}"' in script
    assert 'MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/openpi-arx-matplotlib}"' in script
    assert 'mkdir -p "${ROS_LOG_DIR}" "${MPLCONFIGDIR}"' in script


def test_infer_client_passes_repo_arx_config_to_robot():
    script = (ROOT / "scripts" / "infer_arx_client.sh").read_text()

    assert 'ARX_CONFIG="${PROJECT_ROOT}/third_party/ARX-ONE/act/data/config.yaml"' in script
    assert '--robot.data="${ARX_CONFIG}"' in script


def test_infer_client_adds_openpi_client_package_to_pythonpath():
    script = (ROOT / "scripts" / "infer_arx_client.sh").read_text()

    assert 'OPENPI_CLIENT_PY="${PROJECT_ROOT}/packages/openpi-client/src"' in script
    assert 'export PYTHONPATH="${OPENPI_CLIENT_PY}:${ARX_MSGS_PY}:${PYTHONPATH:-}"' in script


def test_infer_client_exports_default_action_safety_limits():
    script = (ROOT / "scripts" / "infer_arx_client.sh").read_text()

    assert 'ARX_MAX_SAFE_JOINT_STEP="${ARX_MAX_SAFE_JOINT_STEP:-0.03}"' in script
    assert 'ARX_MAX_SAFE_GRIPPER_STEP="${ARX_MAX_SAFE_GRIPPER_STEP:-0.8}"' in script
    assert "export ARX_MAX_SAFE_JOINT_STEP ARX_MAX_SAFE_GRIPPER_STEP" in script


def test_arx_config_reads_slave_feedback_and_publishes_master_commands():
    config = yaml.safe_load((ROOT / "third_party" / "ARX-ONE" / "act" / "data" / "config.yaml").read_text())

    assert config["arm_config"]["controller_left_topic"] == "/arm_master_l_status"
    assert config["arm_config"]["controller_right_topic"] == "/arm_master_r_status"
    assert config["arm_config"]["feedback_left_topic"] == "/arm_slave_l_status"
    assert config["arm_config"]["feedback_right_topic"] == "/arm_slave_r_status"


def test_arx_config_uses_published_color_image_topics():
    config = yaml.safe_load((ROOT / "third_party" / "ARX-ONE" / "act" / "data" / "config.yaml").read_text())

    assert config["camera_config"]["original_image"]["img_head_topic"] == "/camera/camera_h/color/image_raw"
    assert config["camera_config"]["original_image"]["img_left_topic"] == "/camera/camera_l/color/image_raw"
    assert config["camera_config"]["original_image"]["img_right_topic"] == "/camera/camera_r/color/image_raw"
    assert config["camera_config"]["compress_image"]["img_head_topic"] == "/camera/camera_h/color/image_raw/compressed"
    assert config["camera_config"]["compress_image"]["img_left_topic"] == "/camera/camera_l/color/image_raw/compressed"
    assert config["camera_config"]["compress_image"]["img_right_topic"] == "/camera/camera_r/color/image_raw/compressed"
