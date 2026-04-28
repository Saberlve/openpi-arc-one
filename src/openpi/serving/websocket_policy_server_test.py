import numpy as np

from openpi.serving import websocket_policy_server


def test_describe_debug_payload_summarizes_arrays_without_dumping_values():
    payload = {
        "observation/images/head": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/state": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        "prompt": "pick",
    }

    summary = websocket_policy_server.describe_debug_payload(payload)

    assert summary["observation/images/head"] == {
        "dtype": "uint8",
        "shape": [480, 640, 3],
        "min": 0.0,
        "max": 0.0,
    }
    assert summary["observation/state"] == {
        "dtype": "float32",
        "shape": [3],
        "values": [0.0, 1.0, 2.0],
    }
    assert summary["prompt"] == "pick"


def test_describe_debug_payload_truncates_large_vectors():
    summary = websocket_policy_server.describe_debug_payload({"actions": np.arange(20, dtype=np.float32)})

    assert summary["actions"]["shape"] == [20]
    assert summary["actions"]["values"] == [0.0, 1.0, 2.0, 3.0, 4.0, "...", 15.0, 16.0, 17.0, 18.0, 19.0]


def test_save_debug_observation_images_writes_camera_files(tmp_path):
    payload = {
        "observation/images/head": np.zeros((8, 10, 3), dtype=np.uint8),
        "observation/images/left_wrist": np.ones((8, 10, 3), dtype=np.uint8) * 255,
    }

    paths = websocket_policy_server.save_debug_observation_images(payload, tmp_path, prefix="server_obs", step=3)

    assert [path.name for path in paths] == [
        "server_obs_step_000003_head.jpg",
        "server_obs_step_000003_left_wrist.jpg",
    ]
    assert all(path.is_file() for path in paths)
