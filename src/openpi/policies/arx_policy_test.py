import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import arx_policy
from openpi.training import config as _config


def test_arx_inputs_maps_dual_wrist_dataset_to_model_inputs():
    data = {
        "observation/images/head": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "observation/images/left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/images/right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.arange(14, dtype=np.float32),
        "actions": np.ones((10, 14), dtype=np.float32),
        "prompt": "pick up the black pouch",
    }

    inputs = arx_policy.ArxInputs(model_type=_model.ModelType.PI05)(data)

    assert inputs["state"].shape == (14,)
    assert inputs["actions"].shape == (10, 14)
    assert inputs["prompt"] == "pick up the black pouch"
    assert inputs["image"]["base_0_rgb"].shape == (480, 640, 3)
    assert inputs["image"]["left_wrist_0_rgb"].shape == (480, 640, 3)
    assert inputs["image"]["right_wrist_0_rgb"].shape == (480, 640, 3)
    assert inputs["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.True_,
    }


def test_arx_outputs_returns_all_14_action_dims():
    outputs = arx_policy.ArxOutputs()({"actions": np.ones((10, 32), dtype=np.float32)})

    assert outputs["actions"].shape == (10, 14)


def test_pi05_arx_config_points_to_local_lerobot_dataset(monkeypatch):
    class FakeModelTransformFactory:
        def __call__(self, model_config):
            return transforms.Group()

    monkeypatch.setattr(_config, "ModelTransformFactory", FakeModelTransformFactory)

    config = dataclasses.replace(_config.get_config("pi05_arx"), exp_name="test")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert data_config.repo_id == "/mnt/d/share/pick_X_times_filterd_twice"
    assert data_config.asset_id == "arx"
    assert data_config.prompt_from_task
    assert data_config.action_sequence_keys == ("action",)


def test_lerobot_v21_list_feature_metadata_is_supported():
    from datasets.features import Features

    from openpi.training import data_loader as _data_loader

    assert _data_loader is not None
    features = Features.from_dict(
        {
            "action": {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": 14,
                "_type": "List",
            }
        }
    )

    assert features["action"].length == 14
