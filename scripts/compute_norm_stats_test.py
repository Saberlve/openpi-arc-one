import pathlib

from openpi.training import config as _config

from . import compute_norm_stats


def test_norm_stats_output_path_uses_asset_id_for_absolute_repo_id():
    config = _config.TrainConfig(
        name="test_arx",
        assets_base_dir="/tmp/assets",
        data=_config.FakeDataConfig(
            repo_id="/mnt/d/share/pick_X_times_filterd_twice",
            assets=_config.AssetsConfig(asset_id="arx"),
        ),
    )
    data_config = _config.DataConfig(
        repo_id="/mnt/d/share/pick_X_times_filterd_twice",
        asset_id="arx",
    )

    assert compute_norm_stats.get_output_path(config, data_config) == pathlib.Path("/tmp/assets/test_arx/arx")
