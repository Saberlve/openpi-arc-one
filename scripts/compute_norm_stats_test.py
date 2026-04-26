import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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


def test_parquet_low_dim_dataset_builds_action_chunks_without_images(tmp_path):
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(
        """
        {
          "total_episodes": 1,
          "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
          "chunks_size": 1000
        }
        """
    )
    table = pa.table(
        {
            "observation.state": [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]],
            "action": [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]],
        }
    )
    pq.write_table(table, data_dir / "episode_000000.parquet")

    dataset = compute_norm_stats.ParquetLowDimLeRobotDataset(
        tmp_path,
        state_key="observation.state",
        action_key="action",
        action_horizon=2,
    )

    item = dataset[1]

    np.testing.assert_array_equal(item["observation/state"], np.asarray([1.0, 11.0], dtype=np.float32))
    np.testing.assert_array_equal(
        item["actions"],
        np.asarray([[2.0, 12.0], [3.0, 13.0]], dtype=np.float32),
    )
