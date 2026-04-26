"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import bisect
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def get_output_path(config: _config.TrainConfig, data_config: _config.DataConfig) -> pathlib.Path:
    if data_config.asset_id is None:
        raise ValueError("Data config must have an asset_id to save normalization stats.")
    return config.assets_dirs / data_config.asset_id


class ParquetLowDimLeRobotDataset(_data_loader.Dataset):
    """LeRobot parquet reader for norm stats that avoids decoding video columns."""

    def __init__(self, root: pathlib.Path | str, *, state_key: str, action_key: str, action_horizon: int):
        self._root = pathlib.Path(root)
        self._state_key = state_key
        self._action_key = action_key
        self._action_horizon = action_horizon
        info = json.loads((self._root / "meta" / "info.json").read_text())
        self._episodes = []
        self._cumulative_lengths = []
        total_frames = 0
        # Iterate through all episodes and only read the state and action columns
        for episode_index in range(info["total_episodes"]):
            episode_chunk = episode_index // info["chunks_size"]
            path = self._root / info["data_path"].format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
            table = pq.read_table(path, columns=[self._state_key, self._action_key])
            states = np.asarray(table[self._state_key].to_pylist(), dtype=np.float32)
            actions = np.asarray(table[self._action_key].to_pylist(), dtype=np.float32)
            # Only keep the state and action columns
            self._episodes.append((states, actions))
            total_frames += len(states)
            self._cumulative_lengths.append(total_frames)

    def __len__(self) -> int:
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0

    def __getitem__(self, index) -> dict:
        index = int(index)
        episode_index = bisect.bisect_right(self._cumulative_lengths, index)
        episode_start = 0 if episode_index == 0 else self._cumulative_lengths[episode_index - 1]
        local_index = index - episode_start
        states, actions = self._episodes[episode_index]
        action_indices = np.minimum(
            np.arange(local_index, local_index + self._action_horizon),
            len(actions) - 1,
        )
        return {
            "observation/state": states[local_index],
            "actions": actions[action_indices],
        }


def _create_low_dim_lerobot_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
) -> _data_loader.Dataset | None:
    if data_config.repo_id is None or len(data_config.action_sequence_keys) != 1:
        return None
    root = pathlib.Path(data_config.repo_id)
    if not (root / "meta" / "info.json").is_file():
        return None
    # create a dataset only reads the state and action columns
    return ParquetLowDimLeRobotDataset(
        root,
        state_key="observation.state",
        action_key=data_config.action_sequence_keys[0],
        action_horizon=action_horizon,
    )


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _create_low_dim_lerobot_dataset(data_config, action_horizon)
    if dataset is None:
        dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
        input_transforms = data_config.repack_transforms.inputs
    else:
        input_transforms = ()
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *input_transforms,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = get_output_path(config, data_config)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
