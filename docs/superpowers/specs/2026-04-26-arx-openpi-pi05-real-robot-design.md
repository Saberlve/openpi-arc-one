# ARX OpenPI Pi05 Real Robot Design

## Goal

Support real-robot deployment and inference for this repository's `pi05_arx` policy on ARX/ACone hardware.

## Boundary

The OpenPI process owns model loading, normalization, transforms, and checkpoint format. It starts with `scripts/serve_policy.py` and serves a websocket policy endpoint using `--policy.config=pi05_arx`.

The ARX-ONE process owns robot IO: ROS startup, camera/state collection, shared memory, timing, and action publishing. It queries the OpenPI server through `openpi_client.websocket_client_policy.WebsocketClientPolicy`.

## Data Flow

ARX-ONE reads the current robot observation and builds the exact dictionary expected by `src/openpi/policies/arx_policy.py`:

- `observation/images/head`
- `observation/images/left_wrist`
- `observation/images/right_wrist`
- `observation/state`
- `prompt`

The OpenPI server returns `{"actions": action_chunk}`. ARX-ONE executes the first action in the chunk by default, preserving its existing `robot_action(action, action_queue)` path.

## User Entry Points

Model server:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_arx \
  --policy.dir=/path/to/openpi/checkpoint \
  --port=8000 \
  --default-prompt="task instruction"
```

Robot client:

```bash
cd third_party/ARX-ONE/src/edlsrobot/scripts
bash infer_openpi.sh
```

The shell script exposes the server host, port, task prompt, robot type, camera mapping, and dataset repo id as editable values.

## Error Handling

The robot client validates that the returned action chunk contains `actions` and that at least one action exists. If the server is unavailable, the websocket client waits and logs through the existing client behavior. If the server returns an exception string, the client raises the existing runtime error.

## Testing

Unit tests cover the OpenPI observation builder and first-action selection without ROS or camera hardware. Existing ARX-ONE robot integration remains manual because it depends on hardware, ROS topics, CAN devices, and cameras.
