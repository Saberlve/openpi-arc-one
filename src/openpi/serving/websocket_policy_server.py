import asyncio
import http
import logging
import os
import pathlib
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def describe_debug_payload(payload, *, max_vector_values: int = 10):
    if isinstance(payload, dict):
        return {key: describe_debug_payload(value, max_vector_values=max_vector_values) for key, value in payload.items()}

    if isinstance(payload, bytes):
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            return {"type": "bytes", "length": len(payload)}

    if isinstance(payload, str):
        return payload

    try:
        array = np.asarray(payload)
    except Exception:
        return repr(payload)

    if array.dtype == object:
        return repr(payload)

    summary = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }
    if array.size == 0:
        return summary

    if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        if array.ndim <= 1 and array.size <= max_vector_values:
            summary["values"] = array.tolist()
        elif array.ndim <= 1:
            head_count = max_vector_values // 2
            tail_count = max_vector_values - head_count
            summary["values"] = array[:head_count].tolist() + ["..."] + array[-tail_count:].tolist()
        else:
            summary["min"] = float(np.min(array))
            summary["max"] = float(np.max(array))
    return summary


def save_debug_observation_images(payload: dict, debug_dir: str | os.PathLike, *, prefix: str, step: int) -> list[pathlib.Path]:
    from PIL import Image

    output_dir = pathlib.Path(debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    image_keys = [
        "observation/images/head",
        "observation/images/left_wrist",
        "observation/images/right_wrist",
    ]
    for key in image_keys:
        if key not in payload:
            continue
        image = np.asarray(payload[key])
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
        if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
            continue
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

        camera = key.rsplit("/", 1)[-1]
        path = output_dir / f"{prefix}_step_{step:06d}_{camera}.jpg"
        Image.fromarray(image.squeeze()).save(path)
        saved_paths.append(path)
    return saved_paths


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._debug_io = _env_flag("OPENPI_DEBUG_IO")
        self._debug_dir = os.environ.get("OPENPI_DEBUG_DIR", "openpi-arx-debug")
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        step = 0
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                if self._debug_io:
                    logger.info("server observation: %s", describe_debug_payload(obs))
                    saved_paths = save_debug_observation_images(obs, self._debug_dir, prefix="server_obs", step=step)
                    if saved_paths:
                        logger.info("server observation images saved: %s", [str(path) for path in saved_paths])

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                if self._debug_io:
                    logger.info("server action: %s", describe_debug_payload(action))

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
                step += 1

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
