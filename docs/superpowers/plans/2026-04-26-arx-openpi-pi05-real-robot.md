# ARX OpenPI Pi05 Real Robot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ARX-ONE real-robot inference path that uses this repository's `pi05_arx` OpenPI websocket policy server.

**Architecture:** Keep OpenPI model inference in `scripts/serve_policy.py`; keep ARX robot IO in `third_party/ARX-ONE`. Add a small ARX helper module for OpenPI observation/action adaptation and wire `third_party/ARX-ONE/src/edlsrobot/scripts/inference.py` to optionally use it.

**Tech Stack:** Python, pytest, openpi-client websocket policy client, existing ARX-ONE ROS/LeRobot runtime.

---

### Task 1: OpenPI Client Adapter

**Files:**
- Create: `third_party/ARX-ONE/src/edlsrobot/scripts/openpi_client_adapter.py`
- Test: `third_party/ARX-ONE/tests/test_openpi_client_adapter.py`

- [ ] Write a failing test for building OpenPI ARX observations from ARX image/state dictionaries.
- [ ] Run `pytest third_party/ARX-ONE/tests/test_openpi_client_adapter.py -q` and verify import/function failure.
- [ ] Implement `build_openpi_arx_observation(obs_dict, prompt)`.
- [ ] Run the adapter test and verify it passes.

### Task 2: Action Chunk Selection

**Files:**
- Modify: `third_party/ARX-ONE/src/edlsrobot/scripts/openpi_client_adapter.py`
- Test: `third_party/ARX-ONE/tests/test_openpi_client_adapter.py`

- [ ] Write a failing test for selecting the first action from `{"actions": action_chunk}`.
- [ ] Run the adapter test and verify the new test fails.
- [ ] Implement `select_first_openpi_action(result)`.
- [ ] Run the adapter test and verify it passes.

### Task 3: ARX Inference Wiring

**Files:**
- Modify: `third_party/ARX-ONE/src/edlsrobot/scripts/inference.py`

- [ ] Add CLI fields `openpi_server_host`, `openpi_server_port`, and `use_openpi_server` to `InferenceConfig`.
- [ ] In `init_infer_engine`, when `use_openpi_server` is true, create `WebsocketClientPolicy` and skip LeRobot policy creation.
- [ ] In `inference_process`, when OpenPI mode is active, build an OpenPI observation, call `client.infer()`, select the first action, and publish it through `robot_action`.
- [ ] Keep existing LeRobot local-policy inference behavior unchanged when OpenPI mode is false.

### Task 4: Launch Script

**Files:**
- Create: `third_party/ARX-ONE/src/edlsrobot/scripts/infer_openpi.sh`

- [ ] Add a shell entry point for ACone/ARX real-robot OpenPI pi05 inference.
- [ ] Include editable defaults for host, port, prompt, camera devices, robot type, and dataset repo id.

### Task 5: Verification

**Files:**
- Existing tests and touched files.

- [ ] Run `pytest third_party/ARX-ONE/tests/test_openpi_client_adapter.py -q`.
- [ ] Run `python -m py_compile third_party/ARX-ONE/src/edlsrobot/scripts/openpi_client_adapter.py third_party/ARX-ONE/src/edlsrobot/scripts/inference.py`.
- [ ] Report that hardware execution was not run unless a robot is attached.
