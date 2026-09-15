#!/usr/bin/env python3
"""Exercise external client transport with synthetic observations, without hardware."""

from __future__ import annotations

import sys
import json
import time
import argparse
import threading
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import websockets
from validate_xpolicylab import compare, make_observation


def check_chunk(actions, horizon):
    """Validate the complete dual-arm action contract."""
    sizes = {"left_arm_joint_state": 6, "left_ee_joint_state": 1, "right_arm_joint_state": 6, "right_ee_joint_state": 1}
    assert len(actions) == horizon
    for step in actions:
        assert set(step) == set(sizes)
        for key, size in sizes.items():
            value = np.asarray(step[key])
            assert value.shape == (size,) and np.isfinite(value).all()
            if size == 1:
                assert 0 <= value[0] <= 1


def deployment_observation(task):
    """Run the external camera encoder and observation builder on synthetic feedback."""
    from piperx_real.deploy.main import build_observation
    from piperx_real.deploy.cameras import CAMERA_ROLES, PolicyCameras

    cameras = object.__new__(PolicyCameras)
    cameras._lock = threading.Lock()
    cameras.paths = dict.fromkeys(CAMERA_ROLES)
    cameras._failures = {}
    rgb = np.empty((480, 640, 3), dtype=np.uint8)
    rgb[:] = [220, 40, 15]
    cameras._latest = {name: (rgb.copy(), time.monotonic()) for name in CAMERA_ROLES}
    runtime = SimpleNamespace(
        follower_obs=lambda _side, _now: (np.zeros(6, dtype=np.float32), 0.05),
        config={"gripper_calibrations": {side: {"follower_feedback_m": [0.0, 0.1]} for side in ("left", "right")}},
    )
    obs = build_observation(runtime, cameras, SimpleNamespace(task=task, jpeg_quality=95, no_instruction=False))
    for side in ("left", "right"):
        np.testing.assert_allclose(obs["state"][f"{side}_ee_joint_state"], [0.5])
    for camera in obs["vision"].values():
        decoded = cv2.imdecode(np.frombuffer(camera["color"], np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_allclose(decoded[240, 320], rgb[240, 320], atol=3)
    return obs


def check_deployment_chunk(actions):
    """Check actuator conversion without constructing a hardware executor."""
    from piperx_real.deploy.controller import validate_chunk

    limits = {side: (0.0, 0.1) for side in ("left", "right")}
    converted = validate_chunk(actions, np.full(6, -100.0), np.full(6, 100.0), gripper_limits=limits)
    for side, steps in converted.items():
        for action, (joints, meters) in zip(actions, steps, strict=True):
            np.testing.assert_allclose(joints, action[f"{side}_arm_joint_state"])
            np.testing.assert_allclose(meters, 0.1 * action[f"{side}_ee_joint_state"][0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-root", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--deployment-boundary",
        action="store_true",
        help="Also exercise camera, observation and actuator conversion functions on synthetic inputs",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.client_root.resolve() / "src"))
    from piperx_real.policy.tasks import TASKS
    from piperx_real.policy.client import PolicyWsClient, PolicyServerError

    results = []
    with PolicyWsClient(
        args.url,
        evaluation_id="real-client-acceptance",
        trial_id="synthetic",
        action_case_id="synthetic",
        request_timeout=600,
    ) as client:
        instance = client.server_instance_id
        client.heartbeat()
        for slug, instruction in TASKS.items():
            obs = deployment_observation(slug) if args.deployment_boundary else make_observation(instruction)
            obs.pop("env_idx", None)  # Match the actual deployment client's default environment.
            client.prepare_case({"task_name": slug})
            client.reset()
            client.call("update_obs", obs)
            expected = client.call("get_action")
            check_chunk(expected, args.horizon)
            if args.deployment_boundary:
                check_deployment_chunk(expected)
            client.reset()
            compare(client.infer(obs), expected)
            client.trial_end({"synthetic": True})
            try:
                client.call("get_action")
            except PolicyServerError as error:
                assert error.code == "call_failed"
            else:
                raise AssertionError("trial_end left an actionable observation")
            results.append(slug)
        client.reset()
        jpeg = make_observation(next(iter(TASKS.values())))
        for camera in jpeg["vision"].values():
            ok, encoded = cv2.imencode(".jpg", camera["color"])
            assert ok
            camera["color"] = encoded.tobytes()
        check_chunk(client.infer(jpeg), args.horizon)
        client.reset()
        first = make_observation(TASKS["stack_bowls"])
        second = make_observation(TASKS["insert_charger"])
        first["env_idx"], second["env_idx"] = 2, 9
        client.call("update_obs_batch", [first, second])
        batch = client.call("get_action_batch", [9, 2])
        assert len(batch) == 2
        for chunk in batch:
            check_chunk(chunk, args.horizon)
        client.reset()
        bad = make_observation("unsupported task")
        try:
            client.infer(bad)
        except PolicyServerError:
            pass
        else:
            raise AssertionError("Unknown task was accepted")
        client.trial_end()
        client.close()
        client.connect()
        assert client.server_instance_id == instance
        client.heartbeat()
    report = {
        "status": "passed",
        "hardware": False,
        "scope": "transport_with_synthetic_observations",
        "deployment_camera_and_units_validated": args.deployment_boundary,
        "deployment_inputs": "synthetic feedback and calibration; no hardware",
        "tasks": results,
        "client_websockets": websockets.__version__,
        "checks": [
            "hello",
            "prepare_case",
            "reset",
            "update_obs_get_action",
            "infer_parity",
            "trial_end",
            "heartbeat",
            "jpeg",
            "batch",
            "unknown_task",
            "reconnect",
            "close",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
