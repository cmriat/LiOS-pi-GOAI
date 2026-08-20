#!/usr/bin/env python3
"""Exercise a GOAI policy server without starting RoboDojo or Isaac Sim."""

from __future__ import annotations

import uuid
import asyncio
import argparse
from typing import Any

import numpy as np
import websockets

from pi.inference.goai_protocol import decode_frame, encode_frame


def _request(message_type: str, payload: dict[str, Any], *, step: int = 0) -> dict[str, Any]:
    return {
        "message_type": message_type,
        "message_id": str(uuid.uuid4()),
        "evaluation_id": "goai-mock",
        "action_case_id": "stack_bowls_case",
        "trial_id": "stack_bowls-mock",
        "repeat_index": 0,
        "step": step,
        "payload": payload,
    }


def _observation() -> dict[str, Any]:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image.copy()},
            "cam_right_wrist": {"color": image.copy()},
        },
        "state": {
            "left_arm_joint_state": np.zeros(6, dtype=np.float32),
            "left_ee_joint_state": np.asarray([0.5], dtype=np.float32),
            "right_arm_joint_state": np.zeros(6, dtype=np.float32),
            "right_ee_joint_state": np.asarray([0.5], dtype=np.float32),
        },
        "instruction": "stack the bowls",
    }


async def _call(websocket: Any, message_type: str, payload: dict[str, Any], *, step: int = 0) -> dict[str, Any]:
    request = _request(message_type, payload, step=step)
    await websocket.send(encode_frame(request))
    response = decode_frame(await websocket.recv())
    expected = f"{message_type}_ack" if message_type in ("hello", "trial_end", "prepare_case") else f"{message_type}_result"
    if response.get("message_type") != expected:
        raise RuntimeError(f"Expected {expected}, got {response}")
    return response


async def run(url: str, expected_horizon: int) -> None:
    async with websockets.connect(url, max_size=None, compression=None, proxy=None) as websocket:
        await _call(websocket, "hello", {})
        await _call(websocket, "prepare_case", {"task_name": "stack_bowls"})
        await _call(websocket, "reset", {"trial_id": "stack_bowls-mock"})
        # v1.0.0 CALL flow: update_obs stores the observation server-side,
        # get_action infers from it. This exercises the official main path.
        await _call(websocket, "call", {"func_name": "update_obs", "obs": _observation()})
        response = await _call(websocket, "call", {"func_name": "get_action"})
        actions = response["payload"].get("result")  # CALL replies put actions under payload.result
        if not isinstance(actions, list) or len(actions) != expected_horizon:
            raise RuntimeError(
                f"Expected {expected_horizon} actions, got {type(actions).__name__}/{len(actions or [])}"
            )
        required = {
            "left_arm_joint_state": 6,
            "left_ee_joint_state": 1,
            "right_arm_joint_state": 6,
            "right_ee_joint_state": 1,
        }
        for action in actions:
            for key, dimension in required.items():
                value = np.asarray(action[key])
                if value.shape != (dimension,) or not np.isfinite(value).all():
                    raise RuntimeError(f"Invalid action {key}: shape={value.shape}")
        await _call(websocket, "trial_end", {"trial_id": "stack_bowls-mock"})
        print(f"GOAI_MOCK_OK: horizon={len(actions)}, latency_ms={response['payload'].get('latency_ms', 0.0):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:6060")
    parser.add_argument("--expected-horizon", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.expected_horizon))


if __name__ == "__main__":
    main()
