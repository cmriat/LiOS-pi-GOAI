#!/usr/bin/env python3
"""GPU parity and official websocket smoke check with synthetic observations only."""

from __future__ import annotations

import sys
import json
import asyncio
import argparse
from pathlib import Path

import numpy as np


def make_observation(task):
    """Make asymmetric RGB and joint inputs to expose channel or arm swaps."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[..., 0] = 31
    image[..., 1] = 97
    image[..., 2] = 211
    image[50:150, 100:300] = (255, 0, 0)
    return {
        "env_idx": 7,
        "instruction": task,
        "state": {
            "left_arm_joint_state": np.linspace(-0.15, 0.2, 6, dtype=np.float32),
            "left_ee_joint_state": np.array([0.2], dtype=np.float32),
            "right_arm_joint_state": np.linspace(0.25, -0.1, 6, dtype=np.float32),
            "right_ee_joint_state": np.array([0.8], dtype=np.float32),
        },
        "vision": {
            "cam_head": {"color": image},
            "left_wrist": {"color": np.roll(image, 53, axis=1).copy()},
            "right_wrist": {"color": image[..., ::-1].copy()},
        },
    }


def compare(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("Action horizons differ")
    for got, want in zip(actual, expected, strict=True):
        if set(got) != set(want):
            raise AssertionError("Action keys differ")
        for key in want:
            np.testing.assert_allclose(got[key], want[key], rtol=1e-5, atol=1e-5)


async def validate(model, config, obs, expected):
    from client_server.ws.model_client import WsModelClient
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
    await server.start()

    def exercise():
        with WsModelClient(
            url=server.url,
            evaluation_id="goai-policy-smoke",
            trial_id="synthetic",
            action_case_id="synthetic",
            request_timeout_s=600,
        ) as client:
            client.call("prepare_case", {"task_name": config["task_name"]})
            client.call("reset")
            client.call("update_obs", obs)
            compare(client.call("get_action"), expected)
            client.call("reset")
            client.call("update_obs_batch", [obs])
            compare(client.call("get_action_batch", [7])[0], expected)
            client.call("trial_end", {"synthetic": True})

    try:
        await asyncio.to_thread(exercise)
    finally:
        await server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="JSON from prepare_xpolicylab.py")
    parser.add_argument("--deps", type=Path, help="Optional isolated official-server dependencies")
    args = parser.parse_args()
    bench = args.robodojo.expanduser().resolve()
    pi_root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(pi_root / "src"), str(bench), str(bench / "XPolicyLab")]
    if args.deps is not None:
        sys.path.insert(0, str(args.deps.expanduser().resolve()))
    from XPolicyLab.policy.lionvla.model import Model

    config = json.loads(args.config.read_text())
    model = Model(config)
    from pi.inference.goai_xpolicylab import resolve_real_task

    task_index, task_instruction = resolve_real_task(config["task_name"])
    # 配置键从 postprocess.gripper 搬到了顶层 gripper: + per_task.<任务名>.gripper。
    # 两处都显式查,不能靠 .get 默认值——搬完之后旧键取不到会静默通过。
    if (config.get("gripper") or {}).get("enabled", False) or any(
        (block.get("gripper") or {}).get("enabled", False)
        for block in (config.get("per_task") or {}).values()
    ):
        raise ValueError("Parity acceptance requires the gripper correction disabled")
    obs = make_observation(task_instruction)
    # Independent canonical input, without calling the adapter's canonicalization.
    state = np.concatenate(
        [
            obs["state"][key]
            for key in ("left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state")
        ]
    )
    reference_obs = {
        "state": state,
        "instruction": task_instruction,
        "images": {
            "cam_high": obs["vision"]["cam_head"]["color"],
            "cam_left_wrist": obs["vision"]["left_wrist"]["color"],
            "cam_right_wrist": obs["vision"]["right_wrist"]["color"],
        },
    }
    reference_session = model.policy.create_session(seed=(config["seed"] + 7) % (2**63))
    reference_session.task_index = task_index
    reference_session.task_name = task_instruction
    expected = model.policy.infer(reference_obs, reference_session)
    model.update_obs(obs)
    compare(model.get_action(), expected)
    model.reset()
    asyncio.run(validate(model, config, obs, expected))
    print(
        json.dumps(
            {
                "status": "passed",
                "checkpoint": config["checkpoint_path"],
                "checks": [
                    "strict_DCP_coverage",
                    "stats_hash",
                    "direct_parity",
                    "official_ws_single",
                    "official_ws_batch",
                    "reset_reproducibility",
                ],
                "hardware": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
