#!/usr/bin/env python3
"""Run the official debug client with an explicit, test-only task fixture."""

from __future__ import annotations

import sys
import json
import runpy
import argparse
from pathlib import Path

import numpy as np


class TaskFixture:
    """Delegate to the official TestEnv, replacing only its placeholder instruction."""

    def __init__(self, env, instruction):
        self.env = env
        self.instruction = instruction
        self.steps = 0
        self.first_actions = None
        self.observations = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_obs(self, env_idx=0):
        obs = self.env.get_obs(env_idx)
        if obs["instruction"] != "language instruction":
            raise ValueError("Official mock instruction changed; review the fixture")
        obs["instruction"] = self.instruction
        self.observations += 1
        return obs

    def get_obs_batch(self, env_idx_list):
        return [self.get_obs(i) for i in env_idx_list]

    def _record(self, actions):
        required = {
            "left_arm_joint_state": 6,
            "left_ee_joint_state": 1,
            "right_arm_joint_state": 6,
            "right_ee_joint_state": 1,
        }
        for action in actions:
            if set(action) != set(required):
                raise AssertionError("Invalid action keys")
            for key, size in required.items():
                value = np.asarray(action[key])
                if value.shape != (size,) or not np.isfinite(value).all():
                    raise AssertionError(f"Invalid action {key}")
        if self.first_actions is None:
            self.first_actions = [{k: np.asarray(v).copy() for k, v in a.items()} for a in actions]
        self.steps += 1

    def take_action(self, action):
        self._record([action])
        return self.env.take_action(action)

    def take_action_batch(self, actions, env_idx_list):
        self._record(actions)
        return self.env.take_action_batch(actions, env_idx_list)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task", default="stack_bowls")
    parser.add_argument("--encoded", choices=("true", "false"), required=True)
    parser.add_argument("--batch", choices=("true", "false"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bench = args.robodojo.resolve()
    sys.path[:0] = [str(bench), str(bench / "XPolicyLab")]
    from XPolicyLab.policy.Lion_Pi05 import deploy

    from pi.inference.goai_xpolicylab import resolve_real_task

    instruction = resolve_real_task(args.task)[1]
    original_single, original_batch = deploy.eval_one_episode, deploy.eval_one_episode_batch
    episodes = []

    def wrap(callback):
        def run(TASK_ENV, model_client):  # noqa: N803
            fixture = TaskFixture(TASK_ENV, instruction)
            callback(TASK_ENV=fixture, model_client=model_client)
            if fixture.steps != TASK_ENV.episode_step_limit:
                raise AssertionError("Official episode did not reach its step limit")
            expected_envs = 10 if args.batch == "true" else 1
            if len(fixture.first_actions or []) != expected_envs:
                raise AssertionError("Unexpected active environment count")
            if episodes:
                for actual, expected in zip(fixture.first_actions, episodes[0].first_actions, strict=True):
                    for key in expected:
                        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-5, atol=1e-5)
            episodes.append(fixture)

        return run

    # The official program still constructs TestEnv, connects and drives episodes.
    # Only the policy callback receives a test-only observation proxy.
    deploy.eval_one_episode = wrap(original_single)
    deploy.eval_one_episode_batch = wrap(original_batch)
    old_argv = sys.argv
    sys.argv = [
        str(bench / "XPolicyLab/debug_env_client.py"),
        "--bench_name",
        "goai",
        "--task_name",
        args.task,
        "--env_cfg_type",
        "arx_x5",
        "--policy_name",
        "Lion_Pi05",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--eval_episode_num",
        "2",
        "--eval_batch",
        args.batch,
        "--obs_encoded",
        args.encoded,
    ]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        sys.argv = old_argv
        deploy.eval_one_episode, deploy.eval_one_episode_batch = original_single, original_batch
    if len(episodes) != 2:
        raise AssertionError("Expected two completed episodes")
    result = {
        "status": "passed",
        "encoded": args.encoded == "true",
        "batch": args.batch == "true",
        "environments": 10 if args.batch == "true" else 1,
        "episodes": 2,
        "steps_per_episode": [e.steps for e in episodes],
        "observations_per_episode": [e.observations for e in episodes],
        "fixture_change": "instruction only: language instruction -> " + instruction,
        "reset_first_action_parity": True,
        "hardware": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
