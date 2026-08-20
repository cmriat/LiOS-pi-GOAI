"""RoboDojo execution loop for the GOAI policy server.

Self-contained v1.0.0 CALL client: speaks the WS protocol directly using the
sim_server codec bundled in this repository (no XPolicyLab dependency).
The caller (RoboDojo eval_env) provides TASK_ENV with get_obs/take_action and
a model_client with call(func_name, obs=None) semantics, or the loop falls
back to a raw-websocket client created from the connection metadata below.
"""

from __future__ import annotations

import asyncio
from typing import Any

CONNECTION = {"url": None, "trial_id": None, "action_case_id": None}  # set by caller when used raw


def _raw_call(url: str, func_name: str, obs: Any = None, *, trial_id: str, step: int = 0) -> Any:
    """One synchronous CALL round-trip using the bundled codec."""
    import uuid

    import websockets

    from pi.inference.goai_protocol import decode_frame, encode_frame

    payload: dict[str, Any] = {"func_name": func_name}
    if obs is not None:
        payload["obs"] = obs
    frame = {
        "message_type": "call",
        "message_id": str(uuid.uuid4()),
        "evaluation_id": "robodojo",
        "action_case_id": None,
        "trial_id": trial_id,
        "repeat_index": 0,
        "step": step,
        "payload": payload,
    }

    async def _once() -> Any:
        async with websockets.connect(url, max_size=None, compression=None) as ws:
            await ws.send(encode_frame(frame))
            response = decode_frame(await ws.recv())
            if response.get("message_type") != "call_result":
                raise RuntimeError(f"Expected call_result, got {response}")
            return response.get("payload", {}).get("result")

    return asyncio.run(_once())


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        observation = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=observation)
        actions = model_client.call(func_name="get_action")
        for action_index, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_index + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_indices = TASK_ENV.get_running_env_idx_list()
        observations = TASK_ENV.get_obs_batch(env_indices)
        model_client.call(func_name="update_obs_batch", obs=observations)
        action_chunks = model_client.call(func_name="get_action_batch", obs=env_indices)
        chunk_size = len(action_chunks[0])
        for action_index in range(chunk_size):
            TASK_ENV.take_action_batch(
                [chunk[action_index] for chunk in action_chunks],
                env_indices,
            )
            if TASK_ENV.is_episode_end() or action_index + 1 == chunk_size:
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [index for index, env_index in enumerate(env_indices) if env_index in running]
            action_chunks = [action_chunks[index] for index in active]
            env_indices = [env_indices[index] for index in active]
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_indices))
