"""Tests for GOAI policy conversion and XPolicyLab wire compatibility."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scripts.inference.goai.sim_server import GOAIWebSocketServer, decode_frame, encode_frame, _request_task_name

from pi.shared.embodiment import build_embodiment_contract
from pi.inference.goai_sim_policy import (
    GOAI_DELTA_ACTION_MASK,
    GOAISimPolicy,
    GOAIPolicySession,
    assemble_goai_state,
    resolve_goai_task_index,
    action_chunk_to_robodojo,
    adapt_robodojo_observation,
    apply_absolute_actions_goai,
    load_goai_checkpoint_training_contract,
)


class _FakeGenerator:
    def __init__(self) -> None:
        self.seed = None

    def manual_seed(self, seed: int) -> None:
        self.seed = seed


def _raw_observation() -> dict:
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    return {
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image.copy()},
            "cam_right_wrist": {"color": image.copy()},
        },
        "state": {
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
            "left_ee_joint_state": np.asarray([0.25], dtype=np.float32),
            "right_arm_joint_state": np.arange(6, dtype=np.float32) + 10,
            "right_ee_joint_state": np.asarray([0.75], dtype=np.float32),
        },
        "instruction": "stack the bowls",
    }


def test_goai_task_mapping_matches_dataset_and_random_aliases() -> None:
    assert resolve_goai_task_index("arrange_largest_number") == 0
    assert resolve_goai_task_index("push_T") == 6
    assert resolve_goai_task_index("push_t_random") == 6
    assert resolve_goai_task_index("sweep_blocks") == 11
    # Legacy simulator slugs retain their historical slots.
    assert resolve_goai_task_index("stack_bowls") == 9
    # Real evaluation instructions use the shared official six-task matcher.
    assert resolve_goai_task_index("Stack the bowls.") == 3
    assert resolve_goai_task_index("FILL, THE PEN HOLDER!") == 0
    with pytest.raises(ValueError, match="Unsupported GOAI task"):
        resolve_goai_task_index("unknown_task")


def test_shared_server_resolves_task_from_action_case() -> None:
    assert _request_task_name({"action_case_id": "push_T_random_case", "payload": {}}) == "push_T_random"
    assert (
        _request_task_name({"action_case_id": "ignored_case", "payload": {"task_name": "stack_bowls"}}) == "stack_bowls"
    )


def test_shared_server_treats_cuda_graph_allocation_errors_as_fatal() -> None:
    assert GOAIWebSocketServer._is_fatal_policy_error(RuntimeError("beginAllocateToPool failed"))
    assert GOAIWebSocketServer._is_fatal_policy_error(RuntimeError("CUDA out of memory"))
    assert not GOAIWebSocketServer._is_fatal_policy_error(ValueError("bad observation"))


def test_shared_policy_session_binds_task_and_seed() -> None:
    policy = object.__new__(GOAISimPolicy)
    generator = _FakeGenerator()
    session = GOAIPolicySession(generator=generator, seed=0)

    GOAISimPolicy.bind_session(policy, session, task_name="push_T_random", seed=2)

    assert session.task_name == "push_T_random"
    assert session.task_index == 6
    assert session.seed == 2
    assert generator.seed == 2


def test_raw_robodojo_observation_is_packed_and_padded() -> None:
    observation = _raw_observation()
    state = assemble_goai_state(observation)
    adapted = adapt_robodojo_observation(observation)

    np.testing.assert_array_equal(state[:7], np.asarray([0, 1, 2, 3, 4, 5, 0.25]))
    np.testing.assert_array_equal(state[7:], np.asarray([10, 11, 12, 13, 14, 15, 0.75]))
    assert adapted["state"].shape == (14,)
    assert set(adapted["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(image.shape == (224, 224, 3) for image in adapted["image"].values())


def test_joint_delta_inverse_preserves_and_clips_grippers() -> None:
    current_state = np.arange(14, dtype=np.float32)
    actions = np.full((2, 14), 0.5, dtype=np.float32)
    actions[:, 6] = -1.0
    actions[:, 13] = 2.0

    absolute = apply_absolute_actions_goai(actions, current_state, apply_delta=True)

    expected_joints = np.broadcast_to(current_state[GOAI_DELTA_ACTION_MASK] + 0.5, (2, 12))
    np.testing.assert_allclose(absolute[:, GOAI_DELTA_ACTION_MASK], expected_joints)
    np.testing.assert_allclose(absolute[:, 6], 0.0)
    np.testing.assert_allclose(absolute[:, 13], 1.0)


def test_action_chunk_uses_exact_robodojo_joint_schema() -> None:
    chunk = action_chunk_to_robodojo(np.arange(28, dtype=np.float32).reshape(2, 14))

    assert len(chunk) == 2
    assert {key: value.shape for key, value in chunk[0].items()} == {
        "left_arm_joint_state": (6,),
        "left_ee_joint_state": (1,),
        "right_arm_joint_state": (6,),
        "right_ee_joint_state": (1,),
    }


def test_xpolicylab_msgpack_numpy_wire_round_trip() -> None:
    frame = {
        "message_type": "infer",
        "message_id": "request-1",
        "evaluation_id": "eval-1",
        "payload": {"observation": _raw_observation()},
    }

    restored = decode_frame(encode_frame(frame))

    assert restored["message_type"] == "infer"
    np.testing.assert_array_equal(
        restored["payload"]["observation"]["vision"]["cam_head"]["color"],
        frame["payload"]["observation"]["vision"]["cam_head"]["color"],
    )


def test_goai_checkpoint_contract_requires_official_geometry(tmp_path) -> None:
    checkpoint = tmp_path / "step1000"
    checkpoint.mkdir()
    entry = {
        "config_name": "pi05_goai_joint",
        "image_geometry": {
            "width": 640,
            "height": 640,
            "resize_mode": "pad",
            "source_width": 640,
            "source_height": 480,
        },
        "use_task_embedding": True,
        "use_language_with_task_embedding": False,
        "use_quantile_norm": True,
        "use_per_timestamp_action_norm": True,
        "num_tasks": 12,
        "action_horizon": 32,
        "max_token_len": 200,
    }
    (checkpoint / "norm_stats_manifest.json").write_text(json.dumps({"version": 7, "files": [entry]}), encoding="utf-8")

    contract = load_goai_checkpoint_training_contract(checkpoint, "pi05_goai_joint")
    assert contract["use_task_embedding"] is True
    assert contract["task_embedding_target"] == "vlm"
    assert contract["state_conditioning_mode"] == "discrete_vlm"
    assert contract["action_horizon"] == 32

    entry["image_geometry"]["height"] = 480
    (checkpoint / "norm_stats_manifest.json").write_text(json.dumps({"version": 7, "files": [entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry mismatch"):
        load_goai_checkpoint_training_contract(checkpoint, "pi05_goai_joint")


def test_goai_v12_checkpoint_contract_requires_delta_marker(tmp_path) -> None:
    checkpoint = tmp_path / "step1000"
    checkpoint.mkdir()
    entry = {
        "config_name": "pi05_goai_joint",
        "image_geometry": {
            "width": 640,
            "height": 640,
            "resize_mode": "pad",
            "source_width": 640,
            "source_height": 480,
        },
        "use_task_embedding": True,
        "use_language_with_task_embedding": False,
        "use_embodiment_embedding": True,
        "num_embodiments": 2,
        "num_embodiment_tokens": 1,
        "embodiment_contract": build_embodiment_contract(
            num_embodiments=2,
            tokens_per_embodiment=1,
        ),
        "task_embedding_target": "vlm",
        "state_conditioning_mode": "discrete_vlm",
        "use_quantile_norm": True,
        "use_per_timestamp_action_norm": True,
        "num_tasks": 12,
        "action_horizon": 32,
        "max_token_len": 200,
    }
    manifest = {"version": 12, "files": [entry]}
    (checkpoint / "norm_stats_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="apply_delta_transform"):
        load_goai_checkpoint_training_contract(checkpoint, "pi05_goai_joint")

    entry["apply_delta_transform"] = True
    (checkpoint / "norm_stats_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    contract = load_goai_checkpoint_training_contract(checkpoint, "pi05_goai_joint")
    assert contract["apply_delta_transform"] is True


def test_policy_rejects_manifest_delta_mismatch_before_weight_load(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "step1000"
    checkpoint.mkdir()
    (checkpoint / ".metadata").touch()
    norm_stats = checkpoint / "norm_stats_pt.json"
    norm_stats.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "pi.inference.goai_sim_policy.instance_config.get_config",
        lambda _config_name: SimpleNamespace(model=SimpleNamespace(pi05=True, discrete_state_input=True)),
    )
    monkeypatch.setattr(
        "pi.inference.goai_sim_policy.load_goai_checkpoint_training_contract",
        lambda _checkpoint, _config_name: {"apply_delta_transform": True},
    )

    with pytest.raises(ValueError, match="action transform mismatch"):
        GOAISimPolicy(
            checkpoint,
            norm_stats,
            apply_delta=False,
        )


@pytest.mark.parametrize("slots", [6, 12])
def test_checkpoint_task_count_reaches_model_construction(tmp_path, monkeypatch, slots):
    from pi.inference import goai_sim_policy as backend

    checkpoint = tmp_path / "step"
    checkpoint.mkdir()
    (checkpoint / ".metadata").touch()
    stats = {
        key: {"mean": [0.0] * 14, "std": [1.0] * 14, "q01": [-1.0] * 14, "q99": [1.0] * 14}
        for key in ("state", "actions")
    }
    stats_path = checkpoint / "norm_stats_pt.json"
    stats_path.write_text(json.dumps({"norm_stats": stats}))
    contract = dict(
        action_horizon=32,
        max_token_len=200,
        use_task_embedding=True,
        use_language_with_task_embedding=False,
        use_embodiment_embedding=False,
        num_embodiments=1,
        num_embodiment_tokens=1,
        task_embedding_target="vlm",
        state_conditioning_mode="discrete_vlm",
        num_tasks=slots,
        apply_delta_transform=True,
        use_quantile_norm=True,
        use_per_timestamp_action_norm=False,
    )
    monkeypatch.setattr(backend, "load_goai_checkpoint_training_contract", lambda *_args: contract)

    class ReachedModel(Exception):
        pass

    def stop_at_model(config):
        assert config.num_tasks == slots
        raise ReachedModel

    monkeypatch.setattr(backend, "PI0Pytorch", stop_at_model)
    with pytest.raises(ReachedModel):
        backend.GOAISimPolicy(checkpoint, stats_path, apply_delta=True, device="cpu")


@pytest.mark.parametrize("index", [-1, 6, 11])
def test_session_task_index_checked_against_checkpoint(monkeypatch, index):
    from pi.inference import goai_sim_policy as backend

    monkeypatch.setattr(backend, "adapt_robodojo_observation", lambda _obs: {})
    monkeypatch.setattr(backend, "normalize_values", lambda value, *_args, **_kwargs: value)
    policy = SimpleNamespace(
        model_config=SimpleNamespace(use_task_embedding=True, num_tasks=6),
        norm_stats={"state": None},
        use_quantile_norm=True,
    )
    session = SimpleNamespace(task_index=index, task_name="test")
    with pytest.raises(ValueError, match="outside checkpoint slots"):
        backend.GOAISimPolicy._prepare_observation(policy, {}, np.zeros(14), session)


def test_state_is_padded_to_action_dim():
    """模型内部一律是 action_dim 宽:动作侧由 infer 切回来,state 侧必须在这里补齐。

    dual 的连续 Expert token 直接吃这 32 维,漏了会报
    "expects state shape [batch, 1, 32], got (1, 1, 14)"。
    训练侧顺序是先 _tokenize_prompt 再 _pad_state_actions —— 所以 tokenization 仍用
    14 维原始 state,这里只钉住"喂给模型的那份"。
    """
    from pi.inference.goai_sim_policy import _pad_state

    state = np.arange(14, dtype=np.float32)
    padded = _pad_state(state, 32)
    assert padded.shape == (32,)
    np.testing.assert_array_equal(padded[:14], state)
    assert not padded[14:].any(), "必须是右侧零填充,与训练 _pad_state_actions 一致"
    assert _pad_state(state, 32).dtype == np.float32
    assert _pad_state(state, 14).shape == (14,), "已够宽就原样返回"
    assert _pad_state(np.stack([state, state]), 32).shape == (2, 32)
