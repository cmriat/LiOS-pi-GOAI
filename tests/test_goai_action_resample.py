"""CPU coverage for optional prediction-span compression and legacy compatibility."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pi.inference.goai_helpers import unnormalize_actions
from pi.inference.goai_sim_policy import GOAISimPolicy, action_chunk_to_robodojo, apply_absolute_actions_goai
from pi.inference.goai_action_resample import ActionResampler, _pchip


def _resampler(source=32, **kwargs):
    return ActionResampler.from_config(
        {"enabled": True, "source_horizon": source, **kwargs}, execution_horizon=16, model_horizon=32
    )


def _path():
    actions = np.broadcast_to(np.linspace(0, 0.2, 32, dtype=np.float32)[:, None], (32, 14)).copy()
    actions[:, (6, 13)] = 0.5 + actions[:, (6, 13)] / 2
    return actions


@pytest.mark.parametrize("source", [20, 24, 32])
def test_longer_span_preserves_endpoints_dtype_and_shared_time_axis(source):
    actions = _path()
    original = actions.copy()
    output, status = _resampler(source).select(actions, actions[0])
    assert output.shape == (16, 14)
    assert output.dtype == np.float32
    assert status["applied"]
    np.testing.assert_array_equal(output[[0, -1]], actions[[0, source - 1]])
    np.testing.assert_allclose(output[:, 0], np.linspace(0, actions[source - 1, 0], 16), atol=1e-7)
    np.testing.assert_allclose(output[:, 6], 0.5 + output[:, 0] / 2, atol=1e-7)
    np.testing.assert_array_equal(actions, original)


@pytest.mark.parametrize("raw", [None, {}, {"enabled": False}, {"enabled": True, "source_horizon": 16}])
def test_disabled_or_identity_is_exact_prefix(raw):
    resampler = ActionResampler.from_config(raw, execution_horizon=16, model_horizon=32)
    actions = np.random.default_rng(0).normal(size=(32, 14)).astype(np.float32)
    output, status = resampler.select(actions, np.zeros(14))
    assert not status["applied"]
    np.testing.assert_array_equal(output, actions[:16])


def test_pchip_has_no_overshoot_across_plateaus_and_turning_points():
    values = np.array([0, 0.01, 0.7, 0.7, 0.2, 1, 0.99, 0], dtype=np.float32)[:, None]
    output = _pchip(values, 257)
    positions = np.linspace(0, len(values) - 1, len(output))
    left = np.minimum(positions.astype(int), len(values) - 2)
    low = np.minimum(values[left], values[left + 1])
    high = np.maximum(values[left], values[left + 1])
    assert np.isfinite(output).all()
    assert np.all(output >= low - 1e-7)
    assert np.all(output <= high + 1e-7)


def test_pchip_preserves_monotone_dimensions_and_two_point_line():
    ascending = np.array([0, 0.01, 0.02, 0.8, 0.8, 1], dtype=np.float32)
    output = _pchip(np.stack([ascending, -ascending], axis=1), 51)
    assert (np.diff(output[:, 0]) >= 0).all()
    assert (np.diff(output[:, 1]) <= 0).all()
    np.testing.assert_allclose(_pchip(np.array([[0.0], [1.0]]), 5)[:, 0], [0, 0.25, 0.5, 0.75, 1])


@pytest.mark.parametrize("dimension", [6, 13])
def test_gripper_transition_in_previously_discarded_tail_falls_back(dimension):
    actions = _path()
    actions[20:, dimension] = 0.0
    output, status = _resampler().select(actions, actions[0])
    assert status["reason"] == "gripper_guard"
    np.testing.assert_array_equal(output, actions[:16])
    _, unguarded = _resampler(gripper_guard=None).select(actions, actions[0])
    assert unguarded["applied"]


@pytest.mark.parametrize("initial_jump", [False, True])
def test_arm_guard_checks_within_chunk_and_observed_state(initial_jump):
    actions = _path()
    state = actions[0].copy()
    if initial_jump:
        state[7] -= 0.2
    else:
        actions[20:, 7] += 0.2
    output, status = _resampler().select(actions, state)
    assert status["reason"] == "max_step_rad"
    np.testing.assert_array_equal(output, actions[:16])
    _, unguarded = _resampler(max_step_rad=None).select(actions, state)
    assert unguarded["applied"]


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"enable": True},
        {"enabled": "false"},
        {"source_horizon": True},
        {"source_horizon": 24.0},
        {"source_horizon": 15},
        {"source_horizon": 33},
        {"gripper_guard": -0.1},
        {"gripper_guard": 1.1},
        {"gripper_guard": float("nan")},
        {"max_step_rad": 0},
        {"max_step_rad": float("inf")},
        {"max_step_rad": True},
    ],
)
def test_bad_config_rejected(raw):
    with pytest.raises(ValueError, match="action_resample"):
        ActionResampler.from_config(raw, execution_horizon=16, model_horizon=32)


def test_single_output_cannot_preserve_a_longer_span():
    with pytest.raises(ValueError, match="execution_horizon"):
        ActionResampler.from_config({"enabled": True}, execution_horizon=1, model_horizon=32)


@pytest.mark.parametrize("bad", ["short", "nonfinite", "state"])
def test_invalid_decoded_input_rejected(bad):
    actions, state = _path(), np.zeros(14)
    if bad == "short":
        actions = actions[:16]
    elif bad == "nonfinite":
        actions[-1, 0] = np.nan
    else:
        state[0] = np.inf
    with pytest.raises(ValueError, match="action_resample"):
        _resampler().select(actions, state)


def _cpu_policy(config):
    """Use the real infer/decode path with a CPU sampler and no checkpoint load."""
    policy = object.__new__(GOAISimPolicy)
    policy.device = torch.device("cpu")
    policy.model_config = SimpleNamespace(action_horizon=32, action_dim=32)
    policy.execution_horizon = 16
    policy.seed = 0
    policy.task_name = None
    policy.task_index = None
    policy.num_steps = 20
    policy.compile_mode = "none"
    policy.use_quantile_norm = True
    policy.action_norm_mode = "per-timestamp"
    policy.apply_delta = True
    policy.last_resample = None
    # 镜像 __init__ 的按任务结构:覆盖表 + per-task 重采样器缓存(None 为全局缺省)。
    policy.task_settings = {}
    policy._resamplers = {None: ActionResampler.from_config(config, execution_horizon=16, model_horizon=32)}
    policy.action_resampler = policy._resamplers[None]
    center = _path().astype(np.float64)
    width = np.broadcast_to(np.linspace(0.01, 0.02, 32)[:, None], (32, 14))
    policy.norm_stats = {
        "actions": SimpleNamespace(per_timestamp_q01=center - width, per_timestamp_q99=center + width)
    }
    policy._prepare_observation = lambda *_args: None
    policy._sample_actions = lambda **kwargs: kwargs["noise"]
    return policy


def _flatten(chunk):
    keys = ("left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state")
    return np.stack([np.concatenate([step[key] for key in keys]) for step in chunk])


@pytest.mark.parametrize("config", [None, {"enabled": False, "source_horizon": 32}])
def test_disabled_infer_matches_legacy_outputs_and_rng_across_calls(config):
    policy = _cpu_policy(config)
    state = np.linspace(0, 0.5, 14, dtype=np.float32)
    session = policy.create_session(seed=23)
    reference_rng = torch.Generator(device="cpu").manual_seed(23)
    for _ in range(3):
        noise = torch.randn((1, 32, 32), dtype=torch.float32, generator=reference_rng)
        decoded = unnormalize_actions(
            noise[0, :, :14].numpy(), policy.norm_stats["actions"], "per-timestamp", use_quantile_norm=True
        )
        absolute = apply_absolute_actions_goai(decoded, state, apply_delta=True)
        expected = action_chunk_to_robodojo(absolute[:16])
        actual = policy.infer({"state": state}, session)
        np.testing.assert_array_equal(_flatten(actual), _flatten(expected))
        assert torch.equal(session.generator.get_state(), reference_rng.get_state())
        assert policy.last_resample is None
    assert session.step == 3


@pytest.mark.parametrize("source", [20, 24, 32])
def test_infer_resamples_after_original_timestamp_norm_and_absolute_conversion(source):
    policy = _cpu_policy({"enabled": True, "source_horizon": source})
    policy._sample_actions = lambda **kwargs: torch.zeros_like(kwargs["noise"])
    state = np.linspace(0, 0.5, 14, dtype=np.float32)
    session = policy.create_session(seed=23)
    actual = _flatten(policy.infer({"state": state}, session))
    expected = np.broadcast_to(np.linspace(0, (source - 1) * 0.2 / 31, 16)[:, None], (16, 14)).copy()
    expected[:, (6, 13)] = 0.5 + expected[:, (6, 13)] / 2
    expected += 0.5e-6
    arms = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    expected[:, arms] += state[arms]
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert policy.last_resample["applied"]
    assert session.step == 1


def test_server_keeps_fixed_length_final_joint_limits_and_trace(monkeypatch, tmp_path):
    from pi.inference import goai_xpolicylab as adapter

    def load(**kwargs):
        policy = _cpu_policy(kwargs["action_resample"])
        policy.model = object()
        policy._sample_actions = lambda **kwargs: torch.zeros_like(kwargs["noise"])
        return policy

    monkeypatch.setattr(adapter, "validate_checkpoint", lambda _path: (tmp_path, tmp_path / "stats"))
    monkeypatch.setattr(adapter, "resolve_checkpoint_config_name", lambda _path: "unused")
    monkeypatch.setattr(adapter.Model, "_load_policy", staticmethod(load))
    monkeypatch.setattr(adapter.TraceWriter, "_write_images", lambda *_args: None)
    model = adapter.Model(
        {
            "checkpoint_path": str(tmp_path),
            "execution_horizon": 16,
            "action_resample": {"enabled": True, "source_horizon": 32},
            "joint_limits": [[-0.1, 0.1]] * 6,
            "trace": {"enabled": True, "dir": str(tmp_path / "trace")},
        }
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = {
        "instruction": "stack_bowls",
        "state": np.zeros(14),
        "images": {name: image for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
    }
    results = []
    for _ in range(2):
        model.reset()
        model.update_obs(obs)
        result = model.get_action()
        assert len(result) == 16
        assert all(set(step) == {key for key, _ in adapter._STATE_KEYS} for step in result)
        actual = _flatten(result)
        results.append(actual)
        assert np.isfinite(actual).all()
        assert actual[-1, 0] == np.float32(0.1)
        assert actual[-1, 6] == pytest.approx(0.6, abs=1e-6)
        assert model.policy.last_resample["applied"]
    np.testing.assert_array_equal(*results)
    model.trace.close()
    meta = json.loads((model.trace.root / "meta.json").read_text())
    # action_resample 现在按任务分层记录:"global" 是全局默认,其余键是任务名。
    assert meta["action_resample"]["global"]["source_horizon"] == 32
    calls = [json.loads(line) for line in (model.trace.root / "trace.jsonl").read_text().splitlines()]
    calls = [call for call in calls if call["event"] == "call"]
    assert len(calls) == 2
    for call in calls:
        assert call["action_resample"]["reason"] == "resampled"
        np.testing.assert_array_equal(call["action"], results[0])
