"""Regression coverage for the real-task official policy boundary."""

from __future__ import annotations

import json
import hashlib
import logging
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from pi.inference import goai_xpolicylab as adapter


@pytest.fixture
def checkpoint(tmp_path):
    ema = tmp_path / "step52019/ema"
    ema.mkdir(parents=True)
    (ema / ".metadata").touch()
    stats = b'{"norm_stats": {}}'
    (ema.parent / "norm_stats_pt.json").write_bytes(stats)
    entry = {
        "config_name": "pi05_goai_joint",
        "use_task_embedding": True,
        "use_language_with_task_embedding": False,
        "use_embodiment_embedding": True,
        "num_embodiments": 2,
        "num_embodiment_tokens": 1,
        "num_tasks": 12,
        "action_horizon": 32,
        "use_quantile_norm": True,
        "use_per_timestamp_action_norm": True,
        "apply_delta_transform": True,
        "task_embedding_target": "vlm",
        "state_conditioning_mode": "discrete_vlm",
        "sha256": hashlib.sha256(stats).hexdigest(),
    }
    (ema.parent / "norm_stats_manifest.json").write_text(json.dumps({"files": [entry]}))
    return ema


class Backend:
    model = object()

    def __init__(self):
        self.calls = []

    def create_session(self, *, seed):
        return SimpleNamespace(seed=seed, step=0)

    def infer(self, obs, session):
        self.calls.append((obs, session.task_index, session.seed, session.step))
        session.step += 1
        return [
            {key: np.full(size, session.seed + session.step, dtype=np.float32) for key, size in adapter._STATE_KEYS}
            for _ in range(8)
        ]


@pytest.fixture
def model(monkeypatch, checkpoint):
    backend = Backend()

    def load(**kwargs):
        assert kwargs["strict_checkpoint"] is True
        assert kwargs["embodiment_index"] == 0
        assert kwargs["norm_mode"] == "per-timestamp"
        return backend

    monkeypatch.setattr(adapter.Model, "_load_policy", staticmethod(load))
    return adapter.Model({"checkpoint_path": checkpoint, "task_name": "stack_bowls", "seed": 42})


def observation(env_id=0):
    image = np.zeros((480, 640, 3), np.uint8)
    image[..., 0] = 17
    image[..., 2] = 201
    return {
        "env_idx": env_id,
        "instruction": "Stack the bowls.",
        "state": np.arange(14, dtype=np.float32),
        "vision": {
            "cam_head": {"color": image},
            "left_wrist": {"rgb": image.transpose(2, 0, 1)},
            "right_wrist": {"color": image.copy()},
        },
    }


def test_slugs_are_real_slots_and_unknown_tasks_are_rejected():
    assert adapter.resolve_real_task("stack_bowls")[0] == 3
    assert adapter.resolve_real_task("Stack the bowls.")[0] == 3
    assert adapter.resolve_real_task("fill_pen_holder")[0] == 0
    for name in ("sweep_blocks", "disassemble_LEGO", "stack_bowls_random", "stack_bowl"):
        with pytest.raises(ValueError, match="Untrained"):
            adapter.resolve_real_task(name)


def test_batch_order_sessions_and_reset(model):
    model.update_obs_batch([observation(8), observation(2)])
    model.get_action_batch([2, 8])
    assert [(c[1], c[2], c[3]) for c in model.policy.calls] == [(3, 44, 0), (3, 50, 0)]
    model.get_action_batch([8])
    assert model.policy.calls[-1][2:] == (50, 1)
    model.reset()
    with pytest.raises(ValueError):
        model.get_action()
    model.update_obs(observation(8))
    model.get_action()
    assert model.policy.calls[-1][1:] == (3, 50, 0)


def test_observation_copy_color_and_state_preserved(model):
    obs = observation()
    model.update_obs(obs)
    obs["vision"]["cam_head"]["color"][:] = 0
    model.get_action()
    adapted = model.policy.calls[0][0]
    assert tuple(adapted["images"]["cam_high"][0, 0]) == (17, 0, 201)
    assert tuple(adapted["images"]["cam_left_wrist"][0, 0]) == (17, 0, 201)
    np.testing.assert_array_equal(adapted["state"], np.arange(14, dtype=np.float32))


def test_failed_update_is_atomic_and_ids_are_checked(model):
    model.update_obs(observation(1))

    broken = observation(2)
    broken["state"] = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError):
        model.update_obs_batch([observation(3), broken])
    model.get_action_batch([1])
    for ids in ([1, 1], [3], [], [True]):
        with pytest.raises(ValueError):
            model.get_action_batch(ids)
    with pytest.raises(ValueError, match="Duplicate"):
        model.update_obs_batch([observation(1), observation(1)])


@pytest.mark.parametrize("mutation", ["float_image", "bad_geometry", "nonfinite_state", "preprocessed"])
def test_invalid_inputs_rejected(model, mutation):
    obs = observation()
    if mutation == "float_image":
        obs["vision"]["cam_head"]["color"] = np.zeros((480, 640, 3), dtype=np.float32)
    elif mutation == "bad_geometry":
        obs["vision"]["cam_head"]["color"] = np.zeros((224, 224, 3), dtype=np.uint8)
    elif mutation == "nonfinite_state":
        obs["state"][0] = np.nan
    else:
        obs["images_preprocessed"] = True
    with pytest.raises(ValueError):
        model.update_obs(obs)


def test_wrong_stats_rejected(checkpoint):
    """统计量与 manifest 声明不符必拒 —— 拿错的统计量反归一化会**静默**产出垃圾动作。"""
    (checkpoint.parent / "norm_stats_pt.json").write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        adapter.validate_checkpoint(checkpoint)


def test_recipe_is_the_checkpoints_own_business(checkpoint):
    """训练配方不再被一张写死的表管 —— checkpoint 自己就是权威。

    2026-09-15:原先逐字段比对一张钉死的配方表,训练线换到 6 任务 / 单本体后把整批
    checkpoint 全拒了。模型配置由 GOAISimPolicy 从 manifest 重建,结构错配由
    validate_goai_dcp_coverage 按 shape 拦住。这两条一起钉住边界:改配方不拒,改统计量必拒。
    """
    manifest_path = checkpoint.parent / "norm_stats_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["use_embodiment_embedding"] = False
    manifest["files"][0]["num_tasks"] = 12
    manifest["files"][0]["state_conditioning_mode"] = "dual"
    manifest_path.write_text(json.dumps(manifest))

    _, stats = adapter.validate_checkpoint(checkpoint)  # 不该抛
    assert stats == checkpoint.parent / "norm_stats_pt.json"


def test_bad_output_and_case_rejected(model):
    model.update_obs(observation())
    model.policy.infer = lambda _obs, _session: [{"bad": np.zeros(1)}] * 8
    with pytest.raises(RuntimeError, match="keys"):
        model.get_action()
    model.on_trial_end()
    with pytest.raises(ValueError):
        model.get_action()


def test_dcp_coverage_allows_only_true_aliases():
    import torch

    from pi.inference.goai_sim_policy import validate_goai_dcp_coverage

    model = torch.nn.Module()
    model.first = torch.nn.Parameter(torch.zeros(2, 3))
    model.alias = model.first
    model.independent = torch.nn.Parameter(torch.zeros(2, 3))
    metadata = {"first": SimpleNamespace(size=(2, 3)), "independent": SimpleNamespace(size=(2, 3))}
    validate_goai_dcp_coverage(model, metadata)
    with pytest.raises(ValueError, match="independent"):
        validate_goai_dcp_coverage(model, {"first": metadata["first"]})
    with pytest.raises(ValueError, match="shape_mismatch"):
        validate_goai_dcp_coverage(model, {**metadata, "first": SimpleNamespace(size=(3, 2))})


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------


def with_instruction(value):
    """Copy a fixture with a replacement instruction, or omit it for None."""
    obs = observation()
    if value is None:
        obs.pop("instruction")
    else:
        obs["instruction"] = value
    return obs


def test_nearest_match_table():
    accepted = {
        "Stack the bowls": (adapter.GOAI_REAL_TASK_INSTRUCTIONS[3], True),
        "stack_bowls": (adapter.GOAI_REAL_TASK_INSTRUCTIONS[3], True),
        "  STACK THE BOWLS!! ": (adapter.GOAI_REAL_TASK_INSTRUCTIONS[3], True),
        "stack the bowl": (adapter.GOAI_REAL_TASK_INSTRUCTIONS[3], False),
        "Stand up the bottle": (adapter.GOAI_REAL_TASK_INSTRUCTIONS[4], False),
        "stack and cover block": (adapter.GOAI_REAL_TASK_INSTRUCTIONS[2], False),
    }
    for value, (instruction, exact) in accepted.items():
        index, resolved, ratio, was_exact = adapter.resolve_real_task_nearest(value)
        assert (resolved, was_exact) == (instruction, exact), value
        assert 0 < ratio <= 1.0

    for value in ("do the dishes", "stack the boxes", "pick up the cup", "xyzzy"):
        with pytest.raises(ValueError, match="nearest"):
            adapter.resolve_real_task_nearest(value)


def test_client_instruction_decides_the_session_task(model):
    """Check that client instruction decides the session task."""
    model.update_obs(with_instruction("Insert the charger"))
    model.get_action()
    assert model.policy.calls[-1][1] == 5
    assert model._observations[0]["instruction"] == adapter.GOAI_REAL_TASK_INSTRUCTIONS[5]


def test_missing_instruction_falls_back_to_the_configured_task(model):
    model.update_obs(with_instruction(None))
    model.get_action()
    assert model.policy.calls[-1][1] == 3
    assert model._observations[0]["instruction"] == adapter.GOAI_REAL_TASK_INSTRUCTIONS[3]


def test_typo_is_resolved_to_the_nearest_task(model, caplog):
    with caplog.at_level(logging.WARNING):
        model.update_obs(with_instruction("Stack the bowl"))
    assert model._observations[0]["instruction"] == adapter.GOAI_REAL_TASK_INSTRUCTIONS[3]
    assert "未精确匹配" in caplog.text


def test_unrelated_instruction_is_still_rejected(model):
    with pytest.raises(ValueError, match="Untrained"):
        model.update_obs(with_instruction("sweep the floor"))


def test_conflicting_fields_in_one_observation_are_rejected(model):
    obs = with_instruction("Stack the bowls")
    obs["prompt"] = "Insert the charger"
    with pytest.raises(ValueError, match="不同的任务"):
        model.update_obs(obs)


def test_prepare_case_records_but_does_not_reject_a_different_task(model):
    model.prepare_case({"task_name": "fill_pen_holder"})
    model.prepare_case({"task_name": "Fill the pen holder"})
    with pytest.raises(ValueError):
        model.prepare_case({"task_name": "sweep the floor"})


def test_task_can_change_between_episodes(model):
    """Check that task can change between episodes."""
    model.update_obs(with_instruction("Stack the bowls"))
    model.get_action()
    model.on_trial_end()
    model.update_obs(with_instruction("Insert the charger"))
    model.get_action()
    assert [c[1] for c in model.policy.calls] == [3, 5]


def test_active_task_switch_rejects_entire_batch(model):
    model.update_obs(observation(1))
    model.get_action_batch([1])
    switched = observation(1)
    switched["instruction"] = "Insert the charger"
    with pytest.raises(ValueError, match="reset"):
        model.update_obs_batch([observation(2), switched])
    assert list(model._observations) == [1]
    assert model._episode_task[1][0] == 3
    model.get_action_batch([1])
    assert model.policy.calls[-1][1] == 3


def _steps(values):
    """Build joint action steps with specified left and right gripper openings."""
    return [
        {
            "left_arm_joint_state": np.zeros(6, np.float32),
            "left_ee_joint_state": np.array([left], np.float32),
            "right_arm_joint_state": np.zeros(6, np.float32),
            "right_ee_joint_state": np.array([right], np.float32),
        }
        for left, right in values
    ]


def _on(**postprocess):
    """Enable postprocessing explicitly so subtraction tests exercise the correction."""
    cfg = {"enabled": True}
    cfg.update(postprocess)
    return adapter.PostProcess({"postprocess": cfg})


class TestSqueeze:
    def test_defaults(self):
        pp = adapter.PostProcess({})
        assert pp.enabled is False, "缺省必须关掉:漏写 enabled 不许悄悄改变机械臂行为"
        assert pp.squeeze == pytest.approx(adapter.DEFAULT_GRIPPER_SQUEEZE)
        assert pp.squeeze_below == pytest.approx(adapter.DEFAULT_GRIPPER_SQUEEZE_BELOW)
        assert pp.per_task == {}
        assert adapter.PostProcess({"postprocess": {"enabled": True}}).enabled is True

    def test_presses_a_grasp_tighter(self):
        """Check that presses a grasp tighter."""
        pp = _on()
        out = pp.gripper(_steps([(0.34, 0.34)]), 0.05)
        assert out[0]["left_ee_joint_state"][0] == pytest.approx(0.29, abs=1e-6)

    def test_keeps_full_open_intact(self):
        """Check that keeps full open intact."""
        pp = _on()
        for value in (0.90, 0.95, 0.99, 1.0):
            out = pp.gripper(_steps([(value, value)]), 0.05)
            assert out[0]["left_ee_joint_state"][0] == pytest.approx(value, abs=1e-6)

    def test_closing_on_air_stays_near_the_training_range(self):
        """Check that closing on air stays near the training range."""
        pp = _on()
        out = pp.gripper(_steps([(0.34, 0.34)]), 0.05)
        assert out[0]["left_ee_joint_state"][0] > 0.25
        assert out[0]["left_ee_joint_state"][0] != pytest.approx(0.0)

    def test_never_commands_below_zero(self):
        """Check that never commands below zero."""
        pp = _on()
        out = pp.gripper(_steps([(0.04, 0.04)]), 0.05)
        assert out[0]["left_ee_joint_state"][0] == pytest.approx(0.0)

    def test_joints_untouched_and_input_not_mutated(self):
        pp = _on()
        steps = _steps([(0.3, 0.9)])
        steps[0]["left_arm_joint_state"] = np.arange(6, dtype=np.float32)
        out = pp.gripper(steps, 0.05)
        assert np.allclose(out[0]["left_arm_joint_state"], np.arange(6))
        assert steps[0]["left_ee_joint_state"][0] == pytest.approx(0.3)

    def test_dtype_preserved(self):
        out = _on().gripper(_steps([(0.3, 0.3)]), 0.05)
        assert out[0]["left_ee_joint_state"].dtype == np.float32

    def test_counts_for_the_status_line(self):
        pp = _on()
        pp.gripper(_steps([(0.3, 0.99), (0.3, 0.99)]), 0.05)
        assert (pp.last_squeezed, pp.last_total) == (2, 4)

    def test_zero_squeeze_passes_through(self):
        pp = _on()
        out = pp.gripper(_steps([(0.34, 0.34)]), 0.0)
        assert out[0]["left_ee_joint_state"][0] == pytest.approx(0.34)
        assert (pp.last_squeezed, pp.last_total) == (0, 0)

    def test_disabled_passes_through(self):
        pp = adapter.PostProcess({"postprocess": {"enabled": False}})
        out = pp.gripper(_steps([(0.34, 0.34)]), 0.05)
        assert out[0]["left_ee_joint_state"][0] == pytest.approx(0.34)

    def test_per_task_overrides_default(self):
        pp = adapter.PostProcess(
            {"postprocess": {"gripper": {"squeeze": 0.05, "per_task": {"stand_up_bottles": 0.12}}}}
        )
        assert pp.squeeze_for(4) == pytest.approx(0.12)
        assert pp.squeeze_for(2) == pytest.approx(0.05)

    def test_per_task_accepts_canonical_instruction(self):
        pp = adapter.PostProcess({"postprocess": {"gripper": {"per_task": {"Stand up the bottles.": 0.12}}}})
        assert pp.squeeze_for(4) == pytest.approx(0.12)

    @pytest.mark.parametrize("bad", [-0.1, 0.6, 1.0])
    def test_rejects_out_of_range_squeeze(self, bad):
        with pytest.raises(ValueError, match="越界"):
            adapter.PostProcess({"postprocess": {"gripper": {"squeeze": bad}}})

    @pytest.mark.parametrize("bad", [0.2, 0.49, 1.5])
    def test_rejects_bad_squeeze_below(self, bad):
        with pytest.raises(ValueError, match="squeeze_below"):
            adapter.PostProcess({"postprocess": {"gripper": {"squeeze_below": bad}}})

    def test_rejects_unknown_task_key(self):
        with pytest.raises(ValueError, match="不是已知任务"):
            adapter.PostProcess({"postprocess": {"gripper": {"per_task": {"stack_the_boxes": 0.1}}}})

    def test_rejects_typos(self):
        with pytest.raises(ValueError, match="未知字段"):
            adapter.PostProcess({"postprocess": {"gripper": {"sqeeze": 0.05}}})
        with pytest.raises(ValueError, match="未知字段"):
            adapter.PostProcess({"postprocess": {"gripper_threshold": {}}})


LIMITS = [[-2.617994, 2.617994], [0.0, 3.141593], [-2.96706, 0.0],
          [-1.553344, 1.553344], [-1.553344, 1.553344], [-3.141593, 3.141593]]


def _joint_steps(*, left=None, right=None):
    """One action step with explicit arm joint targets (grippers at 0)."""
    step = {
        "left_arm_joint_state": np.asarray(left if left is not None else [0.0] * 6, np.float32),
        "left_ee_joint_state": np.array([0.0], np.float32),
        "right_arm_joint_state": np.asarray(right if right is not None else [0.0] * 6, np.float32),
        "right_ee_joint_state": np.array([0.0], np.float32),
    }
    return [step]


class TestJointClip:
    """2026-09-16: a ~1.7 deg overshoot on J3 faulted a whole episode
    (client validate_chunk rejects the chunk beyond 0.02 rad)."""

    def test_absent_limits_leave_actions_alone(self):
        pp = adapter.PostProcess({})
        assert pp.joint_low is None and pp.joint_high is None
        out = pp.joints(_joint_steps(right=[0.0, 0.0, 0.5, 0.0, 0.0, 0.0]))
        assert out[0]["right_arm_joint_state"][2] == pytest.approx(0.5)

    def test_clips_the_overshooting_joint_only(self):
        pp = adapter.PostProcess({"joint_limits": LIMITS})
        given = [0.1, 0.2, 0.029187, 0.3, 0.4, 0.5]
        q = pp.joints(_joint_steps(right=given))[0]["right_arm_joint_state"]
        assert q[2] == pytest.approx(0.0, abs=1e-7)          # J3 上限 = 0
        for i, expected in enumerate(given):
            if i != 2:
                assert q[i] == pytest.approx(expected, abs=1e-7)

    def test_lower_bound_also_enforced(self):
        pp = adapter.PostProcess({"joint_limits": LIMITS})
        out = pp.joints(_joint_steps(left=[0.0] * 6, right=[0.0, -0.05, 0.0, 0.0, 0.0, 0.0]))
        assert out[0]["right_arm_joint_state"][1] == pytest.approx(0.0, abs=1e-7)

    def test_counts_and_reports_the_worst_correction(self):
        pp = adapter.PostProcess({"joint_limits": LIMITS})
        pp.joints(_joint_steps(right=[0.0, -0.03, 0.029187, 0.0, 0.0, 0.0]))
        assert pp.last_clipped == 2
        assert pp.last_clip_max == pytest.approx(0.03, abs=1e-6)

    def test_in_range_actions_report_zero(self):
        # J3 的限位是 [-2.96706, 0],所以合法姿势里它必须是负的。
        in_range = [0.1, 0.1, -0.1, 0.1, 0.1, 0.1]
        pp = adapter.PostProcess({"joint_limits": LIMITS})
        pp.joints(_joint_steps(left=in_range, right=in_range))
        assert (pp.last_clipped, pp.last_clip_max) == (0, 0.0)

    def test_dtype_preserved_and_input_not_mutated(self):
        pp = adapter.PostProcess({"joint_limits": LIMITS})
        steps = _joint_steps(right=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
        out = pp.joints(steps)
        assert out[0]["right_arm_joint_state"].dtype == np.float32
        assert steps[0]["right_arm_joint_state"][2] == pytest.approx(0.1)

    def test_rejects_malformed_tables(self):
        for bad in ([], [[0.0, 1.0]] * 5, [[0.0, 1.0]] * 7, "x", [[1.0, 0.0]] + [[0.0, 1.0]] * 5):
            with pytest.raises(ValueError):
                adapter.PostProcess({"joint_limits": bad})

    def test_shipped_config_carries_the_sdk_table(self):
        """server.yaml 的 joint_limits 必须与 SDK 实读值一致(见配置里的来源注释)。"""
        import yaml

        path = Path(__file__).resolve().parents[1] / "configs" / "goai_real" / "server.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        pp = adapter.PostProcess(cfg)
        assert pp.joint_low is not None, "server.yaml 丢了 joint_limits"
        assert list(pp.joint_low) == [row[0] for row in LIMITS]
        assert list(pp.joint_high) == [row[1] for row in LIMITS]


class TestRawStats:
    """Check deployment output statistics independently of postprocessing."""

    def test_collects_even_when_disabled(self):
        pp = adapter.PostProcess({"postprocess": {"enabled": False}})
        pp.gripper(_steps([(0.2, 0.8), (0.99, 0.99)]), 0.05)
        out = pp.raw_summary()
        assert "n=4" in out
        assert "0.200..0.990" in out

    def test_reports_how_many_would_open_at_each_probe(self):
        pp = adapter.PostProcess({})
        pp.gripper(_steps([(0.2, 0.2), (0.99, 0.99)]), 0.05)
        out = pp.raw_summary()
        for probe in adapter._PROBE_THRESHOLDS:
            assert f"{probe:g}:50%" in out, probe
        assert "均值 0.595" in out

    def test_empty_before_any_inference(self):
        assert adapter.PostProcess({}).raw_summary() == ""

    def test_reset_clears_the_window(self):
        pp = adapter.PostProcess({})
        pp.gripper(_steps([(0.9, 0.9)]), 0.05)
        assert pp.raw_summary() != ""
        pp.reset_raw_stats()
        assert pp.raw_summary() == ""

    def test_shipped_config_loads(self):
        """Check that shipped config loads."""
        import yaml

        path = Path(__file__).resolve().parents[1] / "configs" / "goai_real" / "server.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        pp = adapter.PostProcess(cfg)
        assert pp.enabled is False, "未验证的修正层必须默认关掉;要开得显式改 server.yaml"
        assert 0 < pp.squeeze < 0.5
        assert 0.5 <= pp.squeeze_below <= 1.0

        assert pp.squeeze_below > 0.73


def test_model_prefix_rejects_empty_images_before_embedding():
    from pi.models_pytorch.pi0_pytorch import PI0Pytorch

    with pytest.raises(ValueError, match="At least one image"):
        PI0Pytorch.embed_prefix(SimpleNamespace(), [], [], None, None)


@pytest.mark.parametrize("slot,instruction", list(enumerate(adapter.GOAI_REAL_TASK_INSTRUCTIONS)))
def test_official_instructions_preserve_legacy_embedding_slots(model, slot, instruction):
    assert adapter.resolve_real_task(instruction) == (slot, adapter.GOAI_REAL_TASK_INSTRUCTIONS[slot])
    assert adapter.resolve_real_task_nearest(instruction)[2:] == (1.0, True)
    obs = observation()
    obs["instruction"] = instruction
    obs["task_name"] = adapter._SLUGS[slot]
    model.update_obs(obs)
    model.get_action()
    assert model.policy.calls[-1][1] == slot
    model.reset()
    obs["instruction"] = adapter.GOAI_REAL_LEGACY_TASK_INSTRUCTIONS[slot]
    model.update_obs(obs)
    model.get_action()
    assert model.policy.calls[-1][1:] == (slot, 42, 0)


def test_stats_must_match_selected_training_entry(checkpoint):
    manifest_path = checkpoint.parent / "norm_stats_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    changed = b'{"norm_stats": {"wrong": true}}'
    (checkpoint.parent / "norm_stats_pt.json").write_bytes(changed)
    manifest["files"].append({"checkpoint_path": "other.json", "sha256": hashlib.sha256(changed).hexdigest()})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA256"):
        adapter.validate_checkpoint(checkpoint)


def test_duplicate_training_entries_rejected(checkpoint):
    path = checkpoint.parent / "norm_stats_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"].append(dict(manifest["files"][0]))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="training entry"):
        adapter.validate_checkpoint(checkpoint)
