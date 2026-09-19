"""Check the official debug fixture and policy callback integration."""

import numpy as np
from XPolicyLab import debug_env_client as debug
from XPolicyLab.policy.Pi_05 import deploy as official_deploy
from XPolicyLab.policy.lionvla import deploy
from XPolicyLab.utils.process_data import decode_obs_images, get_robot_action_dim_info
from scripts.inference.goai.debug_xpolicylab_client import TaskFixture

from pi.inference.goai_xpolicylab import canonical_observation


def make_env(encoded=False):
    env = object.__new__(debug.TestEnv)
    env.robot_action_dim_info = get_robot_action_dim_info("arx_x5")
    env.obs_encoded = encoded
    return env


def test_deploy_reuses_official_functions():
    assert deploy.eval_one_episode is official_deploy.eval_one_episode
    assert deploy.eval_one_episode_batch is official_deploy.eval_one_episode_batch


def test_fixture_preserves_full_official_observation():
    fixture = TaskFixture(make_env(), "Stack the bowls")
    obs = fixture.get_obs(9)
    assert obs["instruction"] == "Stack the bowls"
    assert obs["env_idx"] == 9
    assert len(obs["state"]) == 11
    assert len(obs["vision"]["cam_head"]) == 5
    assert obs["additional_info"]["frequency"] == 30
    assert obs["data_format_version"] == "v1.0"
    adapted = canonical_observation(obs)
    assert adapted["state"].shape == (14,)
    assert set(adapted["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}


def test_official_mixed_jpeg_preserves_asymmetric_color_and_extras():
    env = make_env()
    obs = env.get_obs()
    obs["vision"]["cam_head"]["color"][:] = (17, 91, 203)
    obs["vision"]["cam_right_wrist"]["color"][:] = (199, 73, 11)
    depth = obs["vision"]["cam_head"]["depth"]
    env._encode_obs_colors(obs)
    assert obs["vision"]["cam_head"]["color"].ndim == 1
    assert isinstance(obs["vision"]["cam_left_wrist"]["color"], bytes)
    decoded = decode_obs_images(obs)
    assert decoded["vision"]["cam_head"]["depth"] is depth
    adapted = canonical_observation(decoded)
    for name in ("cam_high", "cam_left_wrist"):
        np.testing.assert_allclose(adapted["images"][name].mean(axis=(0, 1)), (17, 91, 203), rtol=0, atol=3)
    np.testing.assert_array_equal(adapted["images"]["cam_right_wrist"][0, 0], (199, 73, 11))
