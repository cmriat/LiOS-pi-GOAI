"""CPU coverage for the training collate contract in ``scripts/train/data_loader.py``.

The collate is the only bridge between a dataset sample and the model's ``Observation``:
a key it drops is a key the model never sees, and Pi0.5 raises outright when task or
embodiment conditioning is enabled and its index is missing.

No camera list is written down here, because none is written down in the training chain:
the dataset emits the cameras of the recording it read and the model augments
``observation.images.keys()``. A hardcoded 4-camera expectation used to sit in collate and
raised against 3-camera GOAI data before ``Observation.from_dict`` was ever reached, so
these samples come from the real producer (``pi.data._data_inputs``) and the camera set is
cross-checked against the set the frozen serving chain hands the model.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import torch
import pytest

# The training dataloader imports the training stack (wandb) and pi.data (lerobot).
pytest.importorskip("wandb")
pytest.importorskip("lerobot")

import pi.data as pi_data  # noqa: E402
from pi.training import instance_config  # noqa: E402
from pi.models.model import Observation  # noqa: E402
from pi.inference.goai_sim_policy import GOAI_MODEL_IMAGE_KEYS  # noqa: E402
from pi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch  # noqa: E402

# The trainer puts scripts/train on sys.path ("from utils import ...").
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train"))
from utils import run_test_evaluation  # noqa: E402
from data_loader import collate_and_preprocess  # noqa: E402

# The configs the training shell scripts launch and the server serves.
GOAI_CONFIG_NAMES = ("pi05_goai_joint", "pi05_goai_joint_lance", "pi05_b1k_goai")

IMAGE_RESOLUTION = (224, 224)
STATE_DIM = 14
ACTION_HORIZON = 8
ACTION_DIM = 32
CAMERA_SHAPE = (480, 640, 3)


def _raw_step() -> dict:
    """A raw LeRobot step, keyed the way ``_data_inputs`` reads it."""
    ramp = (np.arange(np.prod(CAMERA_SHAPE), dtype=np.uint32) % 256).astype(np.uint8).reshape(CAMERA_SHAPE)
    return {
        "observation/state": np.zeros(STATE_DIM, dtype=np.float32),
        "observation/cam_env": ramp,
        "observation/cam_left_wrist": ramp.copy(),
        "observation/cam_right_wrist": ramp.copy(),
        "prompt": "pick up the cube",
        "actions": np.zeros((ACTION_HORIZON, STATE_DIM), dtype=np.float32),
    }


def _sample(*, task_index: int | None = None, embodiment_index: int | None = None) -> dict:
    """One dataset sample, as the dataset's ``_transform`` leaves it for the collate."""
    step = pi_data._data_inputs(_raw_step())
    step.pop("prompt")  # _tokenize_prompt consumes the raw string before batching
    step = pi_data._resize_images(step, *IMAGE_RESOLUTION)
    step = pi_data._pad_state_actions(step, ACTION_DIM)
    step["tokenized_prompt"] = np.zeros(8, dtype=np.int64)
    step["tokenized_prompt_mask"] = np.ones(8, dtype=bool)
    if task_index is not None:
        step["task_index"] = np.int64(task_index)
    if embodiment_index is not None:
        step["embodiment_index"] = np.int64(embodiment_index)
    return step


def test_batch_carries_the_cameras_the_dataset_produced() -> None:
    """Three cameras, named by the dataset, matching the set the server serves."""
    batch = collate_and_preprocess([_sample()])

    assert len(GOAI_MODEL_IMAGE_KEYS) == 3
    assert set(batch["image"]) == set(GOAI_MODEL_IMAGE_KEYS)
    assert set(batch["image_mask"]) == set(GOAI_MODEL_IMAGE_KEYS)


def test_collate_leaves_images_uint8_for_the_model_to_preprocess() -> None:
    """Images reach the model as the uint8 ``[H, W, C]`` tensors the dataset produced.

    ``Observation.from_dict`` is the one place that converts them and ``PI0Pytorch.forward``
    the one place that augments them; preprocessing here as well would augment twice.
    """
    batch = collate_and_preprocess([_sample()])

    for key, image in batch["image"].items():
        assert image.dtype == torch.uint8, key
        assert tuple(image.shape) == (1, *IMAGE_RESOLUTION, 3), key


def test_three_camera_sample_reaches_the_model_preprocessing() -> None:
    """Collate -> ``Observation.from_dict`` -> model preprocessing keeps the 3 cameras.

    This is the chain a collate-side 4-camera expectation broke: it raised with "images
    dict missing keys: ... 'base_1_rgb'" before the observation was ever built.
    """
    batch = collate_and_preprocess([_sample(task_index=3, embodiment_index=1)])

    processed = preprocess_observation_pytorch(Observation.from_dict(batch), train=True)

    assert set(processed.images) == set(GOAI_MODEL_IMAGE_KEYS)
    for key, image in processed.images.items():
        assert tuple(image.shape) == (1, 3, *IMAGE_RESOLUTION), key
        assert image.dtype == torch.float32, key
        assert float(image.min()) >= -1.0, key
        assert float(image.max()) <= 1.0, key
    assert processed.task_index.tolist() == [3]
    assert processed.embodiment_index.tolist() == [1]


class _ObservationRecorder(torch.nn.Module):
    """Stands in for the policy: records what the test loop hands it."""

    def __init__(self) -> None:
        super().__init__()
        self.observations: list[object] = []

    def forward(self, observation, actions):
        self.observations.append(observation)
        return actions.sum() * 0.0


def test_test_evaluation_hands_the_model_an_observation() -> None:
    """The per-epoch test pass builds the same ``Observation`` the training loop does.

    ``instance_config.pi05_robotwin`` sets ``test_ep_num`` and ``test_step_interval``, so
    this pass runs; a raw collated dict is not something the policy can preprocess.
    """
    model = _ObservationRecorder()

    loss = run_test_evaluation(
        model,
        [collate_and_preprocess([_sample(task_index=3, embodiment_index=1)])],
        torch.device("cpu"),
        rank=0,
        is_main=True,
        epoch=0,
    )

    assert loss == 0.0
    (observation,) = model.observations
    assert isinstance(observation, Observation)
    assert set(observation.images) == set(GOAI_MODEL_IMAGE_KEYS)
    assert observation.task_index.tolist() == [3]


def test_collate_keeps_task_and_embodiment_indices() -> None:
    batch = collate_and_preprocess(
        [_sample(task_index=3, embodiment_index=1), _sample(task_index=5, embodiment_index=1)]
    )

    assert batch["task_index"].tolist() == [3, 5]
    assert batch["embodiment_index"].tolist() == [1, 1]


def test_absent_indices_reach_the_model_as_none() -> None:
    """Language-conditioned samples carry no index; the observation is built without them."""
    observation = Observation.from_dict(collate_and_preprocess([_sample()]))

    assert observation.task_index is None
    assert observation.embodiment_index is None


@pytest.mark.parametrize("config_name", GOAI_CONFIG_NAMES)
def test_goai_configs_get_the_index_pi0_pytorch_asks_for(config_name: str) -> None:
    """Each GOAI config hands the model a lookup index whenever its conditioning is on.

    Pi0.5 raises in ``_prepare_task_indices`` / ``_embed_embodiment_tokens`` when the index
    the config enables is missing, so a batch that cannot carry it would stop training.
    """
    model_config = instance_config.get_config(config_name).model
    observation = Observation.from_dict(collate_and_preprocess([_sample(task_index=3, embodiment_index=1)]))

    if model_config.use_task_embedding:
        assert observation.task_index is not None
    if model_config.use_embodiment_embedding:
        assert observation.embodiment_index is not None
