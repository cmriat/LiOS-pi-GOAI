"""Adapt official GOAI observations to the Pi05 model input contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from pi.shared import image_tools

GOAI_ACTION_DIMS = {"joint": 14, "ee": 16}
GOAI_CAMERA_MAP = {
    "cam_high": "base_0_rgb",
    "cam_left_wrist": "left_wrist_0_rgb",
    "cam_right_wrist": "right_wrist_0_rgb",
}
GOAI_SOURCE_SIZE = (480, 640)
GOAI_LANCE_SIZE = (640, 640)
GOAI_MODEL_SIZE = (224, 224)


def _to_hwc_uint8(image: Any, camera: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"{camera} must be a 3D RGB image, got shape {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"{camera} must have three RGB channels, got shape {array.shape}")
    if array.shape[:2] != GOAI_SOURCE_SIZE:
        raise ValueError(
            f"{camera} must be {GOAI_SOURCE_SIZE[1]}x{GOAI_SOURCE_SIZE[0]}, got {array.shape[1]}x{array.shape[0]}"
        )

    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = array * scale
    return np.clip(array, 0, 255).astype(np.uint8)


def prepare_goai_image(image: Any, camera: str) -> np.ndarray:
    """Apply the exact GOAI geometry sequence used by conversion and training."""
    image_hwc = _to_hwc_uint8(image, camera)
    square = image_tools.resize_with_pad(image_hwc, *GOAI_LANCE_SIZE)
    return image_tools.resize_with_pad(square, *GOAI_MODEL_SIZE)


def prepare_goai_observation(observation: dict[str, Any], action_space: str) -> dict[str, Any]:
    """Validate and map one decoded XPolicyLab GOAI observation for Pi05 inference."""
    if action_space not in GOAI_ACTION_DIMS:
        raise ValueError(f"action_space must be joint or ee, got {action_space!r}")
    if "state" not in observation or "images" not in observation:
        raise KeyError("GOAI observation requires state and images")

    state = np.asarray(observation["state"], dtype=np.float32)
    expected_dim = GOAI_ACTION_DIMS[action_space]
    if state.shape != (expected_dim,):
        raise ValueError(f"GOAI {action_space} state must have shape ({expected_dim},), got {state.shape}")

    source_images = observation["images"]
    images = {}
    for source_key, model_key in GOAI_CAMERA_MAP.items():
        if source_key not in source_images:
            raise KeyError(f"GOAI observation is missing images[{source_key!r}]")
        images[model_key] = prepare_goai_image(source_images[source_key], source_key)

    prompt = observation.get("instruction", observation.get("prompt"))
    if prompt is not None:
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("GOAI instruction must not be empty")

    result = {
        "state": state,
        "image": images,
        "image_mask": {key: np.True_ for key in images},
        "prompt": prompt,
    }
    if "task_index" in observation:
        result["task_index"] = observation["task_index"]
    return result
