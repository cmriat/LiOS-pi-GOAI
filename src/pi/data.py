# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
"""Bare-bones dataset loader without transform abstractions.

Reads either a LeRobot dataset directory or a VideoLance dataset, repacks the
columns into the intermediate layout shared by all platforms, applies the delta
action transform, normalization and tokenization, and batches samples into a
tree of tensors.

Also includes a multi-dataset variant that iterates and batches across several
datasets using one `DatasetConfig` per dataset.
"""

from __future__ import annotations

import json
import bisect
import logging
from collections.abc import Iterator, Sequence

import numpy as np
import torch
import einops
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from pi.models import tokenizer as tokenizer_mod
from pi.shared import image_tools
from pi.training import config as _config
from pi.training.config import DatasetConfig
from pi.shared.embodiment import (
    EMBODIMENT_METADATA_KEY,
    LEGACY_EMBODIMENT_METADATA_KEY,
    resolve_dataset_embodiment,
)
from pi.shared.goai_tasks import build_task_remap
from pi.shared.goai_state_contract import (
    GOAI_JOINT_DELTA_MASK,
    GOAI_SOURCE_STATE_DIM,
    goai_policy_state_indices,
    resolve_goai_policy_state_schema,
)


class _EpisodeIndexMapper:
    """Picklable wrapper for episode index mapping (required for spawn mode).

    This class replaces the closure-based approach which cannot be pickled
    when using multiprocessing with spawn mode.
    """

    def __init__(self, original_method, episode_index_map: dict):
        self._original_method = original_method
        self._episode_index_map = episode_index_map

    def __call__(self, idx: int, ep_idx: int):
        mapped_ep_idx = self._episode_index_map.get(ep_idx, ep_idx)
        return self._original_method(idx, mapped_ep_idx)


def _dataset_uri_values(dataset_uri: str | Sequence[str] | None, fallback: str) -> list[str]:
    """Normalize `dataset_uri` into the list of URIs a run reads.

    Accepts the comma-separated form used by the training shell scripts as well as
    an explicit sequence; `fallback` (usually `repo_id`) covers configs that only
    name a dataset.
    """
    if dataset_uri is None:
        return [fallback]
    if isinstance(dataset_uri, str):
        values = [uri.strip() for uri in dataset_uri.split(",") if uri.strip()]
        return values or [fallback]
    return [str(uri) for uri in dataset_uri if str(uri)]


def _repack_transform(policy_name: str, sample: dict, dataset_format: str = "lerobot") -> dict:
    if dataset_format == "video_lance":
        cam_env_key = "observation_images_cam_env"
        cam_left_wrist_key = "observation_images_cam_left_wrist"
        cam_right_wrist_key = "observation_images_cam_right_wrist"
    else:
        cam_env_key = "observation.images.cam_env"
        cam_left_wrist_key = "observation.images.cam_left_wrist"
        cam_right_wrist_key = "observation.images.cam_right_wrist"

    policy_lower = policy_name.lower()
    if "robotwin" in policy_lower:
        result: dict[str, object] = {
            "observation/cam_env": sample["head_image"],
            "observation/cam_left_wrist": sample["left_wrist_image"],
            "observation/cam_right_wrist": sample["right_wrist_image"],
            "observation/state": sample["state"],
        }
    elif any(name in policy_lower for name in ("airbot", "goai")):
        result: dict[str, object] = {
            "observation/cam_env": sample[cam_env_key],
            "observation/cam_left_wrist": sample[cam_left_wrist_key],
            "observation/cam_right_wrist": sample[cam_right_wrist_key],
            "observation/state": sample["observation.state"],
        }
    else:
        raise ValueError(f"Unsupported policy: {policy_name}")

    if "action" in sample:
        result["actions"] = sample["action"]
    if "task" in sample:
        result["prompt"] = sample["task"]
    if "task_index" in sample:
        result["task_index"] = sample["task_index"]

    return result


def _make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        _make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        _make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * dim)
        else:
            result.extend([False] * (-dim))
    return tuple(result)


GOAI_LANCE_IMAGE_WIDTH = 640
GOAI_LANCE_IMAGE_HEIGHT = 640
GOAI_SOURCE_IMAGE_WIDTH = 640
GOAI_SOURCE_IMAGE_HEIGHT = 480
GOAI_LANCE_CAMERA_KEYS = (
    "observation_images_cam_env",
    "observation_images_cam_left_wrist",
    "observation_images_cam_right_wrist",
)


def resolve_policy_state_schema(policy_name: str, value: str | None) -> str:
    """Resolve a policy's model-visible state schema."""
    normalized = "source" if value is None else str(value).strip().lower()
    if "goai" in policy_name.lower():
        # The GOAI dataset state already is the 14D contract layout, so "source"
        # and the explicit 14D schema describe the same projection; naming it
        # explicitly still stamps the layout into the checkpoint contract.
        return resolve_goai_policy_state_schema(normalized)
    if normalized not in {"source", "identity"}:
        raise ValueError(f"{policy_name} only supports policy_state_schema='source', got {value!r}")
    return "source"


def policy_state_dim(policy_name: str, value: str | None, source_state_dim: int) -> int:
    """Return the state dimension visible to the policy."""
    resolved = resolve_policy_state_schema(policy_name, value)
    if "goai" in policy_name.lower():
        return len(goai_policy_state_indices(resolved))
    return int(source_state_dim)


def project_policy_state(state: np.ndarray, policy_name: str, value: str | None) -> np.ndarray:
    """Project a source state into the model-visible state without mutating it."""
    array = np.asarray(state)
    if "goai" not in policy_name.lower():
        return array
    resolved = resolve_policy_state_schema(policy_name, value)
    if array.shape[-1] != GOAI_SOURCE_STATE_DIM:
        raise ValueError(f"GOAI source state must be {GOAI_SOURCE_STATE_DIM}D, got shape {array.shape}")
    return np.take(array, goai_policy_state_indices(resolved), axis=-1)


def _validate_goai_lance_image_metadata(metadata: dict[bytes, bytes], dataset_uri: str) -> dict[str, object]:
    """Validate GOAI geometry, including trusted robot_to_lance read-time padding."""
    dimension_keys = (b"video_lance:image_width", b"video_lance:image_height")
    missing_dimensions = [key.decode() for key in dimension_keys if key not in metadata]
    if missing_dimensions:
        raise ValueError(f"GOAI Lance dataset {dataset_uri} is missing image dimensions: {missing_dimensions}")

    width = int(metadata[b"video_lance:image_width"])
    height = int(metadata[b"video_lance:image_height"])
    source = metadata.get(b"video_lance:source", b"").decode()
    if source == "robot_to_lance" and (width, height) == (GOAI_SOURCE_IMAGE_WIDTH, GOAI_SOURCE_IMAGE_HEIGHT):
        logging.warning(
            "GOAI Lance dataset %s stores %dx%d robot frames; applying the shared square pad at read time",
            dataset_uri,
            width,
            height,
        )
        return {
            "width": GOAI_LANCE_IMAGE_WIDTH,
            "height": GOAI_LANCE_IMAGE_HEIGHT,
            "resize_mode": "pad",
            "source_width": GOAI_SOURCE_IMAGE_WIDTH,
            "source_height": GOAI_SOURCE_IMAGE_HEIGHT,
        }

    geometry_keys = (b"video_lance:resize_mode", b"video_lance:source_camera_sizes_json")
    missing_geometry = [key.decode() for key in geometry_keys if key not in metadata]
    if missing_geometry:
        raise ValueError(f"GOAI Lance dataset {dataset_uri} is missing image geometry metadata: {missing_geometry}")

    if (width, height) != (GOAI_LANCE_IMAGE_WIDTH, GOAI_LANCE_IMAGE_HEIGHT):
        raise ValueError(
            f"GOAI Lance dataset {dataset_uri} must be "
            f"{GOAI_LANCE_IMAGE_WIDTH}x{GOAI_LANCE_IMAGE_HEIGHT}, got {width}x{height}. "
            "Reconvert the 640x480 simulator images with square padding."
        )

    resize_mode = metadata[b"video_lance:resize_mode"].decode()
    if resize_mode != "pad":
        raise ValueError(f"GOAI Lance resize mode must be pad, got {resize_mode!r}")

    try:
        source_sizes = json.loads(metadata[b"video_lance:source_camera_sizes_json"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"GOAI Lance source camera size metadata is invalid: {error}") from error
    missing_cameras = sorted(set(GOAI_LANCE_CAMERA_KEYS).difference(source_sizes))
    if missing_cameras:
        raise ValueError(f"GOAI Lance source camera metadata is missing: {missing_cameras}")
    wrong_sizes = {
        key: value
        for key, value in source_sizes.items()
        if int(value["width"]) != GOAI_SOURCE_IMAGE_WIDTH or int(value["height"]) != GOAI_SOURCE_IMAGE_HEIGHT
    }
    if wrong_sizes:
        raise ValueError(
            "GOAI source camera images must be "
            f"{GOAI_SOURCE_IMAGE_WIDTH}x{GOAI_SOURCE_IMAGE_HEIGHT}, got: {wrong_sizes}"
        )

    return {
        "width": width,
        "height": height,
        "resize_mode": resize_mode,
        "source_width": GOAI_SOURCE_IMAGE_WIDTH,
        "source_height": GOAI_SOURCE_IMAGE_HEIGHT,
    }


def _get_delta_action_mask(policy_name: str) -> tuple[bool, ...]:
    """Get delta action mask based on policy/robot type.

    Args:
        policy_name: Policy name (e.g., 'pi05_goai_joint', 'pi05_airbot')

    Returns:
        Boolean mask indicating which action dimensions should be delta-transformed.
        True = delta (relative to current state), False = absolute value

    Examples:
        GOAI real-robot and Airbot (14-dim): 6 joints + 1 absolute gripper per arm
        >>> _get_delta_action_mask("pi05_goai_joint")
        (True, True, True, True, True, True, False,
         True, True, True, True, True, True, False)
    """
    policy_lower = policy_name.lower()
    if "goai" in policy_lower:
        return GOAI_JOINT_DELTA_MASK
    if "airbot" in policy_lower or "robotwin" in policy_lower:
        return _make_bool_mask(6, -1, 6, -1)
    raise ValueError(f"Unsupported policy for delta actions: {policy_name}")


def _apply_delta_actions(step: dict, mask: tuple[bool, ...]) -> dict:
    """将绝对动作转换为相对动作(delta actions).

    参考 transforms.DeltaActions 的实现。
    这个转换会在训练时应用，将 actions 从绝对空间转换为相对于 state 的增量。

    Args:
        step: 包含 'state' 和 'actions' 的数据字典
        mask: 布尔掩码，指定哪些动作维度需要转换为 delta

    Returns:
        转换后的数据字典
    """
    if "actions" not in step:
        return step

    state, actions = step["state"], step["actions"]
    mask_array = np.asarray(mask)
    dims = mask_array.shape[-1]

    # If state has multiple frames (history), use only the first frame (most recent state)
    # state shape could be (history_frames, state_dim) or (state_dim,)
    current_state = state[0] if state.ndim > 1 else state
    # delta_action = current_actions - current_state
    actions[..., :dims] -= np.expand_dims(np.where(mask_array, current_state[..., :dims], 0), axis=-2)
    step["actions"] = actions

    return step


def _data_inputs(data: dict) -> dict:
    def _to_uint8_image(array: np.ndarray) -> np.ndarray:
        image = np.asarray(array)
        if np.issubdtype(image.dtype, np.floating):
            image = (255.0 * image).astype(np.uint8)
        if image.ndim == 3 and image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")
        return image.astype(np.uint8, copy=False)

    result = {
        "state": np.asarray(data["observation/state"]),
        "image": {
            "base_0_rgb": _to_uint8_image(data["observation/cam_env"]),
            "left_wrist_0_rgb": _to_uint8_image(data["observation/cam_left_wrist"]),
            "right_wrist_0_rgb": _to_uint8_image(data["observation/cam_right_wrist"]),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        "prompt": data.get("prompt"),
    }
    if "task_index" in data:
        result["task_index"] = data["task_index"]
    if "actions" in data:
        result["actions"] = np.asarray(data["actions"])
    return result


def _normalize_array(x: np.ndarray, stats, use_quantiles: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile stats required when use_quantiles=True")
        q01 = stats.q01[..., : x.shape[-1]]
        q99 = stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    mean = stats.mean[..., : x.shape[-1]]
    std = stats.std[..., : x.shape[-1]]
    # Dimensions with zero spread are guarded by an epsilon; scaled dimensions divide
    # by their exact std so normalization stays a no-op-faithful affine map.
    return (x - mean) / np.where(std == 0.0, 1e-6, std)


def _resize_images(data: dict, height: int, width: int) -> dict:
    result = dict(data)
    result["image"] = {key: image_tools.resize_with_pad(img, height, width) for key, img in data["image"].items()}
    return result


def _tokenize_prompt(
    data: dict,
    tokenizer,
    *,
    discrete_state_input: bool,
    use_task_embedding: bool = False,
    use_language_with_task_embedding: bool = False,
    num_tasks: int = 0,
) -> dict:
    result = dict(data)
    prompt = result.pop("prompt", None)

    # For discrete_state_input: only use the most recent state frame for tokenization with prompt
    # Historical states are preserved in result["state"] for later use by the action head
    # If state has history (shape: (history_frames, state_dim)), extract state[0] (most recent)
    # If state is single frame (shape: (state_dim,)), use it directly
    state_arg = None
    if discrete_state_input:
        state = result["state"]
        if state.ndim > 1:
            # Multi-frame state: state[0] is the most recent frame (t-delay)
            # Note: with delta_timestamps, state[0] = t-state_delay_frames, state[-1] is oldest
            state_arg = state[0]
        else:
            # Single-frame state: use directly
            state_arg = state

    if use_task_embedding:
        if not discrete_state_input or state_arg is None:
            raise ValueError("Task embedding requires a discrete state input.")
        if "task_index" not in result:
            raise ValueError("task_index is required when use_task_embedding=True")
        task_index = np.asarray(result["task_index"])
        if task_index.size != 1:
            raise ValueError(f"Expected one task_index per sample, got shape {task_index.shape}")
        task_index = int(task_index.item())
        if task_index < 0 or task_index >= num_tasks:
            raise ValueError(f"task_index {task_index} is outside [0, {num_tasks})")
        result["task_index"] = np.int64(task_index)
        if use_language_with_task_embedding:
            if prompt is None:
                raise ValueError("Prompt is required when language and task embedding are both enabled.")
            if not isinstance(prompt, str):
                prompt = str(prompt if np.isscalar(prompt) else prompt.item())
            tokens, mask = tokenizer.tokenize(prompt, state_arg)
        else:
            tokens, mask = tokenizer.tokenize_state(state_arg)
    else:
        if use_language_with_task_embedding:
            raise ValueError("use_language_with_task_embedding requires use_task_embedding=True")
        result.pop("task_index", None)
        if prompt is None:
            raise ValueError("Prompt is required for tokenization.")
        if not isinstance(prompt, str):
            prompt = str(prompt if np.isscalar(prompt) else prompt.item())
        tokens, mask = tokenizer.tokenize(prompt, state_arg)
    result["tokenized_prompt"] = tokens
    result["tokenized_prompt_mask"] = mask
    return result


def _normalize_actions(
    actions: np.ndarray,
    stats,
    *,
    use_quantiles: bool,
    use_per_timestamp_action_norm: bool,
) -> np.ndarray:
    if not use_per_timestamp_action_norm:
        return _normalize_array(actions, stats, use_quantiles)

    actions = np.asarray(actions, dtype=np.float64)
    horizon, action_dim = actions.shape[-2], actions.shape[-1]
    if use_quantiles:
        if stats.per_timestamp_q01 is None or stats.per_timestamp_q99 is None:
            raise ValueError(
                "use_per_timestamp_action_norm=True but actions norm_stats is missing "
                "per_timestamp_q01/per_timestamp_q99."
            )
        q01 = stats.per_timestamp_q01[:horizon, :action_dim]
        q99 = stats.per_timestamp_q99[:horizon, :action_dim]
        return (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    if stats.per_timestamp_mean is None or stats.per_timestamp_std is None:
        raise ValueError(
            "use_per_timestamp_action_norm=True but actions norm_stats is missing per_timestamp_mean/per_timestamp_std."
        )
    mean = stats.per_timestamp_mean[:horizon, :action_dim]
    std = stats.per_timestamp_std[:horizon, :action_dim]
    # Same epsilon convention as the global branch above (guard only zero spread, keep the
    # affine map exact elsewhere). Note the frozen inference side inverts with ``std + 1e-6``,
    # so the round trip is off by <=1e-6 relative; every shipped config normalizes with
    # quantiles (``effective_use_quantile_norm``), so this branch is unreachable in practice.
    return (actions - mean) / np.where(std == 0.0, 1e-6, std)


def _normalize(
    data: dict,
    norm_stats: dict,
    use_quantiles: bool,
    use_per_timestamp_action_norm: bool = False,
) -> dict:
    result = dict(data)
    if "state" in result and "state" in norm_stats:
        result["state"] = _normalize_array(result["state"], norm_stats["state"], use_quantiles)
    if "actions" in result and "actions" in norm_stats:
        result["actions"] = _normalize_actions(
            result["actions"],
            norm_stats["actions"],
            use_quantiles=use_quantiles,
            use_per_timestamp_action_norm=use_per_timestamp_action_norm,
        )
    return result


def _pad_state_actions(data: dict, target_dim: int) -> dict:
    def _pad_last_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
        array = np.asarray(array, dtype=np.float64)
        if array.shape[-1] >= target_dim:
            return array
        pad_width = [(0, 0)] * array.ndim
        pad_width[-1] = (0, target_dim - array.shape[-1])
        return np.pad(array, pad_width, constant_values=0.0)

    result = dict(data)
    result["state"] = _pad_last_dim(result["state"], target_dim)
    if "actions" in result:
        result["actions"] = _pad_last_dim(result["actions"], target_dim)
    return result


def _stack_tree(items: list[dict]) -> dict:
    def stack(*values):
        first = values[0]
        if isinstance(first, dict):
            return {key: stack(*[value[key] for value in values]) for key in first}
        if isinstance(first, (list, tuple)):
            packed = [stack(*[value[idx] for value in values]) for idx in range(len(first))]
            return type(first)(packed)
        return torch.stack([torch.as_tensor(value) for value in values], dim=0)

    return stack(*items)


class SimpleLeRobotLoader:
    """Minimal iterator that batches samples from a single dataset.

    Note: `data_config` is intentionally not used here. Only the fields that were
    previously read from `data_config` are accepted directly via constructor
    parameters to avoid requiring a full TrainConfig initialization.

    The declared feature flags below are the state `__init__` fills in; they also
    let a hand-built instance (see tests/test_data_transforms.py) drive `_transform`
    with the optional features switched off.
    """

    dataset_format: str = "lerobot"
    policy_state_schema: str = "source"
    dataset_uri: str = ""
    use_per_timestamp_action_norm: bool = False
    use_task_embedding: bool = False
    use_language_with_task_embedding: bool = False
    num_tasks: int = 0
    use_embodiment_embedding: bool = False
    embodiment_index: int | None = None
    embodiment_source: str | None = None
    local_to_official_task: dict[int, int] | None = None

    def __init__(
        self,
        config: _config.TrainConfig | None = None,
        *,
        # Fields previously sourced from data_config
        repo_id: str | None = None,
        dataset_format: str | None = None,
        dataset_uri: str | Sequence[str] | None = None,
        action_sequence_keys: list[str] | None = None,
        state_sequence_keys: list[str] | None = None,
        norm_stats: dict | None = None,
        use_quantile_norm: bool | None = None,
        use_per_timestamp_action_norm: bool | None = None,
        policy_name: str | None = None,
        policy_state_schema: str | None = None,
        # Fields previously sourced from model/batch config
        batch_size: int | None = None,
        action_horizon: int | None = None,
        action_dim: int | None = None,
        max_token_len: int | None = None,
        discrete_state_input: bool | None = None,
        use_task_embedding: bool | None = None,
        use_language_with_task_embedding: bool | None = None,
        num_tasks: int | None = None,
        use_embodiment_embedding: bool | None = None,
        num_embodiments: int | None = None,
        embodiment_index: int | None = None,
        apply_delta_transform: bool | None = None,
        state_history_frames: int | None = None,
        state_delay_frames: int | None = None,
        test_ep_num: int | None = None,
        mode: str = "train",  # "train" or "test"
    ) -> None:
        # Support both explicit-args path and config path (used by scripts/inference.py).
        if config is not None:
            data_config = config.data
            # Extract policy name from config if not provided
            if policy_name is None:
                policy_name = config.name

            # Extract non-data fields from TrainConfig.
            batch_size = config.batch_size if batch_size is None else batch_size
            action_horizon = config.model.action_horizon if action_horizon is None else action_horizon
            action_dim = config.model.action_dim if action_dim is None else action_dim
            max_token_len = config.model.max_token_len if max_token_len is None else max_token_len
            if discrete_state_input is None:
                discrete_state_input = getattr(config.model, "discrete_state_input", True)
            if use_task_embedding is None:
                use_task_embedding = getattr(config.model, "use_task_embedding", False)
            if use_language_with_task_embedding is None:
                use_language_with_task_embedding = getattr(config.model, "use_language_with_task_embedding", False)
            if num_tasks is None:
                num_tasks = getattr(config.model, "num_tasks", 0)
            if use_embodiment_embedding is None:
                use_embodiment_embedding = getattr(config.model, "use_embodiment_embedding", False)
            if num_embodiments is None:
                num_embodiments = getattr(config.model, "num_embodiments", 2)
            if embodiment_index is None:
                embodiment_index = data_config.embodiment_index
            if state_history_frames is None:
                state_history_frames = getattr(config.model, "state_history_frames", 1)
            if state_delay_frames is None:
                state_delay_frames = getattr(config.model, "state_delay_frames", 0)
            if policy_state_schema is None:
                policy_state_schema = data_config.policy_state_schema

            # Read the remaining dataset params straight off the data config.
            repo_id = data_config.repo_id if repo_id is None else repo_id
            if dataset_format is None:
                dataset_format = data_config.dataset_format
            if dataset_uri is None:
                dataset_uri = data_config.dataset_uri
            action_sequence_keys = (
                list(data_config.action_sequence_keys) if action_sequence_keys is None else action_sequence_keys
            )
            state_sequence_keys = (
                list(data_config.state_sequence_keys) if state_sequence_keys is None else state_sequence_keys
            )
            if norm_stats is None:
                # Training scripts preload the stats into the config; a standalone
                # caller (scripts/inference.py) loads them from the assets directory.
                norm_stats = data_config.norm_stats or data_config.load_norm_stats(config.assets_dirs)
            use_quantile_norm = data_config.use_quantile_norm if use_quantile_norm is None else use_quantile_norm
            if use_per_timestamp_action_norm is None:
                use_per_timestamp_action_norm = data_config.use_per_timestamp_action_norm
            if apply_delta_transform is None:
                apply_delta_transform = data_config.apply_delta_transform
            if test_ep_num is None:
                test_ep_num = data_config.test_ep_num

        # Validate required fields for explicit construction.
        if repo_id is None:
            raise ValueError("repo_id must be set")
        if dataset_format is None:
            dataset_format = "lerobot"
        if norm_stats is None:
            raise ValueError("Normalization stats are required.")
        if action_sequence_keys is None:
            raise ValueError("action_sequence_keys must be set")
        if state_sequence_keys is None:
            raise ValueError("state_sequence_keys must be set")
        if batch_size is None or action_horizon is None or action_dim is None or max_token_len is None:
            raise ValueError("batch_size, action_horizon, action_dim, and max_token_len must be provided")
        if use_quantile_norm is None:
            use_quantile_norm = False
        if use_per_timestamp_action_norm is None:
            use_per_timestamp_action_norm = False
        if discrete_state_input is None:
            discrete_state_input = False
        if use_task_embedding is None:
            use_task_embedding = False
        if use_language_with_task_embedding is None:
            use_language_with_task_embedding = False
        if num_tasks is None:
            num_tasks = 0
        if use_embodiment_embedding is None:
            use_embodiment_embedding = False
        if num_embodiments is None:
            num_embodiments = 2
        if use_task_embedding and (not discrete_state_input or num_tasks <= 0):
            raise ValueError("Task embedding requires discrete_state_input=True and num_tasks > 0")
        if use_language_with_task_embedding and not use_task_embedding:
            raise ValueError("use_language_with_task_embedding requires use_task_embedding=True")
        if use_embodiment_embedding:
            if dataset_format != "video_lance":
                raise ValueError("Embodiment embedding currently requires dataset_format='video_lance'")
            if num_embodiments <= 0:
                raise ValueError("num_embodiments must be positive")
        if apply_delta_transform is None:
            apply_delta_transform = False
        if state_history_frames is None:
            state_history_frames = 1
        if state_delay_frames is None:
            state_delay_frames = 0
        if test_ep_num is None:
            test_ep_num = 0
        if policy_state_schema is None:
            policy_state_schema = "source"

        # Validate policy_name
        if policy_name is None:
            raise ValueError("policy_name must be provided either via config or as explicit argument")

        # Store compact state; avoid keeping train_config/data_config around.
        self.repo_id = repo_id
        self.dataset_format = str(dataset_format)
        if self.dataset_format not in {"lerobot", "video_lance"}:
            raise ValueError(f"Unsupported dataset_format: {self.dataset_format}")
        dataset_uris = _dataset_uri_values(dataset_uri, self.repo_id)
        if len(dataset_uris) != 1:
            raise ValueError(
                f"One loader reads one dataset, got {dataset_uris}. Pass one DatasetConfig per dataset; "
                "MultiLeRobotLoader iterates over all of them."
            )
        self.dataset_uri = dataset_uris[0]
        self.policy_name = policy_name
        self.policy_state_schema = resolve_policy_state_schema(policy_name, policy_state_schema)
        self.action_sequence_keys = list(action_sequence_keys)
        self.state_sequence_keys = list(state_sequence_keys)
        self.norm_stats = norm_stats
        self.use_quantile_norm = use_quantile_norm
        self.use_per_timestamp_action_norm = bool(use_per_timestamp_action_norm)
        self.batch_size = int(batch_size)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.max_token_len = int(max_token_len)
        self.discrete_state_input = bool(discrete_state_input)
        self.use_task_embedding = bool(use_task_embedding)
        self.use_language_with_task_embedding = bool(use_language_with_task_embedding)
        self.num_tasks = int(num_tasks)
        self.use_embodiment_embedding = bool(use_embodiment_embedding)
        self.num_embodiments = int(num_embodiments)
        self.embodiment_index = embodiment_index
        self.embodiment_source: str | None = None
        self.local_to_official_task: dict[int, int] | None = None
        self.apply_delta_transform = bool(apply_delta_transform)
        self.state_history_frames = int(state_history_frames)
        self.state_delay_frames = int(state_delay_frames)
        self.test_ep_num = int(test_ep_num)
        self.mode = str(mode)

        if self.norm_stats is None or "state" not in self.norm_stats:
            raise ValueError("Normalization stats are missing 'state'.")
        norm_state_dim = len(self.norm_stats["state"].mean)
        expected_state_dim = policy_state_dim(self.policy_name, self.policy_state_schema, norm_state_dim)
        if norm_state_dim != expected_state_dim:
            raise ValueError(
                f"{self.policy_name} {self.policy_state_schema} requires {expected_state_dim}D state stats, "
                f"got {norm_state_dim}D"
            )
        if self.mode not in ["train", "test"]:
            raise ValueError(f"mode must be 'train' or 'test', got '{self.mode}'")

        metadata = self._create_metadata()
        # Use dataset fps to build per-key delta timestamps of length action_horizon.
        delta_timestamps = {
            key: [t / metadata.fps for t in range(self.action_horizon)] for key in self.action_sequence_keys
        }

        # Add state history frames: state at [current-delay, current-delay-1, ..., current-delay-history+1]
        # For history frames, we need negative time offsets going backwards in time
        # Example: state_delay_frames=2, state_history_frames=3
        # We want states at: [t-2, t-3, t-4] (current delayed by 2, then 2 more historical frames)
        if self.state_history_frames > 1 or self.state_delay_frames > 0:
            for state_key in self.state_sequence_keys:
                # Create time offsets for historical states (negative = past frames)
                state_offsets = []
                for i in range(self.state_history_frames):
                    # Negative offset: -state_delay_frames, -state_delay_frames-1, -state_delay_frames-2, ...
                    offset = -(self.state_delay_frames + i)
                    state_offsets.append(offset / metadata.fps)
                delta_timestamps[state_key] = state_offsets

        # Split episodes into train/test sets
        total_ep_num = metadata.total_episodes

        # Validate test_ep_num
        if self.test_ep_num < 0:
            raise ValueError(f"test_ep_num must be non-negative, got {self.test_ep_num}")
        if self.test_ep_num >= total_ep_num:
            raise ValueError(
                f"test_ep_num ({self.test_ep_num}) must be less than total episodes ({total_ep_num}). "
                f"At least 1 episode is required for training."
            )

        train_count = total_ep_num - self.test_ep_num

        # Use fixed random seed for reproducible splits
        rng = np.random.RandomState(42)
        total_ep_idx = rng.permutation(total_ep_num)
        train_ep_idx = total_ep_idx[:train_count]
        test_ep_idx = total_ep_idx[train_count:]

        logging.info(
            f"Dataset split for {self.repo_id}: {len(train_ep_idx)} train episodes, "
            f"{len(test_ep_idx)} test episodes (test_ep_num={self.test_ep_num})"
        )

        # Create dataset based on mode
        if self.mode == "train":
            episodes_to_use = train_ep_idx
        else:  # mode == "test"
            episodes_to_use = test_ep_idx

        self.dataset = self._create_dataset(episodes_to_use, delta_timestamps)
        logging.info(f"Created {self.dataset_format} dataset: {self.dataset_uri}")
        # Fix for episode indexing bug when using episodes parameter
        # The episode_data_index is indexed by filtered episode position, but
        # _get_query_indices receives original episode indices from the data
        if self.dataset_format == "lerobot" and episodes_to_use is not None:
            # Create mapping from original episode index to filtered position
            episode_index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(episodes_to_use)}
            # Use picklable wrapper class instead of closure (required for spawn mode)
            self.dataset._get_query_indices = _EpisodeIndexMapper(
                self.dataset._get_query_indices,
                episode_index_map,
            )

        self.tokenizer = tokenizer_mod.PaligemmaTokenizer(self.max_token_len)

    def _create_metadata(self):
        """Read dataset metadata, validating the contract of Lance-backed sources."""
        if self.dataset_format == "lerobot":
            return lerobot_dataset.LeRobotDatasetMetadata(self.repo_id)

        try:
            from video_lance import VideoLanceMetadata
        except ImportError as error:  # pragma: no cover - depends on external setup
            raise ImportError(
                "video_lance is not available. VideoLance datasets need the external "
                "video_lance package installed in the environment before dataset_format='video_lance' can be used."
            ) from error

        if "goai" in self.policy_name.lower() or self.use_embodiment_embedding:
            import lance

            dataset = lance.dataset(self.dataset_uri)
            metadata = dict(dataset.schema.metadata or {})
            if "goai" in self.policy_name.lower():
                _validate_goai_lance_image_metadata(metadata, self.dataset_uri)
            embodiment_keys = (EMBODIMENT_METADATA_KEY.encode(), LEGACY_EMBODIMENT_METADATA_KEY.encode())
            if "goai" in self.policy_name.lower() and any(key in metadata for key in embodiment_keys):
                try:
                    tasks = json.loads(metadata.get(b"lerobot:tasks_json", b"{}"))
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError(f"Invalid lerobot:tasks_json metadata in {self.dataset_uri}") from error
                matches = build_task_remap(tasks)
                self.local_to_official_task = {local: match.slot for local, match in matches.items()}
                logging.info("GOAI task alignment for %s:", self.dataset_uri)
                for local_index, match in matches.items():
                    logging.info(
                        "  local task %d -> official slot %d | %s | %s score=%.4f",
                        local_index,
                        match.slot,
                        match.instruction,
                        match.method,
                        match.score,
                    )
            if self.use_embodiment_embedding:
                self.embodiment_index, self.embodiment_source = resolve_dataset_embodiment(
                    metadata,
                    num_embodiments=self.num_embodiments,
                    fallback=self.embodiment_index,
                )
                log = logging.warning if self.embodiment_source.startswith("default:") else logging.info
                log(
                    "Resolved embodiment_index=%s for %s from %s",
                    self.embodiment_index,
                    self.dataset_uri,
                    self.embodiment_source,
                )
        return VideoLanceMetadata(self.dataset_uri)

    def _create_dataset(self, episodes_to_use, delta_timestamps: dict[str, list[float]]):
        if self.dataset_format == "lerobot":
            return lerobot_dataset.LeRobotDataset(
                self.repo_id,
                episodes=episodes_to_use,
                delta_timestamps=delta_timestamps,
            )

        try:
            from video_lance import VideoLanceDataset
        except ImportError as error:  # pragma: no cover - depends on external setup
            raise ImportError(
                "video_lance is not available. VideoLance datasets need the external "
                "video_lance package installed in the environment before dataset_format='video_lance' can be used."
            ) from error

        logging.info(f"Creating VideoLanceDataset from {self.dataset_uri}")
        return VideoLanceDataset(
            self.dataset_uri,
            episodes=episodes_to_use,
            delta_timestamps=delta_timestamps,
        )

    def remap_task_indices(self, values):
        """Map local dataset task indices to official GOAI real-task slots."""
        if self.local_to_official_task is None:
            return values
        array = np.asarray(values)
        flat = array.reshape(-1)
        remapped = np.empty(flat.shape, dtype=np.int64)
        for index, value in enumerate(flat):
            local_index = int(value)
            if local_index not in self.local_to_official_task:
                raise ValueError(
                    f"Dataset {self.dataset_uri} emitted task_index={local_index}, but its "
                    f"lerobot:tasks_json only defines {sorted(self.local_to_official_task)}"
                )
            remapped[index] = self.local_to_official_task[local_index]
        remapped = remapped.reshape(array.shape)
        return np.int64(remapped.item()) if remapped.ndim == 0 else remapped

    def _transform(self, sample: dict) -> dict:
        if self.local_to_official_task is not None:
            if "task_index" not in sample:
                raise ValueError(f"GOAI aligned dataset {self.dataset_uri} emitted a sample without task_index")
            sample = dict(sample)
            sample["task_index"] = self.remap_task_indices(sample["task_index"])
        # Read a dataset row and map it onto the intermediate layout shared by all platforms.
        step = _repack_transform(self.policy_name, sample, self.dataset_format)
        # Normalize the layout into model inputs; images become uint8 HWC here.
        step = _data_inputs(step)
        if self.use_embodiment_embedding:
            if self.embodiment_index is None:
                raise RuntimeError("embodiment_index was not resolved during dataset initialization")
            step["embodiment_index"] = np.int64(self.embodiment_index)

        # Apply delta action transform before normalization if enabled
        # This converts absolute actions to delta (relative to current state)
        if self.apply_delta_transform:
            step = _apply_delta_actions(step, _get_delta_action_mask(self.policy_name))
        # The delta transform needs the raw source state, so the policy-state
        # projection (identity for GOAI) happens only afterwards.
        step["state"] = project_policy_state(step["state"], self.policy_name, self.policy_state_schema)
        step = _normalize(
            step,
            self.norm_stats,
            self.use_quantile_norm,
            self.use_per_timestamp_action_norm,
        )
        step = _resize_images(step, 224, 224)
        step = _tokenize_prompt(
            step,
            self.tokenizer,
            discrete_state_input=self.discrete_state_input,
            use_task_embedding=self.use_task_embedding,
            use_language_with_task_embedding=self.use_language_with_task_embedding,
            num_tasks=self.num_tasks,
        )
        step = _pad_state_actions(step, self.action_dim)  # state and actions both pad to action_dim
        return step

    def _transform_sample(self, sample: dict) -> dict:
        # _transform handles both single-frame and multi-frame data automatically through numpy broadcasting
        # - _repack_transform: just remaps keys, doesn't care about shape
        # - _normalize: uses _normalize_array which supports broadcasting for (history, dim)
        # - _pad_state_actions: pads last dimension regardless of array.ndim
        transformed = self._transform(sample)

        # If no history frames configured, add time dimension for consistency
        if not (self.state_history_frames > 1 or self.state_delay_frames > 0):
            # transformed["state"] shape: (state_dim,) -> (1, state_dim)
            transformed["state"] = np.expand_dims(transformed["state"], axis=0)
        # else: transformed["state"] already has shape (history_frames, state_dim) from delta_timestamps

        return transformed

    # Random-access API to fetch a single transformed item
    def __getitem__(self, idx: int) -> dict:
        # Now delta_timestamps handles state history and delay automatically
        # The dataset will return historical states based on the delta_timestamps we configured
        sample = self.dataset[idx]
        return self._transform_sample(sample)

    def __getitems__(self, indices: list[int]) -> list[dict]:
        if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
            samples = self.dataset.__getitems__(indices)
        else:
            samples = [self.dataset[idx] for idx in indices]
        return [self._transform_sample(sample) for sample in samples]

    # Dataset length passthrough
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict]:
        batch_size = self.batch_size
        buffer: list[dict] = []

        # Now delta_timestamps handles all temporal queries automatically
        # We can use a simple iteration over the dataset
        for idx in range(len(self.dataset)):
            buffer.append(self[idx])  # Use __getitem__ which handles delta_timestamps properly
            if len(buffer) == batch_size:
                yield _stack_tree(buffer)
                buffer = []
        if buffer:
            yield _stack_tree(buffer)


class MultiLeRobotLoader:
    """Round-robin iterator that batches samples from multiple datasets.

    The global batching/model params are provided once, while dataset-specific
    parameters are carried by `datasets` (previously passed as non-config args).
    """

    def __init__(
        self,
        *,
        datasets: list[DatasetConfig],
        batch_size: int,
        action_horizon: int,
        action_dim: int,
        max_token_len: int,
        discrete_state_input: bool = False,
        use_task_embedding: bool = False,
        use_language_with_task_embedding: bool = False,
        num_tasks: int = 0,
        use_embodiment_embedding: bool = False,
        num_embodiments: int = 2,
        apply_delta_transform: bool = True,
        use_per_timestamp_action_norm: bool = False,
        state_history_frames: int = 1,
        state_delay_frames: int = 0,
        mode: str = "train",  # "train" or "test"
    ) -> None:
        if not datasets:
            raise ValueError("datasets must be a non-empty list")

        self.batch_size = int(batch_size)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.max_token_len = int(max_token_len)
        self.discrete_state_input = bool(discrete_state_input)
        self.use_task_embedding = bool(use_task_embedding)
        self.use_language_with_task_embedding = bool(use_language_with_task_embedding)
        self.num_tasks = int(num_tasks)
        self.use_embodiment_embedding = bool(use_embodiment_embedding)
        self.num_embodiments = int(num_embodiments)
        self.apply_delta_transform = bool(apply_delta_transform)
        self.use_per_timestamp_action_norm = bool(use_per_timestamp_action_norm)
        self.state_history_frames = int(state_history_frames)
        self.state_delay_frames = int(state_delay_frames)
        self.mode = str(mode)

        # Build multiple SimpleLeRobotLoader instances (source of truth).
        self._loaders: list[SimpleLeRobotLoader] = []
        for cfg in datasets:
            loader = SimpleLeRobotLoader(
                None,
                repo_id=cfg.repo_id,
                dataset_format=cfg.dataset_format,
                dataset_uri=cfg.dataset_uri,
                action_sequence_keys=list(cfg.action_sequence_keys),
                state_sequence_keys=list(cfg.state_sequence_keys),
                norm_stats=cfg.norm_stats,
                use_quantile_norm=cfg.use_quantile_norm,
                use_per_timestamp_action_norm=cfg.use_per_timestamp_action_norm,
                policy_name=cfg.policy_name,
                policy_state_schema=cfg.policy_state_schema,
                batch_size=self.batch_size,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                max_token_len=self.max_token_len,
                discrete_state_input=self.discrete_state_input,
                use_task_embedding=self.use_task_embedding,
                use_language_with_task_embedding=self.use_language_with_task_embedding,
                num_tasks=self.num_tasks,
                use_embodiment_embedding=self.use_embodiment_embedding,
                num_embodiments=self.num_embodiments,
                embodiment_index=cfg.embodiment_index,
                apply_delta_transform=self.apply_delta_transform,
                state_history_frames=self.state_history_frames,
                state_delay_frames=self.state_delay_frames,
                test_ep_num=cfg.test_ep_num,
                mode=self.mode,
            )
            self._loaders.append(loader)

        # Precompute index offsets for O(log N) __getitem__ lookup across loaders.
        self._offsets: list[int] = [0]
        total = 0
        for ld in self._loaders:
            total += len(ld)
            self._offsets.append(total)

        self.valid_ptr = None

    # Flattened length across all sub-loaders
    def __len__(self) -> int:  # type: ignore[override]
        return self._offsets[-1]

    def get_task_episode_ranges(self) -> dict[int, list[tuple[int, int]]]:
        """Return flattened local frame ranges grouped by task and episode."""
        from pi.training.samplers import task_episode_ranges_from_arrays

        combined: dict[int, list[tuple[int, int]]] = {}
        for loader_index, loader in enumerate(self._loaders):
            dataset = loader.dataset
            scalar_cache = getattr(dataset, "_scalar_cache", None)
            selected_indices = getattr(dataset, "_indices", None)
            if not isinstance(scalar_cache, dict) or selected_indices is None:
                raise ValueError("task_episode_balanced sampling currently requires a VideoLance dataset")
            if "task_index" not in scalar_cache or "episode_index" not in scalar_cache:
                raise ValueError("VideoLance dataset must contain task_index and episode_index columns")

            selected_indices = np.asarray(selected_indices, dtype=np.int64)
            groups = task_episode_ranges_from_arrays(
                loader.remap_task_indices(np.asarray(scalar_cache["task_index"])[selected_indices]),
                np.asarray(scalar_cache["episode_index"])[selected_indices],
                offset=self._offsets[loader_index],
            )
            for task_id, episode_ranges in groups.items():
                combined.setdefault(task_id, []).extend(episode_ranges)
        return combined

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")

        loader_idx = bisect.bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[loader_idx]
        return loader_idx, local_idx

    # Random-access across concatenated datasets
    def __getitem__(self, idx: int) -> dict:
        i = 0
        local_idx = 0
        try:
            i, local_idx = self._resolve_index(idx)
            data = self._loaders[i][local_idx]
            self.valid_ptr = (i, local_idx)  # remember last valid loader index
            return data
        except Exception as e:
            msg = f"Error fetching index {idx}, repo_id {self._loaders[i].repo_id}, local_idx {local_idx}: {type(e).__name__}: {e}"
            logging.error(msg)
            i, local_idx = self.valid_ptr if self.valid_ptr is not None else (0, 0)
            return self._loaders[i][local_idx]  # fall back to the last valid sample

    def __getitems__(self, indices: list[int]) -> list[dict]:
        grouped: dict[int, list[tuple[int, int]]] = {}
        for pos, idx in enumerate(indices):
            loader_idx, local_idx = self._resolve_index(idx)
            grouped.setdefault(loader_idx, []).append((pos, local_idx))

        results: list[dict | None] = [None] * len(indices)
        for loader_idx, positions in grouped.items():
            loader = self._loaders[loader_idx]
            local_indices = [local_idx for _, local_idx in positions]
            try:
                if hasattr(loader, "__getitems__") and loader.__getitems__:
                    batch = loader.__getitems__(local_indices)
                else:
                    batch = [loader[local_idx] for local_idx in local_indices]
            except Exception as e:
                logging.error(
                    "Error fetching batch indices %s from repo_id %s: %s: %s",
                    local_indices,
                    loader.repo_id,
                    type(e).__name__,
                    e,
                )
                batch = [loader[local_idx] for local_idx in local_indices]

            for (pos, local_idx), item in zip(positions, batch, strict=True):
                results[pos] = item
                self.valid_ptr = (loader_idx, local_idx)

        if any(item is None for item in results):
            raise RuntimeError("Batch fetch returned incomplete results.")
        return [item for item in results if item is not None]

    def __iter__(self) -> Iterator[dict]:
        # Now delta_timestamps handles all temporal queries automatically
        # Index-based round-robin iteration
        indices = [0 for _ in self._loaders]  # next-sample index per sub-loader
        active_idx = list(range(len(self._loaders)))
        buffer: list[dict] = []
        rr = 0

        while active_idx:
            i = active_idx[rr % len(active_idx)]
            loader = self._loaders[i]

            # Check if this loader is exhausted
            if indices[i] >= len(loader):
                # Drop loader i from the active list once all of its samples are consumed.
                active_idx.pop(rr % len(active_idx))
                continue

            # Use loader's __getitem__ which handles delta_timestamps properly
            buffer.append(loader[indices[i]])
            indices[i] += 1
            rr += 1

            if len(buffer) == self.batch_size:
                yield _stack_tree(buffer)
                buffer = []

        if buffer:
            yield _stack_tree(buffer)
