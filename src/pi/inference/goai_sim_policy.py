"""Pi05 policy adapter for the GOAI RoboDojo joint-control simulator."""

from __future__ import annotations

import gc
import logging
import dataclasses
from typing import Any, Literal, Mapping, TypedDict, cast
from pathlib import Path

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

from pi.models import tokenizer as tokenizer_mod
from pi.shared import normalize as normalize_mod
from pi.training import instance_config
from pi.models.model import Observation
from pi.models.pi_config import PiConfig
from pi.shared.embodiment import embodiment_index as resolve_embodiment_index, build_embodiment_contract
from pi.shared.goai_tasks import GOAI_REAL_TASK_INSTRUCTIONS, match_task
from pi.inference.goai_helpers import (
    normalize_values,
    unnormalize_actions,
    resolve_action_norm_mode,
    _load_checkpoint_manifest_entry,
)
from pi.inference.goai_observation import prepare_goai_observation
from pi.models_pytorch.pi0_pytorch import PI0Pytorch

LOGGER = logging.getLogger(__name__)

GOAI_ACTION_DIM = 14
GOAI_STATE_DIM = 14
GOAI_MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
GOAI_TASK_NAMES = (
    "arrange_largest_number",
    "fold_clothes",
    "hang_mugs",
    "make_toast",
    "pack_objects_into_box",
    "pour_liquid_into_cup",
    "push_T",
    "sort_nesting_dolls_by_size",
    "stack_blocks",
    "stack_bowls",
    "store_laptop_and_headphones",
    "sweep_blocks",
)
GOAI_DELTA_ACTION_MASK = np.asarray(
    (True, True, True, True, True, True, False, True, True, True, True, True, True, False),
    dtype=bool,
)


@dataclasses.dataclass
class GOAIPolicySession:
    """Per-RoboDojo-client random state for a shared policy model."""

    generator: torch.Generator
    seed: int
    task_name: str | None = None
    task_index: int | None = None
    step: int = 0
    logged_conditioning: bool = False


def build_goai_task_table() -> dict[str, int]:
    """Return the dataset task-index mapping, including RoboDojo random aliases."""
    table: dict[str, int] = {}
    for task_index, task_name in enumerate(GOAI_TASK_NAMES):
        table[task_name.lower()] = task_index
        table[f"{task_name.lower()}_random"] = task_index
    return table


GOAI_TASK_TABLE = build_goai_task_table()


def resolve_goai_task_index(task_name: str) -> int:
    """Resolve legacy simulator slugs or official real-task language."""
    normalized = task_name.strip().replace("-", "_").lower()
    if normalized in GOAI_TASK_TABLE:
        return GOAI_TASK_TABLE[normalized]
    try:
        task_index, _ = match_task(task_name)
        return task_index
    except ValueError as error:
        sim_supported = ", ".join(GOAI_TASK_NAMES)
        real_supported = "; ".join(GOAI_REAL_TASK_INSTRUCTIONS)
        raise ValueError(
            f"Unsupported GOAI task {task_name!r}; simulator slugs: {sim_supported}; "
            f"real instructions: {real_supported}.\n{error}"
        ) from error


class GOAITrainingContract(TypedDict):
    """Validated manifest fields used to reconstruct the inference model."""

    image_geometry: dict[str, object]
    use_task_embedding: bool
    use_language_with_task_embedding: bool
    use_embodiment_embedding: bool
    num_embodiments: int
    num_embodiment_tokens: int
    embodiment_contract: object
    task_embedding_target: Literal["vlm", "expert"]
    state_conditioning_mode: Literal["discrete_vlm", "dual"]
    use_quantile_norm: bool
    use_per_timestamp_action_norm: bool
    apply_delta_transform: bool | None
    num_tasks: int
    action_horizon: int
    max_token_len: int


def load_goai_checkpoint_training_contract(checkpoint: Path, config_name: str) -> GOAITrainingContract:
    """Load and validate the GOAI train/inference contract stored beside a checkpoint."""
    entry = _load_checkpoint_manifest_entry(checkpoint, config_name)
    geometry = entry.get("image_geometry")
    expected_geometry = {
        "width": 640,
        "height": 640,
        "resize_mode": "pad",
        "source_width": 640,
        "source_height": 480,
    }
    if geometry != expected_geometry:
        raise ValueError(f"Checkpoint image geometry mismatch: expected {expected_geometry}, got {geometry}")

    manifest_version = entry.get("_manifest_version")
    task_embedding_target = entry.get("task_embedding_target")
    state_conditioning_mode = entry.get("state_conditioning_mode")
    legacy_architecture = not isinstance(manifest_version, int) or manifest_version < 11
    if task_embedding_target is None and legacy_architecture:
        task_embedding_target = "vlm"
    if state_conditioning_mode is None and legacy_architecture:
        state_conditioning_mode = "discrete_vlm"
    if task_embedding_target not in ("vlm", "expert"):
        raise ValueError(f"Checkpoint {checkpoint} has invalid task_embedding_target={task_embedding_target!r}")
    if state_conditioning_mode not in ("discrete_vlm", "dual"):
        raise ValueError(f"Checkpoint {checkpoint} has invalid state_conditioning_mode={state_conditioning_mode!r}")

    use_embodiment_embedding = entry.get("use_embodiment_embedding")
    if not isinstance(manifest_version, int) or manifest_version < 12:
        use_embodiment_embedding = False
    if type(use_embodiment_embedding) is not bool:
        raise ValueError(f"Checkpoint {checkpoint} is missing a boolean use_embodiment_embedding marker")

    if use_embodiment_embedding:
        num_embodiments = entry.get("num_embodiments")
        num_embodiment_tokens = entry.get("num_embodiment_tokens")
        if not isinstance(num_embodiments, int) or num_embodiments <= 0:
            raise ValueError(f"Checkpoint {checkpoint} has invalid num_embodiments={num_embodiments!r}")
        if not isinstance(num_embodiment_tokens, int) or num_embodiment_tokens <= 0:
            raise ValueError(f"Checkpoint {checkpoint} has invalid num_embodiment_tokens={num_embodiment_tokens!r}")
        expected_embodiment_contract = build_embodiment_contract(
            num_embodiments=num_embodiments,
            tokens_per_embodiment=num_embodiment_tokens,
        )
        if entry.get("embodiment_contract") != expected_embodiment_contract:
            raise ValueError(
                "Checkpoint embodiment contract mismatch: "
                f"expected {expected_embodiment_contract}, got {entry.get('embodiment_contract')}"
            )
    else:
        num_embodiments = 2
        num_embodiment_tokens = 1

    required_bools = (
        "use_task_embedding",
        "use_language_with_task_embedding",
        "use_quantile_norm",
        "use_per_timestamp_action_norm",
    )
    for key in required_bools:
        if type(entry.get(key)) is not bool:
            raise ValueError(f"Checkpoint {checkpoint} is missing a boolean {key} marker")

    apply_delta_transform = entry.get("apply_delta_transform")
    if isinstance(manifest_version, int) and manifest_version >= 12:
        if type(apply_delta_transform) is not bool:
            raise ValueError(f"Checkpoint {checkpoint} is missing a boolean apply_delta_transform marker")
    elif type(apply_delta_transform) is not bool:
        apply_delta_transform = None

    num_tasks = entry.get("num_tasks")
    action_horizon = entry.get("action_horizon")
    max_token_len = entry.get("max_token_len")
    if not isinstance(num_tasks, int) or num_tasks <= 0:
        raise ValueError(f"Checkpoint {checkpoint} has invalid num_tasks={num_tasks!r}")
    if not isinstance(action_horizon, int) or action_horizon <= 0:
        raise ValueError(f"Checkpoint {checkpoint} has invalid action_horizon={action_horizon!r}")
    if not isinstance(max_token_len, int) or max_token_len <= 0:
        raise ValueError(f"Checkpoint {checkpoint} has invalid max_token_len={max_token_len!r}")
    if entry["use_language_with_task_embedding"] and not entry["use_task_embedding"]:
        raise ValueError("Language-plus-task-embedding checkpoints must enable task embedding")
    if task_embedding_target == "expert" and not entry["use_task_embedding"]:
        raise ValueError("Expert task embedding target requires task embedding")

    return {
        "image_geometry": cast(dict[str, object], geometry),
        "use_task_embedding": entry["use_task_embedding"],
        "use_language_with_task_embedding": entry["use_language_with_task_embedding"],
        "use_embodiment_embedding": use_embodiment_embedding,
        "num_embodiments": num_embodiments,
        "num_embodiment_tokens": num_embodiment_tokens,
        "embodiment_contract": entry.get("embodiment_contract"),
        "task_embedding_target": task_embedding_target,
        "state_conditioning_mode": state_conditioning_mode,
        "use_quantile_norm": entry["use_quantile_norm"],
        "use_per_timestamp_action_norm": entry["use_per_timestamp_action_norm"],
        "apply_delta_transform": apply_delta_transform,
        "num_tasks": num_tasks,
        "action_horizon": action_horizon,
        "max_token_len": max_token_len,
    }


def _vector(mapping: Mapping[str, Any], key: str, dimension: int) -> np.ndarray:
    if key not in mapping:
        raise KeyError(f"GOAI observation state is missing {key!r}")
    value = np.asarray(mapping[key], dtype=np.float32)
    if value.shape != (dimension,):
        raise ValueError(f"GOAI state {key!r} must have shape ({dimension},), got {value.shape}")
    return value


def assemble_goai_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Pack the raw RoboDojo bimanual state into [left6, grip, right6, grip]."""
    raw_state = observation.get("state")
    if not isinstance(raw_state, Mapping):
        state = np.asarray(raw_state, dtype=np.float32)
        if state.shape != (GOAI_STATE_DIM,):
            raise ValueError(f"GOAI state must have shape ({GOAI_STATE_DIM},), got {state.shape}")
    else:
        state = np.concatenate(
            (
                _vector(raw_state, "left_arm_joint_state", 6),
                _vector(raw_state, "left_ee_joint_state", 1),
                _vector(raw_state, "right_arm_joint_state", 6),
                _vector(raw_state, "right_ee_joint_state", 1),
            )
        ).astype(np.float32, copy=False)
    if not np.isfinite(state).all():
        raise ValueError("GOAI state contains non-finite values")
    return state


def _camera_color(vision: Mapping[str, Any], key: str, aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias not in vision:
            continue
        camera = vision[alias]
        if isinstance(camera, Mapping):
            for color_key in ("color", "rgb"):
                if color_key in camera:
                    return camera[color_key]
            raise KeyError(f"GOAI camera {alias!r} has no color/rgb image")
        return camera
    raise KeyError(f"GOAI observation is missing camera {key!r} (aliases={aliases})")


def adapt_robodojo_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one raw RoboDojo observation to the existing GOAI image adapter input."""
    state = assemble_goai_state(observation)
    images = observation.get("images")
    if not isinstance(images, Mapping):
        vision = observation.get("vision")
        if not isinstance(vision, Mapping):
            raise KeyError("GOAI observation requires vision or images")
        images = {
            "cam_high": _camera_color(vision, "cam_high", ("cam_high", "cam_head")),
            "cam_left_wrist": _camera_color(vision, "cam_left_wrist", ("cam_left_wrist",)),
            "cam_right_wrist": _camera_color(vision, "cam_right_wrist", ("cam_right_wrist",)),
        }
    return prepare_goai_observation(
        {
            "state": state,
            "images": dict(images),
            "images_preprocessed": bool(observation.get("images_preprocessed", False)),
            "instruction": observation.get("instruction", observation.get("prompt")),
        },
        "joint",
    )


def apply_absolute_actions_goai(
    actions: np.ndarray,
    current_state: np.ndarray,
    *,
    apply_delta: bool,
) -> np.ndarray:
    """Invert GOAI joint deltas while preserving absolute gripper dimensions."""
    actions = np.asarray(actions, dtype=np.float64).copy()
    current_state = np.asarray(current_state, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != GOAI_ACTION_DIM:
        raise ValueError(f"GOAI actions must have shape (horizon, {GOAI_ACTION_DIM}), got {actions.shape}")
    if current_state.shape != (GOAI_STATE_DIM,):
        raise ValueError(f"GOAI current state must have shape ({GOAI_STATE_DIM},), got {current_state.shape}")
    if apply_delta:
        actions[:, GOAI_DELTA_ACTION_MASK] += current_state[GOAI_DELTA_ACTION_MASK]
    actions[:, (6, 13)] = np.clip(actions[:, (6, 13)], 0.0, 1.0)
    if not np.isfinite(actions).all():
        raise RuntimeError("Policy produced non-finite GOAI actions")
    return actions.astype(np.float32)


def action_chunk_to_robodojo(actions: np.ndarray) -> list[dict[str, np.ndarray]]:
    """Split a 14D action chunk into RoboDojo's per-arm action dictionaries."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != GOAI_ACTION_DIM:
        raise ValueError(f"GOAI actions must have shape (horizon, {GOAI_ACTION_DIM}), got {actions.shape}")
    return [
        {
            "left_arm_joint_state": action[:6].copy(),
            "left_ee_joint_state": action[6:7].copy(),
            "right_arm_joint_state": action[7:13].copy(),
            "right_ee_joint_state": action[13:14].copy(),
        }
        for action in actions
    ]


def validate_goai_dcp_coverage(model: torch.nn.Module, metadata: Mapping[str, Any]) -> None:
    """Reject missing independent model tensors while allowing saved tied aliases."""
    expected = model.state_dict(keep_vars=True)
    # EMA uses named_parameters(), which omits duplicate tied aliases.
    covered_ids = {id(value) for key, value in expected.items() if key in metadata}
    missing = sorted(key for key, value in expected.items() if key not in metadata and id(value) not in covered_ids)
    mismatched = sorted(
        key
        for key, value in expected.items()
        if key in metadata and tuple(getattr(metadata[key], "size", ())) != tuple(value.shape)
    )
    if missing or mismatched:
        raise ValueError(f"Incomplete DCP model: missing={missing}, shape_mismatch={mismatched}")


class GOAISimPolicy:
    """Load a Pi DCP checkpoint and produce RoboDojo action-dict chunks."""

    def __init__(
        self,
        checkpoint: str | Path,
        norm_stats: str | Path,
        *,
        task_name: str | None = None,
        config_name: str = "pi05_goai_joint",
        device: str = "cuda:0",
        action_horizon: int = 32,
        num_steps: int = 10,
        compile_mode: str = "none",
        norm_mode: str | None = None,
        embodiment_index: int = 0,
        apply_delta: bool,
        seed: int = 0,
        strict_checkpoint: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.norm_stats_path = Path(norm_stats).expanduser().resolve()
        self.task_name = task_name
        self.task_index = resolve_goai_task_index(task_name) if task_name is not None else None
        self.config_name = config_name
        self.device = torch.device(device)
        self.apply_delta = apply_delta
        self.seed = seed

        if not (self.checkpoint / ".metadata").is_file():
            raise FileNotFoundError(f"DCP metadata not found: {self.checkpoint / '.metadata'}")
        if not self.norm_stats_path.is_file():
            raise FileNotFoundError(f"Norm stats not found: {self.norm_stats_path}")

        train_config = instance_config.get_config(config_name)
        base_model_config = cast(PiConfig, train_config.model)
        if not base_model_config.pi05 or not base_model_config.discrete_state_input:
            raise ValueError(f"{config_name} is not a discrete-state Pi05 config")
        self.training_contract = load_goai_checkpoint_training_contract(self.checkpoint, config_name)
        checkpoint_apply_delta = self.training_contract["apply_delta_transform"]
        if checkpoint_apply_delta is not None and apply_delta is not checkpoint_apply_delta:
            raise ValueError(
                "Checkpoint action transform mismatch: "
                f"manifest apply_delta_transform={checkpoint_apply_delta}, requested apply_delta={apply_delta}"
            )
        self.model_config = dataclasses.replace(
            base_model_config,
            action_horizon=int(self.training_contract["action_horizon"]),
            max_token_len=int(self.training_contract["max_token_len"]),
            use_task_embedding=bool(self.training_contract["use_task_embedding"]),
            use_language_with_task_embedding=bool(self.training_contract["use_language_with_task_embedding"]),
            use_embodiment_embedding=bool(self.training_contract["use_embodiment_embedding"]),
            num_embodiments=int(self.training_contract["num_embodiments"]),
            num_embodiment_tokens=int(self.training_contract["num_embodiment_tokens"]),
            task_embedding_target=self.training_contract["task_embedding_target"],
            state_conditioning_mode=self.training_contract["state_conditioning_mode"],
            num_tasks=int(self.training_contract["num_tasks"]),
        )
        if self.model_config.use_embodiment_embedding:
            self.embodiment_index = resolve_embodiment_index(
                embodiment_index,
                num_embodiments=self.model_config.num_embodiments,
            )
        else:
            if embodiment_index != 0:
                raise ValueError("embodiment_index is only valid for embodiment-conditioned checkpoints")
            self.embodiment_index = 0
        if self.model_config.num_tasks != len(GOAI_TASK_NAMES):
            raise ValueError(
                f"GOAI task table has {len(GOAI_TASK_NAMES)} entries, checkpoint expects {self.model_config.num_tasks}"
            )
        if not 1 <= action_horizon <= self.model_config.action_horizon:
            raise ValueError(f"action_horizon must be in [1, {self.model_config.action_horizon}], got {action_horizon}")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        self.execution_horizon = action_horizon
        self.num_steps = num_steps
        self.compile_mode = compile_mode

        self.use_quantile_norm = bool(self.training_contract["use_quantile_norm"])
        self.norm_stats = normalize_mod.deserialize_json(self.norm_stats_path.read_text())
        for key in ("state", "actions"):
            if key not in self.norm_stats:
                raise ValueError(f"Norm stats are missing {key!r}")
        if len(self.norm_stats["state"].mean) != GOAI_STATE_DIM:
            raise ValueError("Norm stats do not describe a 14D GOAI state")
        if len(self.norm_stats["actions"].mean) != GOAI_ACTION_DIM:
            raise ValueError("Norm stats do not describe a 14D GOAI action")
        checkpoint_norm_mode = "per-timestamp" if self.training_contract["use_per_timestamp_action_norm"] else "global"
        if norm_mode is not None and norm_mode != checkpoint_norm_mode:
            raise ValueError(
                f"Requested norm_mode={norm_mode!r}, but checkpoint was trained with {checkpoint_norm_mode!r}"
            )
        self.action_norm_mode = resolve_action_norm_mode(
            self.norm_stats["actions"],
            checkpoint_norm_mode,
            use_quantile_norm=self.use_quantile_norm,
        )
        unnormalize_actions(
            np.zeros((self.model_config.action_horizon, GOAI_ACTION_DIM)),
            self.norm_stats["actions"],
            self.action_norm_mode,
            use_quantile_norm=self.use_quantile_norm,
        )

        LOGGER.info("Loading %s from %s on %s", config_name, self.checkpoint, self.device)
        with torch.device(self.device):
            self.model = PI0Pytorch(self.model_config)
        if strict_checkpoint:
            metadata = FileSystemReader(str(self.checkpoint)).read_metadata().state_dict_metadata
            validate_goai_dcp_coverage(self.model, metadata)
        planner = DefaultLoadPlanner(allow_partial_load=True) if self.checkpoint.name == "ema" else None
        dcp.load(self.model.state_dict(), checkpoint_id=str(self.checkpoint), planner=planner)
        self.model.eval()
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        if compile_mode == "none":
            self._sample_actions = self.model.sample_actions
        else:
            self._sample_actions = torch.compile(self.model.sample_actions, mode=compile_mode)
        self.tokenizer = tokenizer_mod.PaligemmaTokenizer(self.model_config.max_token_len)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "pi",
            "checkpoint": str(self.checkpoint),
            "config": self.config_name,
            "task_name": self.task_name,
            "task_index": self.task_index,
            "use_embodiment_embedding": self.model_config.use_embodiment_embedding,
            "embodiment_index": self.embodiment_index if self.model_config.use_embodiment_embedding else None,
            "model_action_horizon": self.model_config.action_horizon,
            "execution_horizon": self.execution_horizon,
            "num_steps": self.num_steps,
            "action_norm_mode": self.action_norm_mode,
            "normalization": "quantile" if self.use_quantile_norm else "z-score",
            "apply_delta": self.apply_delta,
        }

    def create_session(self, *, seed: int | None = None) -> GOAIPolicySession:
        session_seed = self.seed if seed is None else seed
        generator = torch.Generator(device=self.device)
        generator.manual_seed(session_seed)
        return GOAIPolicySession(
            generator=generator,
            seed=session_seed,
            task_name=self.task_name,
            task_index=self.task_index,
        )

    def bind_session(self, session: GOAIPolicySession, *, task_name: str, seed: int | None = None) -> None:
        """Bind one shared-server client to its task and deterministic seed."""
        task_index = resolve_goai_task_index(task_name)
        binding_changed = session.task_index != task_index or session.task_name != task_name
        if session.step and session.task_index != task_index:
            raise ValueError(f"Cannot change an active GOAI session from {session.task_name!r} to {task_name!r}")
        session.task_name = task_name
        session.task_index = task_index
        if seed is not None and seed != session.seed:
            if session.step:
                raise ValueError(f"Cannot reseed an active GOAI session from {session.seed} to {seed}")
            session.seed = seed
            session.generator.manual_seed(seed)
            binding_changed = True
        if binding_changed:
            session.logged_conditioning = False

    def reset(self, session: GOAIPolicySession) -> None:
        session.step = 0
        session.logged_conditioning = False
        session.generator.manual_seed(session.seed)

    def _prepare_observation(
        self,
        observation: Mapping[str, Any],
        raw_state: np.ndarray,
        session: GOAIPolicySession,
    ) -> Observation[torch.Tensor]:
        adapted = adapt_robodojo_observation(observation)
        normalized_state = normalize_values(
            raw_state,
            self.norm_stats["state"],
            use_quantile_norm=self.use_quantile_norm,
        ).astype(np.float32)

        task_index = None
        if self.model_config.use_task_embedding:
            if session.task_index is None or session.task_name is None:
                raise ValueError("GOAI task embedding requires a task bound to the client session")
            task_index = torch.tensor([session.task_index], dtype=torch.long, device=self.device)
            if self.model_config.use_language_with_task_embedding:
                prompt = adapted["prompt"]
                if not prompt:
                    raise ValueError("Language-plus-task-embedding checkpoint requires an instruction")
                tokens, token_mask = self.tokenizer.tokenize(prompt, normalized_state)
            else:
                tokens, token_mask = self.tokenizer.tokenize_state(normalized_state)
        else:
            prompt = adapted["prompt"]
            if not prompt:
                raise ValueError("Language-conditioned checkpoint requires an instruction")
            tokens, token_mask = self.tokenizer.tokenize(prompt, normalized_state)

        if not session.logged_conditioning:
            LOGGER.info("Using GOAI task %s (task_index=%d)", session.task_name, session.task_index)
            session.logged_conditioning = True

        images = {
            key: torch.as_tensor(adapted["image"][key], dtype=torch.uint8, device=self.device).unsqueeze(0)
            for key in GOAI_MODEL_IMAGE_KEYS
        }
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "image": images,
            "image_mask": {key: torch.ones(1, dtype=torch.bool, device=self.device) for key in GOAI_MODEL_IMAGE_KEYS},
            "state": torch.as_tensor(normalized_state, dtype=torch.float32, device=self.device).reshape(1, 1, -1),
            "tokenized_prompt": torch.as_tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0),
            "tokenized_prompt_mask": torch.as_tensor(token_mask, dtype=torch.bool, device=self.device).unsqueeze(0),
        }
        if task_index is not None:
            data["task_index"] = task_index
        if self.model_config.use_embodiment_embedding:
            data["embodiment_index"] = torch.tensor([self.embodiment_index], dtype=torch.long, device=self.device)
        return Observation.from_dict(data)

    def infer(self, observation: Mapping[str, Any], session: GOAIPolicySession) -> list[dict[str, np.ndarray]]:
        """Infer one executable action chunk from a raw RoboDojo observation."""
        raw_state = assemble_goai_state(observation)
        model_observation = self._prepare_observation(observation, raw_state, session)
        noise = torch.randn(
            (1, self.model_config.action_horizon, self.model_config.action_dim),
            dtype=torch.float32,
            device=self.device,
            generator=session.generator,
        )
        with torch.inference_mode():
            if self.compile_mode in ("reduce-overhead", "max-autotune"):
                # A long-lived shared server has no outer model loop from which
                # Torch can infer CUDA Graph iteration boundaries.
                torch.compiler.cudagraph_mark_step_begin()
            normalized = self._sample_actions(
                device=self.device,
                observation=model_observation,
                noise=noise,
                num_steps=self.num_steps,
            )
        normalized = normalized[0, :, :GOAI_ACTION_DIM].float().cpu().numpy()
        actions = unnormalize_actions(
            normalized,
            self.norm_stats["actions"],
            self.action_norm_mode,
            use_quantile_norm=self.use_quantile_norm,
        )
        actions = apply_absolute_actions_goai(actions, raw_state, apply_delta=self.apply_delta)
        session.step += 1
        return action_chunk_to_robodojo(actions[: self.execution_horizon])
