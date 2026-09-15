"""Official XPolicyLab policy interface for the six trained GOAI real tasks."""

from __future__ import annotations

import json
import time
import difflib
import hashlib
import logging
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from pi.shared.goai_tasks import (
    GOAI_REAL_TASK_INSTRUCTIONS,
    GOAI_REAL_OFFICIAL_TASK_INSTRUCTIONS,
    normalize_task_instruction,
)

logger = logging.getLogger(__name__)

_STATUS_INTERVAL_S = 60.0

# Reject unrelated phrases even when a nearest candidate exists.
_NEAREST_MIN_RATIO = 0.90

_TASKS = {normalize_task_instruction(name): (i, name) for i, name in enumerate(GOAI_REAL_TASK_INSTRUCTIONS)}
_SLUGS = (
    "fill_pen_holder",
    "put_objects_into_basket",
    "stack_and_cover_blocks",
    "stack_bowls",
    "stand_up_bottles",
    "insert_charger",
)
_TASKS.update({normalize_task_instruction(slug): (i, GOAI_REAL_TASK_INSTRUCTIONS[i]) for i, slug in enumerate(_SLUGS)})
_TASKS.update(
    {
        normalize_task_instruction(text): (i, GOAI_REAL_TASK_INSTRUCTIONS[i])
        for i, text in enumerate(GOAI_REAL_OFFICIAL_TASK_INSTRUCTIONS)
    }
)
_CAMERAS = {
    "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_wrist", "left_camera", "wrist_left"),
    "cam_right_wrist": ("cam_right_wrist", "right_wrist", "right_camera", "wrist_right"),
}
_STATE_KEYS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)
_GRIPPER_KEYS = ("left_ee_joint_state", "right_ee_joint_state")

# Optional gripper subtraction in normalized opening units; disabled by default.
DEFAULT_GRIPPER_SQUEEZE = 0.05
DEFAULT_GRIPPER_SQUEEZE_BELOW = 0.90
_SQUEEZE_HARD_RANGE = (0.0, 0.5)
_PROBE_THRESHOLDS = (0.25, 0.50, 0.75, 0.90, 0.95)


class PostProcess:
    """Optional server-side corrections to model actions."""

    def __init__(self, model_cfg):
        raw = model_cfg.get("postprocess")
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("postprocess 段必须是 map")
        unknown = sorted(set(raw) - {"gripper", "enabled"})
        if unknown:
            raise ValueError(f"postprocess 段有未知字段 {unknown};可选: ['enabled', 'gripper']")
        self.enabled = bool(raw.get("enabled", False))
        self.squeeze, self.squeeze_below, self.per_task = self._load_gripper(raw.get("gripper"))
        self.last_squeezed = 0
        self.last_total = 0
        self.reset_raw_stats()

    def reset_raw_stats(self):
        self._raw_n = 0
        self._raw_min = None
        self._raw_max = None
        self._raw_sum = 0.0
        self._raw_over = {t: 0 for t in _PROBE_THRESHOLDS}

    @staticmethod
    def _check_squeeze(value, where):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{where} 必须是数字: {value!r}") from None
        lo, hi = _SQUEEZE_HARD_RANGE
        if not lo <= value <= hi:
            raise ValueError(f"{where} = {value} 越界,必须在 [{lo}, {hi}] 之间;units are normalized opening ratios")
        return value

    @classmethod
    def _load_gripper(cls, raw):
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("postprocess.gripper 必须是 map")
        unknown = sorted(set(raw) - {"squeeze", "squeeze_below", "per_task"})
        if unknown:
            raise ValueError(f"postprocess.gripper 有未知字段 {unknown};可选: ['squeeze', 'squeeze_below', 'per_task']")
        squeeze = cls._check_squeeze(raw.get("squeeze", DEFAULT_GRIPPER_SQUEEZE), "postprocess.gripper.squeeze")
        try:
            below = float(raw.get("squeeze_below", DEFAULT_GRIPPER_SQUEEZE_BELOW))
        except (TypeError, ValueError):
            raise ValueError("postprocess.gripper.squeeze_below 必须是数字") from None
        if not 0.5 <= below <= 1.0:
            raise ValueError(
                "postprocess.gripper.squeeze_below 需要在 [0.5, 1.0];"
                "低于 0.5 会把大量过渡帧也下压,高于 1.0 等于一直下压"
            )
        per_task = {}
        for name, value in (raw.get("per_task") or {}).items():
            key = normalize_task_instruction(name)
            if key not in _TASKS:
                raise ValueError(f"postprocess.gripper.per_task 里的 {name!r} 不是已知任务;可选: {sorted(_SLUGS)}")
            per_task[_TASKS[key][0]] = cls._check_squeeze(value, f"postprocess.gripper.per_task.{name}")
        return squeeze, below, per_task

    def squeeze_for(self, task_index):
        return self.per_task.get(task_index, self.squeeze)

    def gripper(self, actions, amount):
        """Subtract the requested opening ratio below the configured threshold."""
        raw = [float(np.asarray(step[key]).reshape(-1)[0]) for step in actions for key in _GRIPPER_KEYS]
        for value in raw:
            self._raw_n += 1
            self._raw_min = value if self._raw_min is None else min(self._raw_min, value)
            self._raw_max = value if self._raw_max is None else max(self._raw_max, value)
            self._raw_sum += value
            for probe in _PROBE_THRESHOLDS:
                if value >= probe:
                    self._raw_over[probe] += 1
        if not self.enabled or amount <= 0:
            self.last_squeezed = self.last_total = 0
            return list(actions)
        out = []
        squeezed = 0
        for step in actions:
            new = dict(step)
            for key in _GRIPPER_KEYS:
                source = np.asarray(step[key])
                limit = source.dtype.type(self.squeeze_below)
                value = source.reshape(-1)[0]
                if value < limit:
                    value = max(source.dtype.type(0.0), value - source.dtype.type(amount))
                    squeezed += 1
                new[key] = np.array([value], dtype=source.dtype)
            out.append(new)
        self.last_squeezed, self.last_total = squeezed, len(actions) * len(_GRIPPER_KEYS)
        return out

    def raw_summary(self):
        """Summarize raw gripper outputs collected since the previous status window."""
        if not self._raw_n:
            return ""
        mean = self._raw_sum / self._raw_n
        probes = " ".join(f"{t:g}:{100.0 * self._raw_over[t] / self._raw_n:.0f}%" for t in _PROBE_THRESHOLDS)
        return (
            f"夹爪原始输出 n={self._raw_n} 范围 {self._raw_min:.3f}..{self._raw_max:.3f} "
            f"均值 {mean:.3f} | 判为张开的比例 {probes}"
        )


def resolve_real_task(value):
    """Resolve only the trained real tasks; never fall back to simulator slots."""
    if not isinstance(value, str):
        raise ValueError("A trained real task name or instruction is required")
    key = normalize_task_instruction(value)
    if key not in _TASKS:
        raise ValueError(f"Untrained GOAI real task {value!r}; supported: {GOAI_REAL_TASK_INSTRUCTIONS}")
    return _TASKS[key]


def resolve_real_task_nearest(value):
    """Resolve normalized task language, requiring a nearest-match score of at least 0.90."""
    if not isinstance(value, str):
        raise ValueError("A trained real task name or instruction is required")
    key = normalize_task_instruction(value)
    if key in _TASKS:
        index, instruction = _TASKS[key]
        return index, instruction, 1.0, True
    best = max(
        (
            (difflib.SequenceMatcher(None, key, candidate).ratio(), index, instruction)
            for candidate, (index, instruction) in _TASKS.items()
        ),
        key=lambda item: item[0],
    )
    ratio, index, instruction = best
    if ratio < _NEAREST_MIN_RATIO:
        raise ValueError(
            f"Untrained GOAI real task {value!r}; nearest {instruction!r} only matches "
            f"{ratio:.2f} (< {_NEAREST_MIN_RATIO}). Supported: {GOAI_REAL_TASK_INSTRUCTIONS}"
        )
    return index, instruction, ratio, False


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_checkpoint(checkpoint):
    """Check the trained ABC contract and its exact adjacent statistics before loading."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(f"Missing DCP metadata: {checkpoint}")
    step = checkpoint.parent if checkpoint.name == "ema" else checkpoint
    manifest = json.loads((step / "norm_stats_manifest.json").read_text())
    entries = [e for e in manifest["files"] if e.get("config_name") == "pi05_goai_joint"]
    if len(entries) != 1:
        raise ValueError("Expected exactly one pi05_goai_joint manifest entry")
    entry = entries[0]
    expected = {
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
    }
    for key, value in expected.items():
        if type(entry.get(key)) is not type(value) or entry[key] != value:
            raise ValueError(f"Unsupported ABC contract {key}={entry.get(key)!r}; expected {value!r}")
    stats = step / "norm_stats_pt.json"
    if hashlib.sha256(stats.read_bytes()).hexdigest() != entry.get("sha256"):
        raise ValueError("Checkpoint norm stats SHA256 mismatch")
    return checkpoint, stats


def canonical_observation(obs):
    """Copy decoded RGB input without resizing or changing the trained joint units."""
    if not isinstance(obs, Mapping):
        raise ValueError("Observation must be a mapping")
    if obs.get("images_preprocessed", False):
        raise ValueError("Official policy expects raw 640x480 images, not preprocessed images")
    raw_state = obs.get("state")
    if isinstance(raw_state, Mapping):
        pieces = []
        for key, size in _STATE_KEYS:
            piece = np.asarray(raw_state[key], dtype=np.float32)
            if piece.shape != (size,):
                raise ValueError(f"{key} must have shape ({size},)")
            pieces.append(piece)
        state = np.concatenate(pieces)
    else:
        state = np.asarray(raw_state, dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("State must be a finite 14D vector")
    source = obs.get("vision", obs.get("images"))
    if not isinstance(source, Mapping):
        raise ValueError("Observation requires decoded vision or images")
    images = {}
    for name, aliases in _CAMERAS.items():
        image = next((source[k] for k in aliases if k in source), None)
        if isinstance(image, Mapping):
            image = image.get("color", image.get("rgb"))
        image = np.asarray(image)
        if image.shape == (3, 480, 640):
            image = image.transpose(1, 2, 0)
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"{name} must be decoded uint8 RGB, 480x640x3 or 3x480x640")
        images[name] = np.array(image, copy=True, order="C")
    return {"state": state.copy(), "images": images}


class Model:
    """Implement ModelTemplate by duck typing, using the unchanged official server.

    One model instance belongs to one evaluator. Batch environments have separate
    observations and RNGs; the official no-argument reset resets the entire batch.
    """

    def __init__(self, model_cfg):
        if model_cfg.get("action_type", "joint") != "joint":
            raise ValueError("ABC checkpoints support joint actions only")
        configured_task = model_cfg.get("task_name")
        self.default_task = resolve_real_task(configured_task) if configured_task is not None else None
        self.seed = _integer(model_cfg.get("seed", 0), "seed")
        self.execution_horizon = _integer(model_cfg.get("execution_horizon", 8), "execution_horizon", 1)
        if self.execution_horizon > 32:
            raise ValueError("execution_horizon cannot exceed the trained horizon 32")
        num_steps = _integer(model_cfg.get("num_steps", 20), "num_steps", 1)
        if model_cfg.get("embodiment_index", 0) != 0:
            raise ValueError("This policy serves only the trained real-data embodiment index 0")
        checkpoint, stats = validate_checkpoint(model_cfg["checkpoint_path"])
        self.policy = self._load_policy(
            checkpoint=checkpoint,
            norm_stats=stats,
            task_name=None,
            device=model_cfg.get("device", "cuda:0"),
            action_horizon=self.execution_horizon,
            num_steps=num_steps,
            compile_mode=model_cfg.get("compile_mode", "none"),
            norm_mode="per-timestamp",
            embodiment_index=0,
            apply_delta=True,
            seed=self.seed,
            strict_checkpoint=True,
        )
        self.model = self.policy.model
        self.postprocess = PostProcess(model_cfg)
        self._last_task_note = "尚未收到观测"
        self._last_infer_ms = None
        self._status_at = None
        self.reset()

    @staticmethod
    def _load_policy(**kwargs):
        from pi.inference.goai_sim_policy import GOAISimPolicy

        return GOAISimPolicy(**kwargs)

    def reset(self):
        """Forget all observations and RNGs; a new observation is required after reset."""
        self._observations = {}
        self._sessions = {}
        self._episode_task = {}
        self._latest_ids = []

    def prepare_case(self, case_meta=None):
        """Validate a declared task; episode observations select the actual session task."""
        if isinstance(case_meta, Mapping) and case_meta.get("task_name") is not None:
            resolve_real_task_nearest(case_meta["task_name"])

    def _resolve_observation_task(self, obs):
        """Resolve agreeing observation task fields, using the configured task only if all are absent."""
        chosen = None
        for field in ("instruction", "prompt", "task_name"):
            value = obs.get(field)
            if value is None:
                continue
            index, instruction, ratio, exact = resolve_real_task_nearest(value)
            if not exact:
                logger.warning(
                    "观测 %s=%r 未精确匹配,按最近任务 %r 处理(相似度 %.2f)", field, value, instruction, ratio
                )
            self._last_task_note = f"slot {index} {instruction!r} <- {field}={value!r}" + (
                "" if exact else f"(最近匹配 {ratio:.2f})"
            )
            if chosen is None:
                chosen = (index, instruction)
            elif chosen[0] != index:
                raise ValueError(f"观测里 {field}={value!r} 与其它字段指向不同的任务")
        if chosen is not None:
            return chosen
        if self.default_task is None:
            raise ValueError("观测未提供 instruction/prompt/task_name,且服务配置也没有 task_name 兜底")
        return self.default_task

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if not isinstance(obs_list, (list, tuple)) or not obs_list:
            raise ValueError("A nonempty observation batch is required")
        pending, tasks = {}, {}
        for position, obs in enumerate(obs_list):
            if not isinstance(obs, Mapping):
                raise ValueError("Observation must be a mapping")
            env_id = _integer(obs.get("env_idx", position), "env_idx")
            if env_id in pending:
                raise ValueError(f"Duplicate env_idx {env_id}")
            tasks[env_id] = self._resolve_observation_task(obs)
            session = self._sessions.get(env_id)
            if session is not None and session.task_index != tasks[env_id][0]:
                raise ValueError("Cannot change an active episode task; reset before switching tasks")
            adapted = canonical_observation(obs)
            adapted["instruction"] = tasks[env_id][1]
            pending[env_id] = adapted
        # Commit only once the entire batch has passed validation.
        self._observations = pending
        self._episode_task = tasks
        self._latest_ids = list(pending)

    def _session(self, env_id):
        if env_id not in self._sessions:
            session = self.policy.create_session(seed=(self.seed + env_id) % (2**63))
            task = self._episode_task.get(env_id) or self.default_task
            if task is None:
                raise ValueError("session 没有可绑定的任务")
            # Bind the real slot directly: legacy simulator stack_bowls is slot 9.
            session.task_name, session.task_index = task[1], task[0]
            self._sessions[env_id] = session
        return self._sessions[env_id]

    def get_action(self):
        if len(self._latest_ids) != 1:
            raise ValueError("get_action requires exactly one latest observation")
        return self.get_action_batch(self._latest_ids)[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            ids = self._latest_ids
        else:
            if not isinstance(env_idx_list, (list, tuple, np.ndarray)):
                raise ValueError("env_idx_list must be a sequence")
            ids = [_integer(i, "env_idx") for i in env_idx_list]
        if not ids or len(ids) != len(set(ids)) or any(i not in self._observations for i in ids):
            raise ValueError("Request must reference unique env_idx values from the latest observation batch")
        began = time.perf_counter()
        result = []
        for env_id in ids:
            actions = self.policy.infer(self._observations[env_id], self._session(env_id))
            if len(actions) != self.execution_horizon:
                raise RuntimeError("Backend returned an unexpected action horizon")
            for action in actions:
                if set(action) != {key for key, _ in _STATE_KEYS}:
                    raise RuntimeError("Backend returned unexpected action keys")
                for key, size in _STATE_KEYS:
                    value = np.asarray(action[key])
                    if value.shape != (size,) or not np.isfinite(value).all():
                        raise RuntimeError(f"Backend returned invalid {key}")
            amount = self.postprocess.squeeze_for(self._session(env_id).task_index)
            actions = self.postprocess.gripper(actions, amount)
            result.append(actions)
        self._last_infer_ms = (time.perf_counter() - began) * 1000.0
        self._log_status()
        return result

    def _log_status(self):
        """Periodically report task selection, inference latency and raw gripper statistics."""
        now = time.monotonic()
        if self._status_at is not None and now - self._status_at < _STATUS_INTERVAL_S:
            return
        self._status_at = now
        latency = "—" if self._last_infer_ms is None else f"{self._last_infer_ms:.0f} ms"
        pp = self.postprocess
        gripper = ""
        raw = pp.raw_summary()
        if raw:
            state = f"已开启(下压 {pp.squeeze:g})" if pp.enabled else "未开启(仅统计)"
            gripper = f" | 夹爪后处理 {state},本批下压 {pp.last_squeezed}/{pp.last_total} | {raw}"
            pp.reset_raw_stats()  # 每个状态行窗口统计一次,不累加
        logger.warning("运行状态 | 任务 %s | 最近一次推理 %s%s", self._last_task_note, latency, gripper)

    def on_trial_end(self, result=None):
        """Release policy state; physical reset remains entirely with the evaluator."""
        del result
        self.reset()
