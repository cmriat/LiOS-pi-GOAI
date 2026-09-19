"""Official XPolicyLab policy interface for the six trained GOAI real tasks."""

from __future__ import annotations

import time
import difflib
import hashlib
import logging
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from pi.shared.goai_tasks import (
    GOAI_REAL_TASK_INSTRUCTIONS,
    GOAI_REAL_LEGACY_TASK_INSTRUCTIONS,
    normalize_task_instruction,
)
from pi.inference.goai_helpers import load_checkpoint_manifest, resolve_checkpoint_config_name
from pi.inference.goai_trace import TraceWriter

logger = logging.getLogger(__name__)

_STATUS_INTERVAL_S = 2.0

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
        for i, text in enumerate(GOAI_REAL_LEGACY_TASK_INSTRUCTIONS)
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
_ARM_KEYS = ("left_arm_joint_state", "right_arm_joint_state")

# Optional gripper subtraction in normalized opening units; disabled by default.
DEFAULT_GRIPPER_SQUEEZE = 0.05
DEFAULT_GRIPPER_SQUEEZE_BELOW = 0.90
_SQUEEZE_HARD_RANGE = (0.0, 0.5)
_PROBE_THRESHOLDS = (0.25, 0.50, 0.75, 0.90, 0.95)


def execution_horizon_for_task(model_cfg, task_instruction):
    """该任务的执行块长度:per_task 覆盖优先,否则全局 execution_horizon。

    给服务端之外的调用方(预热、验收脚本)用,避免它们各自硬编码全局值。
    task_instruction 可以是官方整句、短名或下划线 slug —— 一律归一到 task_index 再比,
    因为同一个任务的多种写法归一化后仍是不同的键(如 insert charger / Insert the charger)。
    """
    default = _integer(model_cfg.get("execution_horizon", 8), "execution_horizon", 1)
    per_task = model_cfg.get("per_task") or {}
    if not per_task:
        return default
    key = normalize_task_instruction(task_instruction)
    if key not in _TASKS:
        return default
    target = _TASKS[key][0]
    for name, block in per_task.items():
        alias = normalize_task_instruction(name)
        if alias in _TASKS and _TASKS[alias][0] == target:
            return _integer(block.get("execution_horizon", default), f"per_task.{name}.execution_horizon", 1)
    return default


class PostProcess:
    """Optional server-side corrections to model actions."""

    def __init__(self, model_cfg, per_task_gripper=None):
        if model_cfg.get("postprocess") is not None:
            raise ValueError(
                "postprocess 段已废弃:夹爪配置移到顶层 gripper:,按任务覆盖移到 per_task.<任务名>.gripper"
            )
        self.enabled, self.squeeze, self.squeeze_below = self._load_gripper(model_cfg.get("gripper"))
        # {task_index: (enabled, squeeze, squeeze_below)},由 Model 解析 per_task 段后传入。
        self.per_task = dict(per_task_gripper or {})
        self.joint_low, self.joint_high = self._load_joint_limits(model_cfg.get("joint_limits"))
        self.last_squeezed = 0
        self.last_total = 0
        self.last_clipped = 0
        self.last_clip_max = 0.0
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
    def _load_joint_limits(cls, raw):
        """Parse six [low, high] pairs from the arm SDK. Absent -> clipping is off.

        The server environment has no pyAgxArm, so the table lives in server.yaml;
        its provenance is recorded next to it there. Values are radians.
        """
        if raw is None:
            return None, None
        if not isinstance(raw, (list, tuple)) or len(raw) != 6:
            raise ValueError("joint_limits 需要 6 组 [下限, 上限](弧度)")
        low, high = [], []
        for index, pair in enumerate(raw):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"joint_limits[{index}] 需要 [下限, 上限]")
            try:
                lo, hi = float(pair[0]), float(pair[1])
            except (TypeError, ValueError):
                raise ValueError(f"joint_limits[{index}] 必须是数字") from None
            if not lo < hi:
                raise ValueError(f"joint_limits[{index}] 下限必须小于上限: [{lo}, {hi}]")
            low.append(lo)
            high.append(hi)
        return np.asarray(low, dtype=float), np.asarray(high, dtype=float)

    def joints(self, actions):
        """Clip arm joint targets into the SDK limits before they reach the client.

        ``deploy/controller.py:validate_chunk`` rejects the **entire** chunk and faults
        the episode when a joint target sits more than 0.02 rad outside the SDK limits,
        so a ~1.7 degree overshoot stops the run. Clipping here keeps the model's
        overshoot inside what the client tolerates.

        The client stays the enforcer: a stale table here costs amplitude, it cannot
        let an out-of-limit target through.
        """
        if self.joint_low is None:
            self.last_clipped, self.last_clip_max = 0, 0.0
            return list(actions)
        out = []
        clipped = 0
        worst = 0.0
        for step in actions:
            new = dict(step)
            for key in _ARM_KEYS:
                source = np.asarray(step[key])
                value = source.astype(float)
                bounded = np.clip(value, self.joint_low, self.joint_high)
                delta = np.abs(bounded - value)
                if delta.max() > 0:
                    clipped += int((delta > 0).sum())
                    worst = max(worst, float(delta.max()))
                new[key] = bounded.astype(source.dtype)
            out.append(new)
        self.last_clipped, self.last_clip_max = clipped, worst
        return out

    @classmethod
    def _load_gripper(cls, raw, where="gripper"):
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where} 必须是 map")
        unknown = sorted(set(raw) - {"enabled", "squeeze", "squeeze_below"})
        if unknown:
            raise ValueError(f"{where} 有未知字段 {unknown};可选: ['enabled', 'squeeze', 'squeeze_below']")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{where}.enabled 必须是布尔")
        squeeze = cls._check_squeeze(raw.get("squeeze", DEFAULT_GRIPPER_SQUEEZE), f"{where}.squeeze")
        try:
            below = float(raw.get("squeeze_below", DEFAULT_GRIPPER_SQUEEZE_BELOW))
        except (TypeError, ValueError):
            raise ValueError(f"{where}.squeeze_below 必须是数字") from None
        if not 0.5 <= below <= 1.0:
            raise ValueError(
                f"{where}.squeeze_below 需要在 [0.5, 1.0];"
                "低于 0.5 会把大量过渡帧也下压,高于 1.0 等于一直下压"
            )
        return enabled, squeeze, below

    def spec_for(self, task_index):
        """本任务生效的 (enabled, squeeze, squeeze_below);没有按任务覆盖就用全局。"""
        if task_index is not None and int(task_index) in self.per_task:
            return self.per_task[int(task_index)]
        return self.enabled, self.squeeze, self.squeeze_below

    def squeeze_for(self, task_index):
        """本任务配置的下压量(不含 enabled 门;是否生效由 gripper() 的 enabled 决定)。"""
        return self.spec_for(task_index)[1]

    def gripper(self, actions, amount, below=None, enabled=None):
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
        if not (self.enabled if enabled is None else enabled) or amount <= 0:
            self.last_squeezed = self.last_total = 0
            return list(actions)
        out = []
        squeezed = 0
        for step in actions:
            new = dict(step)
            for key in _GRIPPER_KEYS:
                source = np.asarray(step[key])
                limit = source.dtype.type(self.squeeze_below if below is None else below)
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
    """Check the checkpoint is complete and carries the statistics its weights were trained against.

    Two file-level checks, and deliberately nothing else:

    - the DCP ``.metadata`` exists -- without it this is not a checkpoint at all;
    - ``norm_stats_pt.json`` hashes to a digest the selected training manifest entry declares.
      Unnormalizing with statistics the weights never saw produces garbage actions
      *silently*, which is the one failure here worth paying to catch.

    It does **not** compare the training recipe against a frozen table. ``GOAISimPolicy``
    rebuilds the model config from the checkpoint's own manifest entry (``dataclasses.replace``
    over eleven architecture fields), so the weights are authoritative for their own shape,
    and a structural mismatch is still caught downstream by ``validate_goai_dcp_coverage``
    on tensor shapes. A pinned recipe table can only go stale: the run that moved to six
    tasks and a single embodiment (2026-09-15) was rejected wholesale by one.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not (checkpoint / ".metadata").is_file():
        raise FileNotFoundError(f"Missing DCP metadata: {checkpoint}")
    step = checkpoint.parent if checkpoint.name == "ema" else checkpoint
    manifest = load_checkpoint_manifest(checkpoint)
    stats = step / "norm_stats_pt.json"
    config_name = resolve_checkpoint_config_name(checkpoint)
    entries = [entry for entry in manifest.get("files", []) if entry.get("config_name") == config_name]
    if len(entries) != 1:
        raise ValueError(f"Expected one {config_name!r} training entry, found {len(entries)}")
    if hashlib.sha256(stats.read_bytes()).hexdigest() != entries[0].get("sha256"):
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
        self.num_steps = num_steps
        # 按任务覆盖:任务名 -> task_index,启动时校验;不写的任务走上面的全局默认。
        self.per_task_names, self.per_task = self._parse_per_task(model_cfg.get("per_task"))
        self.execution_horizon_by_task: dict[int, int] = {}
        self.num_steps_by_task: dict[int, int] = {}
        for task_index, block in self.per_task.items():
            where = f"per_task.{self.per_task_names[task_index]}"
            horizon = _integer(block.get("execution_horizon", self.execution_horizon), f"{where}.execution_horizon", 1)
            if horizon > 32:
                raise ValueError(f"{where}.execution_horizon cannot exceed the trained horizon 32")
            steps = _integer(block.get("num_steps", num_steps), f"{where}.num_steps", 1)
            self.execution_horizon_by_task[task_index] = horizon
            self.num_steps_by_task[task_index] = steps
        per_task_gripper = {}
        for task_index, block in self.per_task.items():
            if "gripper" in block:
                per_task_gripper[task_index] = PostProcess._load_gripper(
                    block["gripper"], where=f"per_task.{self.per_task_names[task_index]}.gripper"
                )
        if model_cfg.get("embodiment_index", 0) != 0:
            raise ValueError("This policy serves only the trained real-data embodiment index 0")
        checkpoint, stats = validate_checkpoint(model_cfg["checkpoint_path"])
        self.policy = self._load_policy(
            checkpoint=checkpoint,
            norm_stats=stats,
            # 架构随权重走:从 ckpt 自己的 manifest 读,不写死,否则换一条训练线就得改代码。
            config_name=resolve_checkpoint_config_name(checkpoint),
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
            action_resample=model_cfg.get("action_resample"),
            task_settings={
                task_index: {
                    "execution_horizon": self.execution_horizon_by_task[task_index],
                    "num_steps": self.num_steps_by_task[task_index],
                    "action_resample": self.per_task[task_index].get(
                        "action_resample", model_cfg.get("action_resample")
                    ),
                }
                for task_index in self.per_task
            },
        )
        self.model = self.policy.model
        self.postprocess = PostProcess(model_cfg, per_task_gripper=per_task_gripper)
        self.trace = TraceWriter(model_cfg)
        resamplers = getattr(self.policy, "_resamplers", None) or {}
        summary = {
            ("global" if key is None else self.per_task_names.get(key, str(key))): value.summary()
            for key, value in resamplers.items()
            if getattr(value, "enabled", False)
        }
        resample_meta = {"action_resample": summary} if summary else {}
        if summary:
            logger.warning("Action resampling configured: %s", summary)
        if self.per_task:
            logger.warning("Per-task overrides: %s", self.per_task_summary())
        self.trace.write_meta(
            {
                "policy_name": model_cfg.get("policy_name"),
                "checkpoint_path": str(model_cfg.get("checkpoint_path")),
                "execution_horizon": model_cfg.get("execution_horizon"),
                "num_steps": model_cfg.get("num_steps"),
                "per_task": self.per_task_summary(),
                **resample_meta,
            }
        )
        self._raw_obs = {}
        self._obs_at = {}
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
        self._raw_obs = {}
        self._obs_at = {}
        self.trace.event("reset")

    def prepare_case(self, case_meta=None):
        """Validate a declared task; episode observations select the actual session task."""
        self.trace.event("prepare_case", {"case_meta": case_meta})
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
            self._obs_at[env_id] = time.perf_counter()  # 观测到达时刻，用于算端到端
            if env_id in pending:
                raise ValueError(f"Duplicate env_idx {env_id}")
            self._raw_obs[env_id] = {
                "fields": {f: obs.get(f) for f in ("instruction", "prompt", "task_name") if obs.get(f) is not None},
                "keys": sorted(str(k) for k in obs.keys()),
                "images_preprocessed": obs.get("images_preprocessed"),
            }
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

    @staticmethod
    def _parse_per_task(raw):
        """per_task 段 -> (任务名表, {task_index: 覆盖块});未知任务名/字段启动即报错。"""
        if raw is None:
            return {}, {}
        if not isinstance(raw, Mapping):
            raise ValueError("per_task 段必须是 map")
        names, parsed = {}, {}
        for name, block in raw.items():
            key = normalize_task_instruction(name)
            if key not in _TASKS:
                raise ValueError(f"per_task 里的 {name!r} 不是已知任务;可选: {sorted(_SLUGS)}")
            if not isinstance(block, Mapping):
                raise ValueError(f"per_task.{name} 必须是 map")
            unknown = sorted(set(block) - {"execution_horizon", "num_steps", "action_resample", "gripper"})
            if unknown:
                raise ValueError(
                    f"per_task.{name} 有未知字段 {unknown};"
                    "可选: ['execution_horizon', 'num_steps', 'action_resample', 'gripper']"
                )
            task_index = _TASKS[key][0]
            if task_index in parsed:
                raise ValueError(f"per_task 里 {name!r} 与另一个别名指向同一个任务,只能写一次")
            names[task_index] = name
            parsed[task_index] = dict(block)
        return names, parsed

    def execution_horizon_for(self, task_index):
        """本任务的执行块长度;没有按任务覆盖就用全局。"""
        if task_index is None:
            return self.execution_horizon
        return self.execution_horizon_by_task.get(int(task_index), self.execution_horizon)

    def num_steps_for(self, task_index):
        if task_index is None:
            return self.num_steps
        return self.num_steps_by_task.get(int(task_index), self.num_steps)

    def per_task_summary(self):
        """人读的按任务覆盖表,写进 meta.json / 启动日志。"""
        if not self.per_task:
            return None
        return {
            self.per_task_names[task_index]: {
                "execution_horizon": self.execution_horizon_by_task[task_index],
                "num_steps": self.num_steps_by_task[task_index],
                **({"action_resample": block["action_resample"]} if "action_resample" in block else {}),
                **({"gripper": block["gripper"]} if "gripper" in block else {}),
            }
            for task_index, block in self.per_task.items()
        }

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
        # 本次调用【到达】服务端的时刻。它与 update_obs 的到达时刻之差，就是客户端
        # 两次调用之间的间隔（含网络往返与客户端自身处理），即我们要的端到端延迟。
        called_at = time.perf_counter()
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
            infer_began = time.perf_counter()
            actions = self.policy.infer(self._observations[env_id], self._session(env_id))
            # wall_ms is measured here, so it stays meaningful even for a backend
            # that reports no stage breakdown of its own.
            timing = dict(getattr(self.policy, "last_timing", None) or {})
            timing["wall_ms"] = (time.perf_counter() - infer_began) * 1000.0
            if len(actions) != self.execution_horizon_for(self._session(env_id).task_index):
                raise RuntimeError("Backend returned an unexpected action horizon")
            for action in actions:
                if set(action) != {key for key, _ in _STATE_KEYS}:
                    raise RuntimeError("Backend returned unexpected action keys")
                for key, size in _STATE_KEYS:
                    value = np.asarray(action[key])
                    if value.shape != (size,) or not np.isfinite(value).all():
                        raise RuntimeError(f"Backend returned invalid {key}")
            actions = self.postprocess.joints(actions)
            squeeze_enabled, amount, below = self.postprocess.spec_for(self._session(env_id).task_index)
            actions = self.postprocess.gripper(actions, amount, below=below, enabled=squeeze_enabled)
            infer_ms = timing.get("wall_ms")
            if infer_ms is None:
                infer_ms = (time.perf_counter() - began) * 1000.0
            obs_at = self._obs_at.get(env_id)
            e2e_ms = None if obs_at is None else (called_at - obs_at) * 1000.0
            resample_status = getattr(self.policy, "last_resample", None)
            if resample_status is not None:
                logger.warning("Action resampling env=%s: %s", env_id, resample_status)
            self.trace.record(
                env_id=env_id,
                task_index=self._session(env_id).task_index,
                obs=self._observations[env_id],
                actions=actions,
                raw=self._raw_obs.get(env_id),
                timing=timing,
                infer_ms=infer_ms,
                e2e_ms=e2e_ms,
                action_resample=resample_status,
            )
            self._last_infer_ms = infer_ms
            self._log_call(e2e_ms)
            result.append(actions)
        return result

    def _log_call(self, e2e_ms=None):
        """One line per inference: task (prompt + slot id) and both latencies.

        infer_ms = 模型前向本身；e2e_ms = 客户端两次调用（update_obs → get_action）
        到达服务端的间隔——含网络往返与客户端自身处理，**不含服务端推理**。
        两者相加 ≈ 客户端从发出观测到收到动作的全部等待（仍不含动作回包的返程网络）。
        """
        latency = "—" if self._last_infer_ms is None else f"{self._last_infer_ms:.0f} ms"
        tail = "" if e2e_ms is None else f" | 端到端 {e2e_ms:.0f} ms"
        logger.warning("[推理] %s | 推理 %s%s", self._last_task_note, latency, tail)

    def on_trial_end(self, result=None):
        """Release policy state; physical reset remains entirely with the evaluator."""
        self.trace.event("trial_end", {"result": result})
        self.reset()
