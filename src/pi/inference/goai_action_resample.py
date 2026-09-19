"""Optional fixed-length resampling of decoded GOAI joint targets."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

_ARM_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
_GRIPPER_INDICES = (6, 13)


def _pchip(values: np.ndarray, output_steps: int) -> np.ndarray:
    """Resample uniformly spaced points with shape-preserving cubic Hermite interpolation."""
    y = values.astype(np.float64)
    delta = np.diff(y, axis=0)
    slopes = np.zeros_like(y)
    if len(y) == 2:
        slopes[:] = delta[0]
    else:
        left, right = delta[:-1], delta[1:]
        same_sign = (left * right) > 0
        slopes[1:-1][same_sign] = 2 * left[same_sign] * right[same_sign] / (left[same_sign] + right[same_sign])
        for index, edge, neighbor in ((0, delta[0], delta[1]), (-1, delta[-1], delta[-2])):
            slope = (3 * edge - neighbor) / 2
            slopes[index] = np.where(
                np.sign(slope) != np.sign(edge),
                0,
                np.sign(edge) * np.minimum(np.abs(slope), 3 * np.abs(edge)),
            )

    positions = np.linspace(0, len(y) - 1, output_steps)
    indices = np.minimum(positions.astype(int), len(y) - 2)
    u = (positions - indices)[:, None]
    result = (
        (2 * u**3 - 3 * u**2 + 1) * y[indices]
        + (u**3 - 2 * u**2 + u) * slopes[indices]
        + (-2 * u**3 + 3 * u**2) * y[indices + 1]
        + (u**3 - u**2) * slopes[indices + 1]
    )
    result = result.astype(values.dtype)
    result[0], result[-1] = values[0], values[-1]
    return result


@dataclass(frozen=True)
class ActionResampler:
    """Select a longer prediction span without changing the execution horizon."""

    enabled: bool
    source_horizon: int
    execution_horizon: int
    gripper_guard: float | None
    max_step_rad: float | None

    @classmethod
    def from_config(cls, raw, *, execution_horizon: int, model_horizon: int) -> ActionResampler:
        """Validate experiment settings before model weights are loaded."""
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("action_resample must be a mapping")
        unknown = sorted(set(raw) - {"enabled", "source_horizon", "gripper_guard", "max_step_rad"})
        if unknown:
            raise ValueError(f"Unknown action_resample fields: {unknown}")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("action_resample.enabled must be a boolean")
        source = raw.get("source_horizon", model_horizon)
        if isinstance(source, bool) or not isinstance(source, int) or not execution_horizon <= source <= model_horizon:
            raise ValueError(
                f"action_resample.source_horizon must be an integer in [{execution_horizon}, {model_horizon}]"
            )
        if enabled and source > execution_horizon and execution_horizon < 2:
            raise ValueError("action_resample requires execution_horizon >= 2 to preserve both endpoints")

        def threshold(name, default, upper=None):
            value = raw.get(name, default)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"action_resample.{name} must be a finite number or null")
            if value <= 0 or (upper is not None and value > upper):
                raise ValueError(f"action_resample.{name} must be positive" + (f" and <= {upper}" if upper else ""))
            return float(value)

        return cls(
            enabled, source, execution_horizon, threshold("gripper_guard", 0.2, 1), threshold("max_step_rad", 0.10)
        )

    def summary(self) -> dict:
        """Return the effective configuration for logs and run metadata."""
        return {
            "enabled": self.enabled,
            "source_horizon": self.source_horizon,
            "execution_horizon": self.execution_horizon,
            "method": "pchip",
            "gripper_guard": self.gripper_guard,
            "max_step_rad": self.max_step_rad,
        }

    def select(self, actions: np.ndarray, current_state: np.ndarray) -> tuple[np.ndarray, dict]:
        """Resample absolute targets, or return the original prefix when a guard fires.

        All source timestamps must already have been unnormalized using their own
        statistics and arm deltas converted to absolute targets. The downstream
        joint limits and optional gripper correction still run on the result.
        """
        baseline = actions[: self.execution_horizon]
        status = {
            "source_horizon": self.source_horizon,
            "output_steps": self.execution_horizon,
            "applied": False,
            "reason": "disabled",
        }
        if not self.enabled:
            return baseline, status
        if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < self.source_horizon:
            raise ValueError("action_resample requires enough decoded 14D source actions")
        source = actions[: self.source_horizon]
        state = np.asarray(current_state)
        if state.shape != (14,) or not np.isfinite(state).all() or not np.isfinite(source).all():
            raise ValueError("action_resample requires finite 14D actions and state")
        if self.source_horizon == self.execution_horizon:
            status["reason"] = "identity"
            return baseline, status
        if self.gripper_guard is not None and np.ptp(source[:, _GRIPPER_INDICES], axis=0).max() > self.gripper_guard:
            status["reason"] = "gripper_guard"
            return baseline, status

        # Arms and grippers share the same time axis, including both endpoints.
        output = _pchip(source, self.execution_horizon)
        if self.max_step_rad is not None:
            arm_path = np.concatenate([state[None, _ARM_INDICES], output[:, _ARM_INDICES]], axis=0)
            max_step = float(np.abs(np.diff(arm_path, axis=0)).max())
            status["max_step_rad"] = max_step
            if max_step > self.max_step_rad:
                status["reason"] = "max_step_rad"
                return baseline, status
        status.update(applied=True, reason="resampled")
        return output, status
