"""Self-contained GOAI inference helpers (no B1K / data-loader dependencies).

These functions were factored out of the B1K policy module so the GOAI
submission tree does not pull in the B1K data contract and LeRobot loader.
"""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import numpy as np


def normalize_quantile(array: np.ndarray, stats: Any) -> np.ndarray:
    """Apply the quantile normalization used by Pi05 training."""
    array = np.asarray(array, dtype=np.float64)
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("Quantile statistics q01/q99 are required")
    q01 = np.asarray(stats.q01)[..., : array.shape[-1]]
    q99 = np.asarray(stats.q99)[..., : array.shape[-1]]
    return (array - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(array: np.ndarray, stats: Any) -> np.ndarray:
    """Invert Pi05 quantile normalization."""
    array = np.asarray(array, dtype=np.float64)
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("Quantile statistics q01/q99 are required")
    q01 = np.asarray(stats.q01)[..., : array.shape[-1]]
    q99 = np.asarray(stats.q99)[..., : array.shape[-1]]
    return (array + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def normalize_zscore(array: np.ndarray, stats: Any) -> np.ndarray:
    """Apply mean/std normalization."""
    array = np.asarray(array, dtype=np.float64)
    mean = np.asarray(stats.mean)[..., : array.shape[-1]]
    std = np.asarray(stats.std)[..., : array.shape[-1]]
    return (array - mean) / (std + 1e-6)


def unnormalize_zscore(array: np.ndarray, stats: Any) -> np.ndarray:
    """Invert mean/std normalization."""
    array = np.asarray(array, dtype=np.float64)
    mean = np.asarray(stats.mean)[..., : array.shape[-1]]
    std = np.asarray(stats.std)[..., : array.shape[-1]]
    return array * (std + 1e-6) + mean


def _load_checkpoint_manifest_entry(checkpoint: Path, config_name: str) -> dict[str, Any]:
    manifest_path = checkpoint / "norm_stats_manifest.json"
    if not manifest_path.is_file() and checkpoint.name == "ema":
        parent_manifest_path = checkpoint.parent / "norm_stats_manifest.json"
        if parent_manifest_path.is_file():
            manifest_path = parent_manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint training manifest not found: {manifest_path}. "
            "Legacy checkpoints used wrong 640x480 padding and are not supported by this pipeline."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint manifest {manifest_path}: {error}") from error

    entries = [entry for entry in manifest.get("files", []) if entry.get("config_name") == config_name]
    if len(entries) != 1:
        raise ValueError(f"Expected one {config_name!r} entry in {manifest_path}, found {len(entries)}")
    entry = dict(entries[0])
    entry["_manifest_version"] = manifest.get("version")
    return entry


def normalize_values(array: np.ndarray, stats: Any, *, use_quantile_norm: bool) -> np.ndarray:
    """Apply the normalization family declared by the checkpoint contract."""
    if use_quantile_norm:
        return normalize_quantile(array, stats)
    return normalize_zscore(array, stats)


def unnormalize_values(array: np.ndarray, stats: Any, *, use_quantile_norm: bool) -> np.ndarray:
    """Invert the normalization family declared by the checkpoint contract."""
    if use_quantile_norm:
        return unnormalize_quantile(array, stats)
    return unnormalize_zscore(array, stats)


def _per_timestamp_stat_names(use_quantile_norm: bool) -> tuple[str, str]:
    if use_quantile_norm:
        return "per_timestamp_q01", "per_timestamp_q99"
    return "per_timestamp_mean", "per_timestamp_std"


def resolve_action_norm_mode(stats: Any, requested: str, *, use_quantile_norm: bool = True) -> str:
    """Validate global versus per-timestamp action statistics."""
    if requested not in ("global", "per-timestamp"):
        raise ValueError(f"Unsupported action norm mode: {requested}")

    first_name, second_name = _per_timestamp_stat_names(use_quantile_norm)
    has_first = getattr(stats, first_name) is not None
    has_second = getattr(stats, second_name) is not None
    if has_first != has_second:
        raise ValueError(f"Per-timestamp action stats must contain both {first_name} and {second_name}")
    if requested == "per-timestamp" and not has_first:
        raise ValueError(
            f"Per-timestamp action norm requested, but the stats do not contain {first_name}/{second_name}"
        )
    return requested


def unnormalize_actions(
    array: np.ndarray,
    stats: Any,
    mode: str,
    *,
    use_quantile_norm: bool,
) -> np.ndarray:
    """Invert the action normalization declared by the checkpoint contract."""
    mode = resolve_action_norm_mode(stats, mode, use_quantile_norm=use_quantile_norm)
    if mode == "global":
        return unnormalize_values(array, stats, use_quantile_norm=use_quantile_norm)

    array = np.asarray(array, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"Per-timestamp actions must have at least 2 dimensions, got {array.shape}")
    horizon, action_dim = array.shape[-2:]
    first_name, second_name = _per_timestamp_stat_names(use_quantile_norm)
    first = np.asarray(getattr(stats, first_name))
    second = np.asarray(getattr(stats, second_name))
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValueError(
            f"Invalid per-timestamp statistic shapes: {first_name}={first.shape}, {second_name}={second.shape}"
        )
    if first.shape[0] < horizon or first.shape[1] < action_dim:
        raise ValueError(
            f"Per-timestamp statistics have shape {first.shape}, but actions require at least {(horizon, action_dim)}"
        )
    first = first[:horizon, :action_dim]
    second = second[:horizon, :action_dim]
    if use_quantile_norm:
        return (array + 1.0) / 2.0 * (second - first + 1e-6) + first
    return array * (second + 1e-6) + first
