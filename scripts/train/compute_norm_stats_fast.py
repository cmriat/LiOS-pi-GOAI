r"""Fast compute_norm_stats: scalar-only direct path for VideoLance datasets.

Bypasses the DataLoader / torchrun / DistributedSampler stack of
compute_norm_stats.py. Reads the state, action and episode_index columns directly
from the Lance dataset, applies the same delta action transform as training, and
computes normalization statistics in a single process via numpy and the canonical
RunningStats class.

Output is a drop-in replacement: it writes through `normalize.save`, so the
resulting norm_stats.json is bit-compatible (mean/std) with the canonical script.
Quantile bin edges can differ by a fraction of a percent due to histogram
redistribution noise, which does not affect mean/std-based normalization.

The canonical script only consumes batch["state"] and batch["actions"]; image
columns are decoded but never read, so skipping the DataLoader removes almost all
of the wall time.

Usage (single process; do NOT launch with torchrun):
    pixi run -e dev python scripts/train/compute_norm_stats_fast.py \
        pi05_goai_joint_lance \
        --data.dataset-format video_lance \
        --data.dataset-uri /path/to/goai_2026_joint.lance \
        --data.asset-id /abs/path/to/assets/pi05_goai_joint_lance \
        --model.action-horizon 32

    Add --data.use-per-timestamp-action-norm to also write
    actions.per_timestamp_{mean,std,q01,q99} for per-horizon action normalization.

    --model.action-horizon must match the training run: the action stack is cut at
    that horizon, so mismatched stats normalize a different chunk length than the
    model predicts.
"""

from __future__ import annotations

import time
import pathlib

import numpy as np
import torch

import pi.shared.normalize as normalize
import pi.training.instance_config as train_config
from pi.data import (
    _dataset_uri_values,
    project_policy_state,
    _get_delta_action_mask,
    resolve_policy_state_schema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stack_fixed_size_list(col) -> np.ndarray:
    """Convert a pyarrow FixedSizeList column to a contiguous 2D ndarray.

    Tries the zero-copy fast path; falls back to per-row stacking. Lance
    typically returns a single chunk for a contiguous read like ours.
    """
    try:
        return np.stack(col.to_numpy(zero_copy_only=False))
    except Exception:
        return np.asarray(col.to_pylist(), dtype=np.float32)


def _build_episode_end_per_frame(ep_idx: np.ndarray) -> np.ndarray:
    """For each frame, the exclusive end-row index of its episode.

    Episodes are detected as contiguous runs of equal episode_index, the same
    convention the VideoLance dataset uses when it builds its scalar index cache.
    """
    n = len(ep_idx)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    diffs = np.where(np.diff(ep_idx) != 0)[0] + 1
    boundaries = np.concatenate([diffs, [n]]).astype(np.int64)
    end_per_frame = np.empty(n, dtype=np.int64)
    cursor = 0
    for b in boundaries:
        end_per_frame[cursor:b] = b
        cursor = b
    return end_per_frame


def _build_action_horizon_stack(
    action_arr: np.ndarray,  # (N, dim)
    end_per_frame: np.ndarray,  # (N,)
    horizon: int,
) -> np.ndarray:
    """For each frame i, gather action[clamp(i+offset, end-1)] for offset in 0..H-1.

    Mirrors the dataset's scalar-delta query: the last action of an episode is
    repeated for every horizon step that reaches past the episode end, so a chunk
    never crosses into the next episode.
    """
    n = action_arr.shape[0]
    offsets = np.arange(horizon, dtype=np.int64).reshape(1, -1)  # (1, H)
    base = np.arange(n, dtype=np.int64).reshape(-1, 1)  # (N, 1)
    clamped = np.minimum(base + offsets, end_per_frame.reshape(-1, 1) - 1)
    return action_arr[clamped]  # (N, H, dim)


def _apply_delta_joint_vectorized(
    state_arr: np.ndarray,  # (N, dim)
    action_stack: np.ndarray,  # (N, H, dim)
    mask: np.ndarray,  # (dim,) bool
) -> np.ndarray:
    """Vectorized equivalent of pi.data._apply_delta_actions.

    The reference implementation subtracts the current state from the masked
    action dimensions; with a single state frame the current state is the row
    itself, so the whole dataset can be transformed in one pass.
    """
    dims = mask.shape[-1]
    delta = np.where(mask, state_arr[:, :dims], 0.0).astype(action_stack.dtype, copy=False)
    out = action_stack.copy()
    out[..., :dims] -= delta[:, None, :]
    return out


def _read_required_columns(uri: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Pull just the three columns we need (state / action / episode_index).

    Returns (state_arr, action_arr, episode_index_arr, state_col_name).
    """
    import lance

    ds = lance.dataset(uri)
    schema_names = {f.name for f in ds.schema}

    # Converted datasets use the Lance-native 'observation_state' column; the
    # loader bridges it back to 'observation.state' at read time. Accept both.
    if "observation_state" in schema_names:
        state_col = "observation_state"
    elif "observation.state" in schema_names:
        state_col = "observation.state"
    else:
        raise SystemExit(
            f"dataset has neither 'observation_state' nor 'observation.state'; got {sorted(schema_names)[:20]}..."
        )
    for required in ("action", "episode_index"):
        if required not in schema_names:
            raise SystemExit(f"dataset missing required column: {required!r}")

    tbl = ds.to_table(columns=[state_col, "action", "episode_index"])
    state_arr = _stack_fixed_size_list(tbl[state_col]).astype(np.float32, copy=False)
    action_arr = _stack_fixed_size_list(tbl["action"]).astype(np.float32, copy=False)
    ep_idx = tbl["episode_index"].to_numpy().astype(np.int64, copy=False)
    return state_arr, action_arr, ep_idx, state_col


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = train_config.cli()

    if config.data.dataset_format != "video_lance":
        raise SystemExit(
            "compute_norm_stats_fast.py only supports dataset_format=video_lance. "
            "For lerobot datasets, use the canonical compute_norm_stats.py."
        )

    uris = _dataset_uri_values(config.data.dataset_uri, config.data.repo_id or "")
    if not uris or not uris[0]:
        raise SystemExit("No dataset URI or repo_id provided.")

    asset_id = config.data.asset_id or config.data.repo_id
    target_dir = pathlib.Path(config.assets_dirs) / asset_id
    target_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(config.model.action_horizon)
    policy_name = config.name
    apply_delta = bool(config.data.apply_delta_transform)
    use_per_timestamp = bool(config.data.use_per_timestamp_action_norm)
    policy_state_schema = resolve_policy_state_schema(policy_name, config.data.policy_state_schema)

    print(f"[fast] uri_count             = {len(uris)}")
    for idx, uri in enumerate(uris):
        print(f"[fast] uri[{idx}]               = {uri}")
    print(f"[fast] target_dir            = {target_dir}")
    print(f"[fast] policy_name           = {policy_name}")
    print(f"[fast] action_horizon        = {horizon}")
    print(f"[fast] apply_delta_transform = {apply_delta}")
    print(f"[fast] per_timestamp_actions = {use_per_timestamp}")
    print(f"[fast] policy_state_schema   = {policy_state_schema}")

    # The GPU is only a constant-factor accelerator for the histogram update.
    device = "cuda:0" if torch.cuda.is_available() else None
    t0 = time.time()
    state_stats = normalize.RunningStats(device=device)
    action_stats = normalize.RunningStats(device=device)
    per_t_action_stats = [normalize.RunningStats(device=device) for _ in range(horizon)] if use_per_timestamp else None
    expected_state_dim = None
    expected_action_dim = None
    total_rows = 0

    for uri_idx, uri in enumerate(uris):
        read_t = time.time()
        print(f"[fast] reading scalar columns ({uri_idx + 1}/{len(uris)}) ...", flush=True)
        state_arr, action_arr, ep_idx, state_col = _read_required_columns(uri)
        n, state_dim = state_arr.shape
        action_dim = action_arr.shape[-1]
        total_rows += n
        print(f"[fast]   state_col={state_col!r}  N={n}  state_dim={state_dim}  action_dim={action_dim}")
        print(f"[fast]   read took {time.time() - read_t:.2f}s")

        if expected_state_dim is None:
            expected_state_dim = state_dim
            expected_action_dim = action_dim
        elif state_dim != expected_state_dim or action_dim != expected_action_dim:
            raise SystemExit(
                f"All datasets must share state/action dims. Expected "
                f"state_dim={expected_state_dim}, action_dim={expected_action_dim}; "
                f"got state_dim={state_dim}, action_dim={action_dim} for {uri!r}."
            )
        if state_dim != action_dim:
            raise SystemExit(
                f"state_dim ({state_dim}) != action_dim ({action_dim}); the joint delta transform assumes equal dims."
            )
        # Catch a policy/data mismatch early: a dataset whose dims differ from the
        # policy's delta mask would otherwise produce stats of the wrong width.
        if apply_delta:
            mask_len = len(_get_delta_action_mask(policy_name))
            if mask_len != state_dim:
                raise SystemExit(
                    f"policy {policy_name!r} expects {mask_len}-dim state/action "
                    f"but dataset has {state_dim}-dim. "
                    "Pick a policy whose delta_action_mask matches the data dims."
                )

        stack_t = time.time()
        print(f"[fast] building (N={n}, H={horizon}, D={action_dim}) action stack ...", flush=True)
        end_per_frame = _build_episode_end_per_frame(ep_idx)
        action_stack = _build_action_horizon_stack(action_arr, end_per_frame, horizon)
        print(
            f"[fast]   stack took {time.time() - stack_t:.2f}s, "
            f"shape={action_stack.shape}, mem={action_stack.nbytes / 1e6:.1f}MB"
        )

        if apply_delta:
            delta_t = time.time()
            mask = np.asarray(_get_delta_action_mask(policy_name), dtype=bool)
            action_stack = _apply_delta_joint_vectorized(state_arr, action_stack, mask)
            print(f"[fast]   delta took {time.time() - delta_t:.2f}s")

        projected_state_arr = project_policy_state(state_arr, policy_name, policy_state_schema)
        stats_t = time.time()
        print(f"[fast] updating stats on device={device or 'cpu'} ...", flush=True)
        state_stats.update(projected_state_arr)
        action_stats.update(action_stack)
        if per_t_action_stats is not None:
            for t in range(horizon):
                per_t_action_stats[t].update(action_stack[:, t, :])
        print(f"[fast]   stats update took {time.time() - stats_t:.2f}s")

    action_norm_stats = action_stats.get_statistics()
    if per_t_action_stats is not None:
        per_t = [stats.get_statistics() for stats in per_t_action_stats]
        action_norm_stats = normalize.NormStats(
            mean=action_norm_stats.mean,
            std=action_norm_stats.std,
            q01=action_norm_stats.q01,
            q99=action_norm_stats.q99,
            per_timestamp_mean=np.stack([stats.mean for stats in per_t], axis=0),
            per_timestamp_std=np.stack([stats.std for stats in per_t], axis=0),
            per_timestamp_q01=np.stack([stats.q01 for stats in per_t], axis=0),
            per_timestamp_q99=np.stack([stats.q99 for stats in per_t], axis=0),
        )
        print(f"[fast] added per-timestamp action stats: shape={action_norm_stats.per_timestamp_mean.shape}")

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_norm_stats,
    }

    out = target_dir / "norm_stats.json"
    print(f"[fast] writing {out}", flush=True)
    normalize.save(target_dir, norm_stats)
    print(f"[fast] done. rows={total_rows} total {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
