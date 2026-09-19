# Optional fixed-length action resampling

This experiment uses more of the existing 32-step prediction while returning
exactly `execution_horizon` actions. It does not change model weights, denoising
steps, client code, or the active `configs/goai_real/server.yaml`.

The setting is read at server startup. It is not a hot-reload switch and does not
affect an already running server. Without the block, or with `enabled: false`,
inference follows the original `actions[:execution_horizon]` path.

## Configuration

Copy the current server config to a separate experiment config. Add the contents
of `configs/goai_real/action_resample.example.yaml` at the YAML top level:

```yaml
action_resample:
  enabled: true
  source_horizon: 32
  gripper_guard: 0.2
  max_step_rad: 0.10
```

Select that complete config with `GOAI_SERVER_CONFIG` when starting an experiment
through the existing `pixi run serve_goai` entry point. The example fragment alone
is not a runnable server config. A second simultaneous server needs its own port
and sufficient GPU resources; no experiment server is started by this change.

With `execution_horizon: 16`:

| Configuration | Prediction span | Returned actions |
| --- | --- | --- |
| Missing block / disabled | Original indices 0 through 15 | 16 |
| Enabled, source_horizon: 20 | Original indices 0 through 19 | 16 |
| Enabled, source_horizon: 24 | Original indices 0 through 23 | 16 |
| Enabled, source_horizon: 32 | Original indices 0 through 31 | 16 |

`source_horizon` must be an integer between `execution_horizon` and the checkpoint's
prediction horizon. Equal horizons are an exact identity. Compressing a longer
span requires at least two output actions. Unknown fields, invalid booleans,
nonfinite thresholds, and invalid horizons fail before model weights are loaded.
The new switch is independent of `postprocess.enabled`; enabling it does not enable
the existing gripper squeeze correction.

## Action semantics and guards

1. Generate all model actions with the existing sampler and RNG.
2. Unnormalize every original timestep with its original per-timestamp statistics.
3. Convert arm deltas to absolute joint targets; retain absolute gripper openings.
4. Apply pure NumPy PCHIP resampling to the selected source span, with both endpoints
   preserved and one shared time axis for all 14 dimensions.
5. Run the existing output validation, joint clipping, and optional gripper squeeze.

PCHIP does not overshoot adjacent source values. It does not guarantee successful
tracking or safe contact timing. This is trajectory time compression, not a
multiplication of joint positions or cumulative summation of deltas.

The guards fall back to the original first `execution_horizon` actions for that
chunk. They do not send a shorter chunk, clip velocities, or suppress existing
baseline actions:

- `gripper_guard`: maximum allowed range for either gripper over the entire source
  span, including the previously discarded tail. Default 0.2; valid values `(0, 1]`.
- `max_step_rad`: maximum absolute change per arm joint, checked from the observed
  state to the first output target and between consecutive output targets, before
  the existing joint clipping. Default 0.10; must be positive.
- Either guard can explicitly be set to `null` to disable it.

## Observability and validation

Enabled experiments print the effective configuration at startup and a status per
chunk. Existing trace `meta.json` includes `action_resample`; each `call` record
includes source/output horizons, `applied`, and the reason:
`resampled`, `identity`, `gripper_guard`, or `max_step_rad`. Disabled runs do not
add these trace fields. As before, recorded actions are the final output after
joint clipping and optional squeeze; full predictions are not newly recorded.

CPU tests cover legacy output and RNG equality across successive calls, original
timestamp normalization, shared arm/gripper timing, endpoints, monotonicity and
overshoot, guard fallbacks, schema validation, config forwarding, and trace fields.
They use synthetic CPU samples without loading checkpoints. No GPU experiment or
real-robot evaluation is implied by these tests.

For a controlled experiment, keep checkpoint, denoising steps, execution horizon,
and gripper correction fixed. Compare baseline with 20-to-16, 24-to-16, and finally
32-to-16. Check the applied/fallback counts as well as task progress per action
budget, tracking error, grasp failures, and drops.
