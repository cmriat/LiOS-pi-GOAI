# GOAI 2026 Submission Inference

Policy server for the GOAI 2026 bimanual track (RoboDojo Generalization
dimension). Self-contained WebSocket server implementing the XPolicyLab
v1.0.0 protocol (CALL frames + legacy infer), with an ops /healthz endpoint.

## Files

| Path | Role |
|--|--|
| `scripts/inference/goai/sim_server.py` | WS server: protocol dispatch, per-connection session/RNG, healthz |
| `src/pi/inference/goai_sim_policy.py` | Policy: checkpoint contract validation, observation prep, inference |
| `src/pi/inference/goai_observation.py` | GOAI camera mapping + image preprocessing |
| `src/pi/inference/goai_helpers.py` | Self-contained manifest/normalization helpers (no B1K deps) |

## Checkpoint

Trained on the official GOAI 2026 dataset (1200 episodes / 12 tasks).
Final submission checkpoint: `lion-vla-ckpt/ema`
(torch DCP format), with `norm_stats_pt.json` beside it.

## Environment

A minimal inference environment is provided (no training deps):

```bash
pixi install -e goai-inference
pixi run -e goai-inference python scripts/inference/goai/sim_server.py ...
```

## Run

```bash
pixi run -e goai-inference python scripts/inference/goai/sim_server.py \
  --checkpoint lion-vla-ckpt/ema \
  --norm-stats lion-vla-ckpt/norm_stats_pt.json \
  --task-name stack_bowls \
  --config-name pi05_goai_joint \
  --host 0.0.0.0 --port 28000 \
  --device cuda:0 \
  --action-horizon 8 --num-steps 20 \
  --compile-mode default \
  --norm-mode per-timestamp \
  --apply-delta --seed 0 --warmup
```

Health check:

```bash
curl http://127.0.0.1:28000/healthz
```

## Offline notes

- The paligemma tokenizer is bundled at `assets/paligemma_tokenizer.model` and
  loads from there by default (no network). Override with `GOAI_TOKENIZER_PATH`.
- Before packaging the checkpoint, strip training-machine paths from its
  manifest: `python scripts/inference/goai/sanitize_manifest.py lion-vla-ckpt/norm_stats_manifest.json`
- Server defaults are the frozen submission config: `--action-horizon 8` and
  `--num-steps 20` need no flags; `--apply-delta` remains an explicit required
  argument. No env injection required.

## Protocol

- v1.0.0 CALL: `hello` → `prepare_case` → `reset` → `call(update_obs)` →
  `call(get_action)` per step → `trial_end`. `get_action` returns 8-step ×
  14-dim absolute joint-position chunks (arm = delta + current state,
  grippers clipped to [0, 1]).
- Legacy `infer` frames are still accepted for backward compatibility.
- `action_case_id` must be `<task>_case` or `<task>_random_case` (12-task
  table: arrange_largest_number, fold_clothes, hang_mugs, make_toast,
  pack_objects_into_box, pour_liquid_into_cup, push_T,
  sort_nesting_dolls_by_size, stack_blocks, stack_bowls,
  store_laptop_and_headphones, sweep_blocks).
