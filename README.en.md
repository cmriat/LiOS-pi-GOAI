# LiOS-pi-GOAI

**A self-contained Pi0.5 vision-language-action stack for six-task dual-arm manipulation on the PiperX robot.**

An extension of [`lios-pi`](https://github.com/cmriat/LiOS/tree/main/lios-pi) — the Pi0 / Pi0.5 VLA
subproject of the [**LiOS**](https://github.com/cmriat/LiOS) embodied-AI infrastructure stack.

**English** | [简体中文](README.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](pixi.toml)

---

## Overview

LiOS-pi-GOAI packages everything needed to reproduce a six-task dual-arm policy on the GOAI
2026 real-robot track: the model, the training stack that produced it, and a serving layer that
speaks the **XPolicyLab v1.0.0** protocol without modifying the official evaluation code.

The policy is a Pi0.5 VLA fine-tuned on six real-robot manipulation tasks. Inference is
self-contained — no external private dependencies — so an evaluator can reproduce a run from
this repository plus a separately delivered checkpoint.

## Relation to LiOS

This project **extends [`lios-pi`](https://github.com/cmriat/LiOS/tree/main/lios-pi)** rather
than forking it. The upstream [LiOS](https://github.com/cmriat/LiOS) repository describes the
general embodied-AI stack (`lios-pi` for VLA models, `lios-webrtc` for edge→cloud image
transport); this repository holds the GOAI-2026-specific work built on top of it.

| | |
|---|---|
| **Adapted from `lios-pi`** | Pi0 / Pi0.5 PyTorch implementation (ported upstream from [openpi](https://github.com/Physical-Intelligence/openpi)), the PaliGemma / Gemma / SigLIP transformer stack, FSDP training loop, normalization and image utilities |
| **Added in this repository** | GOAI real-robot task suite and dataset pipeline, the PiperX 14-D state/action contract, the XPolicyLab serving layer and `lionvla` policy plugin, action resampling, per-task inference configuration, on-disk inference tracing, and the competition submission documents |

Attribution is recorded in [`NOTICE`](NOTICE).

## Highlights

- **Self-contained inference** — the model, tokenizer and serving code are all in-repo; evaluation
  needs no private packages and no network access at run time.
- **Official server left untouched** — the policy plugs into the unmodified official XPolicyLab
  server through a thin `policy/lionvla/` adapter. No control logic lives in this repository's
  client-side code.
- **PiperX joint-space contract** — a documented 14-dimensional state/action layout
  (6 left-arm joints + left gripper + 6 right-arm joints + right gripper), shared by training
  and inference.
- **Optional action resampling** — a shape-preserving (PCHIP) time compression of the predicted
  chunk, which trades trajectory playback speed for a longer effective prediction span.
  Off by default. See [`docs-GOAI/action_resampling.md`](docs-GOAI/action_resampling.md).
- **Inference tracing** — every call can be recorded to disk (camera frames, state, the final
  action chunk and per-stage latency) for offline review, with a bundled
  [viewer](tools/trace_viewer.py). Off by default.

## Getting started

Requires [pixi](https://pixi.sh). The environment is locked to Python 3.10 / PyTorch 2.7.1 (CUDA 12.9).

```bash
pixi install --locked

# Point checkpoint_path in configs/goai_real/server.yaml at the delivered checkpoint,
# then start the real-robot policy server.
pixi run serve_goai
```

The server prints a `READY FOR INFERENCE ws://<host>:<port>` line once it is serving.

For the standalone simulation server and a protocol smoke test:

```bash
pixi run -e goai-inference python scripts/inference/goai/sim_server.py \
  --checkpoint lion-vla-ckpt/ema \
  --norm-stats lion-vla-ckpt/norm_stats_pt.json \
  --task-name stack_bowls \
  --config-name pi05_goai_joint \
  --host 0.0.0.0 --port 28000 \
  --device cuda:0 --norm-mode per-timestamp --apply-delta --seed 0

curl http://127.0.0.1:28000/healthz
```

## Checkpoints

Checkpoint weights, normalization statistics and the manifest are **not** stored in this
repository. They are delivered separately and unpacked into `lion-vla-ckpt/`, preserving the
relative layout:

```
lion-vla-ckpt/
├── ema/                  # DCP shards + .metadata
└── norm_stats_pt.json    # per-timestamp action normalization
```

The tokenizer is bundled at `assets/paligemma_tokenizer.model` so inference works offline.

## Repository layout

```
.
├── src/pi/
│   ├── inference/           # GOAI inference chain: protocol codec, observation
│   │                        # preprocessing, policy, normalization, tracing, resampling
│   ├── training/            # Training configuration and dataset plumbing
│   ├── models/              # PaliGemma / Gemma / SigLIP transformer fork
│   └── models_pytorch/      # Pi0 / Pi0.5 PyTorch implementation
├── scripts/
│   ├── inference/goai/      # Serving entry points, validators, smoke-test client
│   └── train/               # FSDP training and normalization-statistics tooling
├── policy/lionvla/          # XPolicyLab policy plugin (__init__.py + deploy.py)
├── configs/goai_real/       # Server configuration (global defaults + per-task overrides)
├── tools/trace_viewer.py    # Read-only browser for recorded inference traces
├── assets/                  # Bundled tokenizer
├── docs/                    # Topic documentation (bilingual: EN + zh-CN)
├── docs-GOAI/               # GOAI 2026 submission documents
└── pixi.toml / pixi.lock    # Locked environment definition
```

## Protocol

Implements **XPolicyLab v1.0.0** (with backward compatibility for the legacy `infer` call):

```
hello → prepare_case → reset → call(update_obs) → call(get_action) → trial_end
```

- `get_action` returns an absolute joint-position chunk of `execution_horizon` steps × 14
  dimensions (arm targets are deltas added to the current state; gripper openings are clipped
  to `[0, 1]`).
- Each connection owns an independent session (isolated RNG and task binding). Concurrent
  environments are expected to use one connection each; the server does not merge them into a
  batched forward pass.
- `action_case_id` follows `<task>_case` / `<task>_random_case`.

## Results

| Benchmark | Result |
|---|---|
| RoboDojo Generalization — local native evaluation (24 configs × 3 seeds × 25 episodes) | **Score 25.11 / SR 18.1%**, 0 failures |

The simulation numbers above do not measure physical success rate on the real robot. See
[`docs-GOAI/technical_solution.md`](docs-GOAI/technical_solution.md) for the full configuration
and per-task breakdown.

## Documentation

| | |
|---|---|
| [`docs/`](docs/) | Architecture, training, datasets, deployment, porting — English and 简体中文 |
| [`docs-GOAI/`](docs-GOAI/) | Competition submission: data sources, project introduction, technical solution, evaluation walkthrough |

## Acknowledgements

Built on [`lios-pi`](https://github.com/cmriat/LiOS/tree/main/lios-pi) from the
[LiOS](https://github.com/cmriat/LiOS) stack, which in turn adapts
[openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence). See
[`NOTICE`](NOTICE) for the full list.

## License

[Apache License 2.0](LICENSE).
