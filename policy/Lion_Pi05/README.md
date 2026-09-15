# Lion_Pi05 real-robot policy

This plugin serves the six trained GOAI tasks on PiperX using the existing
Pi05 task and embodiment embeddings. It runs in the unchanged official
XPolicyLab server. The preliminary simulation entry remains `policy/pi05_goai`.

## Install and configure

From this repository root, on Linux with a compatible NVIDIA driver:

```bash
pixi install --locked
# Set checkpoint_path in configs/goai_real/server.yaml to your checkpoint.
pixi run serve_goai
```

The default environment is the locked inference runtime. The legacy development
stack is optional (`-e dev`); it is not installed by this serving workflow.
Like the Pi server, the launcher defaults to `configs/goai_real/server.yaml`
and discovers an adjacent `XPolicyLab/`. It also accepts `../RoboDojo/XPolicyLab`.
For a different checkout or a private local config:

```bash
export GOAI_ROBODOJO=/path/to/RoboDojo
pixi run python scripts/inference/goai/prepare_xpolicylab.py \
  --robodojo "$GOAI_ROBODOJO" --checkpoint /path/to/step/ema \
  --output configs/goai_real/server.local.json
GOAI_SERVER_CONFIG="$PWD/configs/goai_real/server.local.json" pixi run serve_goai
```

`bash scripts/serve_goai.sh` and the installed policy shell entry delegate to
this same task. `GOAI_SERVER_TIMEOUT` (default 300 seconds) and
`GOAI_WARMUP_ROUNDS` (default 5) retain their Pi launcher meanings; explicit
`--timeout` / `--warmup-rounds` take precedence. No client checkout, isolated
`GOAI_SERVER_DEPS` directory or `GOAI_SERVER_PYTHON` override is required:
all server dependencies are locked in this repository.

`$GOAI_ROBODOJO/XPolicyLab` must contain the official framework, including
`setup_policy_server.py`. Debug acceptance also uses the official sibling
`env_cfg/` directory from RoboDojo. The integration baseline is XPolicyLab commit
`432f82b1758c5b1202e42a3dfe014546dbc50871`. No private training repository or
robot client package is needed. Weights are delivered separately from Git:

```text
step/
  norm_stats_pt.json
  norm_stats_manifest.json
  ema/
    .metadata
    <DCP shard files>
```

The manifest describes the architecture and hashes the exact statistics file.
`scripts/inference/goai/sanitize_manifest.py` removes training paths while
preserving inference fields. Do not substitute another checkpoint's statistics.

Preparation installs `XPolicyLab/policy/Lion_Pi05` as a symlink. It refuses to
overwrite a different plugin or configuration. Local configurations are ignored
by Git. `configs/goai_real/server.yaml` is a template with a placeholder path.

One model serves all six tasks selected by client observations. Preparation's
optional `--task` adds a fallback for missing task fields. Defaults: eight
actions, 20 denoising steps, embodiment 0, `reduce-overhead` compilation,
gripper postprocessing disabled. Runtime options live in the local config.

The supervisor uses the official `WsModelClient` for handshake, synthetic
warmup and reset before printing `READY FOR INFERENCE`. New tasks or physical
observations may still trigger compilation. Ctrl-C terminates the supervised
server. `--timeout` and `--warmup-rounds` are forwarded by the shell launcher.

For official orchestration, use `policy_name: Lion_Pi05` and the same config:

```bash
pixi run -e goai-inference python scripts/inference/goai/launch_xpolicylab.py \
  --robodojo "$GOAI_ROBODOJO" --config "$GOAI_SERVER_CONFIG"
```

`deploy.py` re-exports the official Pi_05 episode callbacks. The included
`setup_eval_policy_server.sh` uses this explicit configuration interface,
not the preliminary simulator's ten positional arguments. Hardware orchestration still requires validation on the supplied evaluation machine.

## Acceptance

After preparation, run CPU boundary and JPEG fixture checks:

```bash
PYTHONPATH="$PWD/src:$PWD:$GOAI_ROBODOJO:$GOAI_ROBODOJO/XPolicyLab" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pixi run -e goai-inference python -m pytest -q \
  tests/test_goai_xpolicylab.py tests/test_goai_debug_client.py
```

GPU checks require a prepared JSON config with a `task_name`,
`compile_mode: none` and disabled postprocessing:

```bash
pixi run -e goai-inference python scripts/inference/goai/validate_xpolicylab.py \
  --robodojo "$GOAI_ROBODOJO" --config /path/to/acceptance.json
pixi run -e goai-inference python scripts/inference/goai/validate_xpolicylab_debug.py \
  --robodojo "$GOAI_ROBODOJO" --config /path/to/acceptance.json \
  --output .tmp/debug-acceptance
```

The official debug wrapper replaces only the placeholder instruction, exercising
raw/JPEG and single/batch paths. These tests do not measure physical success.
See [INTERFACE.md](INTERFACE.md) for the policy boundary.

To verify supervised warmup, all six official full instructions, action parity
with their legacy short aliases after reset, and shutdown with the configured compilation mode:

```bash
pixi run -e goai-inference python scripts/inference/goai/validate_xpolicylab_startup.py \
  --robodojo "$GOAI_ROBODOJO" --config "$GOAI_SERVER_CONFIG" \
  --output .tmp/startup-acceptance
```

## External real-client acceptance

An optional hardware-free check can run in the real client's environment
against this server. Only the client's policy/codec modules are imported;
CAN, cameras and deployment controllers are never started:

```bash
pixi run --manifest-path /path/to/client/pixi.toml python \
  /path/to/LiOS-pi-GOAI/scripts/inference/goai/validate_real_client.py \
  --client-root /path/to/client --url ws://127.0.0.1:6000 \
  --output /path/to/local/client-acceptance.json
```

This exercises hello, prepare_case, reset, both inference APIs, trial_end,
heartbeat, JPEG, batching and reconnect using the external client implementation
with synthetic observations in the server contract. It does not exercise the
client camera encoder, gripper conversion or playback controller.
It supplements the official client checks; it does not establish hardware task
success or replace organizer acceptance on the supplied inference machine.


Add `--deployment-boundary` to also run the external client's actual camera JPEG
encoder, observation builder and action-unit converter with synthetic frames,
feedback and calibration. This mode imports deployment modules (including their
YAML dependency), but never constructs a hardware runtime or sends CAN commands.
It verifies RGB channel order and normalized-opening conversion independently of
transport. Physical calibration and task success still require hardware testing.
