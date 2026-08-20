# pi05_goai policy directory (RoboDojo convention)

Self-contained policy-directory interface for the GOAI 2026 bimanual track.
It launches the bundled GOAI policy server from THIS repository — no external
Pi repo required.

## Files

| File | Role |
|--|--|
| `setup_eval_policy_server.sh` | Start the WS policy server (v1.0.0 CALL + legacy infer) |
| `eval.sh` | RoboDojo policy-directory eval entry (delegates to the setup script) |
| `deploy.py` | Episode execution loop (update_obs → get_action → take_action) |
| `deploy.yml` | Policy metadata (name / action_type / embodiment) |

## Start

```bash
bash policy/pi05_goai/setup_eval_policy_server.sh \
  RoboDojo stack_bowls \
  lion-vla-ckpt/ema \
  arx_x5 joint 0 0 pi05-goai 28000 0.0.0.0
```

Defaults are the frozen submission config: action-horizon 8 / num-steps 20 /
apply-delta (per-timestamp norm). See `scripts/inference/goai/README.md` for
the protocol and checkpoint layout.
