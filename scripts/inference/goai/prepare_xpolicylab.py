#!/usr/bin/env python3
"""Install the Lion_Pi05 plugin and generate a portable real-policy configuration."""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pi.inference.goai_xpolicylab import resolve_real_task, validate_checkpoint


def install_policy(bench):
    xpl = Path(bench).expanduser().resolve() / "XPolicyLab"
    if not (xpl / "setup_policy_server.py").is_file():
        raise FileNotFoundError("Expected the official XPolicyLab checkout inside --robodojo")
    source = ROOT / "policy/Lion_Pi05"
    destination = xpl / "policy/Lion_Pi05"
    if destination.is_symlink() and destination.resolve() == source:
        return destination
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace existing policy: {destination}")
    destination.symlink_to(source, target_is_directory=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Local JSON config (also valid YAML)")
    parser.add_argument("--task", help="Optional fallback; normally the client selects the task")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument(
        "--compile-mode", choices=("none", "default", "reduce-overhead", "max-autotune"), default="reduce-overhead"
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.execution_horizon <= 32 or args.num_steps < 1:
        parser.error("Invalid port, execution horizon or denoising steps")
    checkpoint, _ = validate_checkpoint(args.checkpoint)
    config = dict(
        policy_name="Lion_Pi05",
        protocol="ws",
        host=args.host,
        port=args.port,
        action_type="joint",
        seed=0,
        checkpoint_path=str(checkpoint),
        device="cuda:0",
        embodiment_index=0,
        execution_horizon=args.execution_horizon,
        num_steps=args.num_steps,
        compile_mode=args.compile_mode,
        eval_batch=True,
        postprocess={"enabled": False},
    )
    if args.task:
        resolve_real_task(args.task)
        config["task_name"] = args.task
    output = args.output.expanduser().resolve()
    if output.exists() and json.loads(output.read_text()) != config:
        raise FileExistsError(f"Refusing to replace different config: {output}")
    destination = install_policy(args.robodojo)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"policy": str(destination), "config": str(output)}, indent=2))


if __name__ == "__main__":
    main()
