#!/usr/bin/env python3
"""GPU acceptance for supervised startup, six-task serving and clean shutdown."""

from __future__ import annotations

import sys
import json
import time
import socket
import argparse
import subprocess
from pathlib import Path

import yaml
import numpy as np
from validate_xpolicylab import compare, make_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    bench = args.robodojo.expanduser().resolve()
    sys.path[:0] = [str(root / "src"), str(bench), str(bench / "XPolicyLab")]
    from client_server.ws.model_client import WsModelClient

    from pi.shared.goai_tasks import GOAI_REAL_TASK_INSTRUCTIONS, GOAI_REAL_LEGACY_TASK_INSTRUCTIONS

    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())
    # 配置键从 postprocess.gripper 搬到了顶层 gripper: + per_task.<任务名>.gripper。
    # 两处都显式查,不能靠 .get 默认值——搬完之后旧键取不到会静默通过。
    if (cfg.get("gripper") or {}).get("enabled", False) or any(
        (block.get("gripper") or {}).get("enabled", False)
        for block in (cfg.get("per_task") or {}).values()
    ):
        raise ValueError("Startup acceptance requires the gripper correction disabled")
    cfg.pop("task_name", None)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    cfg.update(host="127.0.0.1", port=port)
    config_path = out / "server.json"
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    log_path = out / "server.log"
    results = []
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(root / "scripts/inference/goai/serve_xpolicylab.py"),
                "--robodojo",
                str(bench),
                "--config",
                str(config_path),
                "--timeout",
                "900",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=root,
        )
        try:
            deadline = time.monotonic() + 1200
            while "READY FOR INFERENCE" not in log_path.read_text():
                if process.poll() is not None:
                    raise RuntimeError(f"Supervisor exited with {process.returncode}; see {log_path}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Supervisor did not become ready; see {log_path}")
                time.sleep(1)
            with WsModelClient(
                url=f"ws://127.0.0.1:{port}",
                evaluation_id="startup-acceptance",
                trial_id="synthetic",
                action_case_id="synthetic",
                request_timeout_s=900,
            ) as client:
                # Warmup reset must have removed its last observation.
                try:
                    client.call("get_action")
                except Exception as error:
                    if "exactly one latest observation" not in str(error):
                        raise
                else:
                    raise AssertionError("Warmup left an actionable observation behind")
                # Exercise official full sentences against the real server and weights.
                for slot, instruction in enumerate(GOAI_REAL_TASK_INSTRUCTIONS):
                    client.call("reset")
                    client.call("update_obs", make_observation(instruction))
                    began = time.monotonic()
                    actions = client.call("get_action")
                    from pi.inference.goai_xpolicylab import execution_horizon_for_task

                    expected = execution_horizon_for_task(cfg, instruction)
                    if len(actions) != expected:
                        raise AssertionError(f"Unexpected action horizon for {instruction!r}: {len(actions)} != {expected}")
                    for action in actions:
                        if set(action) != {
                            "left_arm_joint_state",
                            "left_ee_joint_state",
                            "right_arm_joint_state",
                            "right_ee_joint_state",
                        }:
                            raise AssertionError("Unexpected action keys")
                        for key, size in (
                            ("left_arm_joint_state", 6),
                            ("left_ee_joint_state", 1),
                            ("right_arm_joint_state", 6),
                            ("right_ee_joint_state", 1),
                        ):
                            value = np.asarray(action[key])
                            if value.shape != (size,) or not np.isfinite(value).all():
                                raise AssertionError(f"Invalid action {key}")
                    elapsed_ms = (time.monotonic() - began) * 1000
                    client.call("reset")
                    client.call("update_obs", make_observation(GOAI_REAL_LEGACY_TASK_INSTRUCTIONS[slot]))
                    compare(client.call("get_action"), actions)
                    results.append({"task": instruction, "slot": slot, "legacy_parity": True, "latency_ms": elapsed_ms})
                client.call("trial_end", {"synthetic": True})
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("Supervisor failed to terminate") from None
    with socket.socket() as check:
        check.settimeout(2)
        if check.connect_ex(("127.0.0.1", port)) == 0:
            raise AssertionError("Supervisor left a listening server after shutdown")
    result = {
        "status": "passed",
        "compile_mode": cfg.get("compile_mode", "none"),
        "warmup_reset": True,
        "shutdown": True,
        "tasks": results,
        "hardware": False,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
