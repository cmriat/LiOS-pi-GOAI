#!/usr/bin/env python3
"""Exercise a real official server process with the official debug program."""

from __future__ import annotations

import os
import sys
import json
import time
import socket
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deps", type=Path, help="Optional isolated dependencies")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    scripts = root / "scripts/inference/goai"
    bench = args.robodojo.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        ([str(args.deps.resolve())] if args.deps else [])
        + [str(root / "src"), str(root), str(bench), str(bench / "XPolicyLab")]
    )
    env["PYTHONUNBUFFERED"] = "1"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    config = json.loads(args.config.read_text())
    config.update(host="127.0.0.1", port=port)
    config_path = out / "server.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    results = []
    with (out / "server.log").open("w") as server_log:
        server = subprocess.Popen(
            [
                sys.executable,
                str(scripts / "launch_xpolicylab.py"),
                "--robodojo",
                str(bench),
                "--config",
                str(config_path),
                *(["--deps", str(args.deps.resolve())] if args.deps else []),
            ],
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited with {server.returncode}; see {out / 'server.log'}")
                try:
                    from websockets.sync.client import connect

                    with connect(f"ws://127.0.0.1:{port}", open_timeout=1, close_timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Server startup timed out") from None
                    time.sleep(1)
            # The unmodified mock must still reject its unknown task placeholder.
            raw_command = [
                sys.executable,
                str(bench / "XPolicyLab/debug_env_client.py"),
                "--bench_name",
                "goai",
                "--task_name",
                config["task_name"],
                "--env_cfg_type",
                "arx_x5",
                "--policy_name",
                "Lion_Pi05",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--eval_episode_num",
                "1",
            ]
            with (out / "unmodified-mock.log").open("w") as log:
                negative = subprocess.run(raw_command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            negative_log = (out / "unmodified-mock.log").read_text()
            if negative.returncode == 0 or "Untrained GOAI real task 'language instruction'" not in negative_log:
                raise AssertionError("Expected explicit rejection of the original mock placeholder")
            for batch in ("false", "true"):
                for encoded in ("false", "true"):
                    name = f"batch-{batch}_jpeg-{encoded}"
                    print(f"START {name}", flush=True)
                    with (out / f"{name}.log").open("w") as log:
                        subprocess.run(
                            [
                                sys.executable,
                                str(scripts / "debug_xpolicylab_client.py"),
                                "--robodojo",
                                str(bench),
                                "--port",
                                str(port),
                                "--task",
                                config["task_name"],
                                "--batch",
                                batch,
                                "--encoded",
                                encoded,
                                "--output",
                                str(out / f"{name}.json"),
                            ],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=900,
                            check=True,
                        )
                    results.append(json.loads((out / f"{name}.json").read_text()))
                    print(f"PASS {name}", flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    result = {
        "status": "passed",
        "checkpoint": config["checkpoint_path"],
        "unmodified_mock_placeholder_rejected": True,
        "matrix": results,
        "hardware": False,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
