#!/usr/bin/env python3
"""Supervise the unchanged official server and warm it through its official client."""

from __future__ import annotations

import os
import sys
import time
import signal
import socket
import argparse
import subprocess
from pathlib import Path

import yaml
from prepare_xpolicylab import install_policy
from validate_xpolicylab import make_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    default_bench = root.parent if (root.parent / "XPolicyLab").is_dir() else root.parent / "RoboDojo"
    parser.add_argument("--robodojo", type=Path, default=os.environ.get("GOAI_ROBODOJO", default_bench))
    parser.add_argument(
        "--config", type=Path, default=os.environ.get("GOAI_SERVER_CONFIG", root / "configs/goai_real/server.yaml")
    )
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("GOAI_SERVER_TIMEOUT", "300")))
    parser.add_argument("--warmup-rounds", type=int, default=int(os.environ.get("GOAI_WARMUP_ROUNDS", "5")))
    args = parser.parse_args()
    if args.timeout <= 0 or args.warmup_rounds < 1:
        parser.error("Timeout and warmup rounds must be positive")
    root = Path(__file__).resolve().parents[3]
    bench = args.robodojo.expanduser().resolve()
    config = args.config.expanduser().resolve()
    cfg = yaml.safe_load(config.read_text())
    if cfg.get("policy_name") != "Lion_Pi05" or cfg.get("protocol", "ws") != "ws":
        parser.error("Expected Lion_Pi05 with the official ws protocol")
    install_policy(bench)
    sys.path[:0] = [str(bench), str(bench / "XPolicyLab")]
    from websockets.sync.client import connect
    from client_server.ws.model_client import WsModelClient

    bind_host = cfg.get("host", "127.0.0.1")
    # Refuse an occupied endpoint before attempting any warmup calls.
    with socket.socket(socket.AF_INET6 if ":" in bind_host else socket.AF_INET) as reservation:
        reservation.bind((bind_host, int(cfg["port"])))
    host = bind_host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    if host == "::":
        host = "::1"
    host = f"[{host}]" if ":" in host else host
    url = f"ws://{host}:{cfg['port']}"
    server = subprocess.Popen(
        [
            sys.executable,
            str(root / "scripts/inference/goai/launch_xpolicylab.py"),
            "--robodojo",
            str(bench),
            "--config",
            str(config),
        ],
        cwd=root,
    )
    client = None

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        deadline = time.monotonic() + args.timeout
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Official server exited before readiness: {server.returncode}")
            try:
                with connect(url, open_timeout=2, close_timeout=2):
                    pass
                break
            except (OSError, TimeoutError, ConnectionError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Official server readiness timed out") from None
                time.sleep(1)
        client = WsModelClient(
            url=url,
            evaluation_id="warmup",
            trial_id="warmup",
            action_case_id="warmup",
            request_timeout_s=args.timeout,
            connect_timeout_s=2,
            handshake_timeout_s=2,
            max_connect_attempts=1,
            max_connect_seconds=2,
            close_timeout_s=2,
        )
        client.call("reset")
        obs = make_observation(cfg.get("task_name") or "Insert the charger")
        for _ in range(args.warmup_rounds):
            client.call("update_obs", obs)
            began = time.monotonic()
            actions = client.call("get_action")
            if len(actions) != cfg.get("execution_horizon", 8):
                raise RuntimeError("Warmup returned an unexpected action horizon")
            elapsed = (time.monotonic() - began) * 1000
        client.call("reset")
        client.close()
        client = None
        if server.poll() is not None:
            raise RuntimeError("Server exited during warmup")
        print(f"READY FOR INFERENCE {url}; synthetic warmup last call {elapsed:.1f} ms", flush=True)
        print("Physical observations and each new task may incur additional compilation.", flush=True)
        raise SystemExit(server.wait())
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass  # Server termination must still run if client cleanup fails.
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
