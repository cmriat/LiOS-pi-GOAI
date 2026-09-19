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


def check_endpoint_available(host: str, port: int) -> None:
    """Check for a listener while allowing recently closed TCP connections."""
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as reservation:
        # Match the official asyncio server's POSIX reuse of TIME_WAIT addresses.
        reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reservation.bind((host, port))


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
    if cfg.get("policy_name") != "lionvla" or cfg.get("protocol", "ws") != "ws":
        parser.error("Expected lionvla with the official ws protocol")
    install_policy(bench)
    sys.path[:0] = [str(bench), str(bench / "XPolicyLab")]
    from websockets.sync.client import connect
    from client_server.ws.model_client import WsModelClient

    bind_host = cfg.get("host", "127.0.0.1")
    # Refuse an occupied endpoint before attempting any warmup calls.
    try:
        check_endpoint_available(bind_host, int(cfg["port"]))
    except OSError as exc:
        parser.error(f"Cannot bind policy server to {bind_host}:{cfg['port']}: {exc}")
    # `url` stays the loopback endpoint: it is what the readiness probe and the
    # synthetic warmup actually dial, and a NAT'd public address would not
    # necessarily hairpin back.
    host = bind_host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    if host == "::":
        host = "::1"
    host = f"[{host}]" if ":" in host else host
    url = f"ws://{host}:{cfg['port']}"
    # `display_url` is for the operator's eyes only: when bound to all
    # interfaces it names the address the on-site client must dial.
    display_host = cfg.get("public_host") if bind_host in ("0.0.0.0", "::") else bind_host
    display_host = str(display_host) if display_host else host.strip("[]")
    display_host = f"[{display_host}]" if ":" in display_host else display_host
    display_url = f"ws://{display_host}:{cfg['port']}"
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
        warmup_task = cfg.get("task_name") or "Insert the charger"
        obs = make_observation(warmup_task)
        # 预热观测同样会命中 per_task 覆盖,期望长度必须用同一套任务名归一化来解析,
        # 否则给预热任务(默认 Insert the charger)配了 execution_horizon 就会误报启动失败。
        from pi.inference.goai_xpolicylab import execution_horizon_for_task

        expected_horizon = execution_horizon_for_task(cfg, warmup_task)
        for _ in range(args.warmup_rounds):
            client.call("update_obs", obs)
            began = time.monotonic()
            actions = client.call("get_action")
            if len(actions) != expected_horizon:
                raise RuntimeError(
                    f"Warmup returned an unexpected action horizon: got {len(actions)}, want {expected_horizon}"
                )
            elapsed = (time.monotonic() - began) * 1000
        client.call("reset")
        client.close()
        client = None
        if server.poll() is not None:
            raise RuntimeError("Server exited during warmup")
        print(f"READY FOR INFERENCE {display_url}; synthetic warmup last call {elapsed:.1f} ms", flush=True)
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
