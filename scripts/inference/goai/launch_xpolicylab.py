#!/usr/bin/env python3
"""Launch the unchanged official XPolicyLab server in the Pi Python environment."""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deps", type=Path, help="Optional isolated official-server dependencies")
    args = parser.parse_args()
    pi_root = Path(__file__).resolve().parents[3]
    bench = args.robodojo.expanduser().resolve()
    xpl = bench / "XPolicyLab"
    config = args.config.expanduser().resolve()
    server = xpl / "setup_policy_server.py"
    if not server.is_file() or not config.is_file() or not (xpl / "policy/lionvla/model.py").is_file():
        parser.error("Missing official server, config or policy entry; run prepare_xpolicylab.py first")
    env = dict(os.environ)
    paths = [str(pi_root / "src"), str(bench), str(xpl)]
    if args.deps is not None:
        paths.insert(0, str(args.deps.expanduser().resolve()))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONUNBUFFERED"] = "1"
    os.chdir(pi_root)
    os.execve(sys.executable, [sys.executable, str(server), "--config_path", str(config)], env)


if __name__ == "__main__":
    main()
