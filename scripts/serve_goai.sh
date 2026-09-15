#!/usr/bin/env bash
# The default environment contains only the locked inference runtime.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT"
exec "${PIXI_BIN:-pixi}" run --manifest-path "$ROOT/pixi.toml" serve_goai "$@"
