#!/usr/bin/env bash
# Explicit XPolicyLab config interface; see README.md for official integration.
set -euo pipefail
POLICY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(cd -- "$POLICY_ROOT/../.." && pwd -P)
exec bash "$ROOT/scripts/serve_goai.sh" "$@"
