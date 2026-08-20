#!/usr/bin/env bash
# RoboDojo policy-directory eval entry: launches the GOAI policy server and
# leaves it running for the caller (see setup_eval_policy_server.sh).
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_env=$9
eval_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
policy_server_host=localhost
policy_server_port="${POLICY_SERVER_PORT:-28000}"

echo "[eval.sh] launching GOAI policy server on ${policy_server_host}:${policy_server_port}"
exec bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
    "${seed}" "${policy_gpu_id}" "${policy_env}" "${policy_server_port}" "${policy_server_host}"
