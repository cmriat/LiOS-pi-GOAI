#!/usr/bin/env bash
# GOAI policy-server launcher compatible with the RoboDojo policy-directory
# convention. Self-contained: serves from THIS repository (lios-pi) via the
# bundled goai-inference pixi environment; no external Pi repo required.
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIXI_BIN="${PIXI_BIN:-${HOME}/.pixi/bin/pixi}"

if [[ "${action_type}" != "joint" ]]; then
    echo "[SERVER][ERROR] pi05_goai supports only joint actions, got ${action_type}" >&2
    exit 2
fi

if [[ "${ckpt_name}" == /* ]]; then
    checkpoint="${ckpt_name}"
elif [[ -d "${REPO_ROOT}/checkpoints/${ckpt_name}" ]]; then
    checkpoint="${REPO_ROOT}/checkpoints/${ckpt_name}"
else
    echo "[SERVER][ERROR] checkpoint not found: ${ckpt_name}" >&2
    echo "[SERVER][ERROR] checked ${REPO_ROOT}/checkpoints/${ckpt_name}" >&2
    exit 1
fi
checkpoint="$(cd "${checkpoint}" && pwd -P)"

stats_root="${checkpoint}"
if [[ "$(basename "${checkpoint}")" == "ema" ]]; then
    stats_root="$(dirname "${checkpoint}")"
fi
norm_stats="${PI05_NORM_STATS:-${stats_root}/norm_stats_pt.json}"
if [[ ! -f "${checkpoint}/.metadata" ]]; then
    echo "[SERVER][ERROR] DCP metadata not found: ${checkpoint}/.metadata" >&2
    exit 1
fi
if [[ ! -f "${norm_stats}" ]]; then
    echo "[SERVER][ERROR] norm stats not found: ${norm_stats}" >&2
    exit 1
fi

echo "[SERVER] policy=pi05_goai task=${task_name} port=${policy_server_port} gpu=${policy_gpu_id}"
echo "[SERVER] checkpoint=${checkpoint}"

cd "${REPO_ROOT}"
exec env \
    -u PIXI_PROJECT_MANIFEST \
    -u PIXI_PROJECT_ROOT \
    -u PIXI_ENVIRONMENT_NAME \
    CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-13.0}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONUNBUFFERED=1 \
    "${PIXI_BIN}" run -e goai-inference python scripts/inference/goai/sim_server.py \
        --checkpoint "${checkpoint}" \
        --norm-stats "${norm_stats}" \
        --task-name "${task_name}" \
        --config-name pi05_goai_joint \
        --host "${policy_server_host}" \
        --port "${policy_server_port}" \
        --device cuda:0 \
        --action-horizon "${PI05_ACTION_HORIZON:-8}" \
        --num-steps "${PI05_NUM_STEPS:-20}" \
        --compile-mode "${PI05_COMPILE_MODE:-none}" \
        --norm-mode per-timestamp \
        --apply-delta \
        --seed "${seed}" \
        "${PI05_WARMUP_ARGS[@]:-}"
