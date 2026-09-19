#!/bin/zsh

# Shared Pi05 FSDP launcher for GOAI VideoLance datasets.

set -euo pipefail

REPO_ROOT="${PI_REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
if [[ ! -f "$REPO_ROOT/pixi.toml" ]]; then
    echo "ERROR: repository not found: $REPO_ROOT. Set PI_REPO_ROOT." >&2
    exit 1
fi
cd "$REPO_ROOT"

CONFIG_NAME="${1:-${CONFIG_NAME:-}}"
EXP_NAME="${2:-${EXP_NAME:-}}"
NUM_GPUS="${3:-${NUM_GPUS:-8}}"
DATASET_URI="${4:-${DATASET_URI:-}}"
case "$CONFIG_NAME" in
    pi05_goai_joint|pi05_goai_ee|pi05_goai_joint_lance)
        ;;
    *)
        echo "Usage: zsh scripts/train/goai/train.sh <pi05_goai_joint|pi05_goai_ee|pi05_goai_joint_lance> <exp_name> <gpus_per_node> <dataset.lance>" >&2
        echo "Or set CONFIG_NAME, EXP_NAME, NUM_GPUS, and DATASET_URI." >&2
        exit 2
        ;;
esac
if [[ -z "$EXP_NAME" || -z "$DATASET_URI" ]]; then
    echo "ERROR: EXP_NAME and DATASET_URI are required." >&2
    exit 2
fi

# Multi-URI support: normalize comma-separated values and validate every local URI.
DATASET_URIS=()
for raw_uri in "${(@s:,:)DATASET_URI}"; do
    uri="${raw_uri#"${raw_uri%%[![:space:]]*}"}"
    uri="${uri%"${uri##*[![:space:]]}"}"
    if [[ -z "$uri" ]]; then
        continue
    fi
    if [[ "$uri" != *://* ]]; then
        uri="${uri:A}"
        if [[ ! -d "$uri" ]]; then
            echo "ERROR: Lance dataset not found: $uri" >&2
            exit 1
        fi
    fi
    DATASET_URIS+=("$uri")
done
if (( ${#DATASET_URIS[@]} == 0 )); then
    echo "ERROR: DATASET_URI does not contain any dataset URI." >&2
    exit 2
fi
DATASET_URI="${(j:,:)DATASET_URIS}"

# Stats dir default derives from the first dataset URI.
DATASET_BASENAME="${DATASET_URIS[1]##*/}"
DATASET_NAME="${DATASET_BASENAME%.lance}"
DATASET_PARENT="${DATASET_URIS[1]%/*}"
if [[ -z "${STATS_DIR:-}" && -z "${NORM_STATS_PATH:-}" && ${#DATASET_URIS[@]} -gt 1 ]]; then
    echo "ERROR: multiple datasets require an explicit STATS_DIR or NORM_STATS_PATH" >&2
    echo "pointing at joint norm stats. Refusing to use single-dataset stats." >&2
    exit 2
fi
STATS_DIR="${STATS_DIR:-$DATASET_PARENT/$DATASET_NAME}"
NORM_MODE="${NORM_MODE:-}"
if [[ -z "$NORM_MODE" ]]; then
    case "${USE_PER_TIMESTAMP_ACTION_NORM:-1}" in
        1|true|TRUE|yes|YES) NORM_MODE=per_timestamp ;;
        0|false|FALSE|no|NO) NORM_MODE=global ;;
        *)
            echo "ERROR: USE_PER_TIMESTAMP_ACTION_NORM must be 0/1 or false/true." >&2
            exit 2
            ;;
    esac
fi
case "$NORM_MODE" in
    global)
        # 推理侧（冻结）硬编码读 step*/norm_stats_pt.json，goai_xpolicylab 又钉死 per-timestamp，
        # 所以 global 模式训出来的 checkpoint 会被**自家推理拒收**。宁可在这里失败，
        # 也不要产出一个本仓库加载不了的 checkpoint。
        echo "ERROR: NORM_MODE=global is not supported by this repository." >&2
        echo "       The inference side reads step*/norm_stats_pt.json (per-timestamp) only," >&2
        echo "       so a globally-normalized checkpoint cannot be loaded. Use per_timestamp." >&2
        exit 2
        ;;
    per_timestamp)
        NORM_MODE_ARGS=(--data.use-per-timestamp-action-norm)
        DEFAULT_STATS_PATH="$STATS_DIR/norm_stats_pt.json"
        ;;
    *)
        echo "ERROR: NORM_MODE must be global or per_timestamp, got: $NORM_MODE" >&2
        exit 2
        ;;
esac

NORM_STATS_PATH="${NORM_STATS_PATH:-$DEFAULT_STATS_PATH}"
NORM_STATS_PATH="${NORM_STATS_PATH:A}"
if [[ ! -f "$NORM_STATS_PATH" ]]; then
    echo "ERROR: norm stats not found: $NORM_STATS_PATH" >&2
    echo "Run scripts/train/goai/norm.sh with the same CONFIG_NAME, DATASET_URI, and NORM_MODE." >&2
    exit 1
fi

RUNTIME_STATS_DIR="$(mktemp -d /tmp/goai_norm_stats.XXXXXX)"
cp -- "$NORM_STATS_PATH" "$RUNTIME_STATS_DIR/norm_stats.json"
export PI_NORM_STATS_SOURCE_PATH="$NORM_STATS_PATH"

NNODES="${SLURM_NNODES:-1}"
NODE_RANK=$(( ${SLURM_PROCID:-${JOB_COMPLETION_INDEX:-0}} ))
MASTER_ADDR="${SLURM_JOB_FIRST_NODE_IP:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
TOTAL_GPUS=$((NNODES * NUM_GPUS))
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-24}"
BATCH_SIZE="${BATCH_SIZE:-$((TOTAL_GPUS * PER_GPU_BATCH_SIZE))}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
STATE_HISTORY_FRAMES="${STATE_HISTORY_FRAMES:-1}"
STATE_DELAY_FRAMES="${STATE_DELAY_FRAMES:-0}"
POLICY_STATE_SCHEMA="${POLICY_STATE_SCHEMA:-source}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
TEST_EP_NUM="${TEST_EP_NUM:-0}"
NUM_TASKS="${NUM_TASKS:-12}"
TASK_CONDITIONING="${TASK_CONDITIONING:-language}"
TASK_EMBEDDING_TARGET="${TASK_EMBEDDING_TARGET:-vlm}"
STATE_CONDITIONING_MODE="${STATE_CONDITIONING_MODE:-discrete_vlm}"
USE_EMBODIMENT_EMBEDDING="${USE_EMBODIMENT_EMBEDDING:-0}"
NUM_EMBODIMENTS="${NUM_EMBODIMENTS:-2}"
NUM_EMBODIMENT_TOKENS="${NUM_EMBODIMENT_TOKENS:-1}"
SAVE_STEP_INTERVAL="${SAVE_STEP_INTERVAL:-5000}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-./checkpoints}"
PROJECT_NAME="${PROJECT_NAME:-$CONFIG_NAME}"
RUN_MODE="${RUN_MODE:-overwrite}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
PEAK_LR="${PEAK_LR:-3e-5}"
DECAY_STEPS="${DECAY_STEPS:-30000}"
DECAY_LR="${DECAY_LR:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.9}"

case "$TASK_CONDITIONING" in
    language)
        CONDITIONING_ARGS=(--model.no-use-task-embedding --model.no-use-language-with-task-embedding)
        ;;
    task_embedding)
        CONDITIONING_ARGS=(--model.use-task-embedding --model.no-use-language-with-task-embedding)
        ;;
    language_task_embedding)
        CONDITIONING_ARGS=(--model.use-task-embedding --model.use-language-with-task-embedding)
        ;;
    *)
        echo "ERROR: TASK_CONDITIONING must be language, task_embedding, or language_task_embedding." >&2
        exit 2
        ;;
esac

case "$TASK_EMBEDDING_TARGET" in
    vlm|expert) ;;
    *)
        echo "ERROR: TASK_EMBEDDING_TARGET must be vlm or expert." >&2
        exit 2
        ;;
esac
if [[ "$TASK_EMBEDDING_TARGET" == "expert" && "$TASK_CONDITIONING" == "language" ]]; then
    echo "ERROR: TASK_EMBEDDING_TARGET=expert requires task embedding conditioning." >&2
    exit 2
fi

case "$USE_EMBODIMENT_EMBEDDING" in
    1|true|TRUE|yes|YES)
        if [[ "$NUM_EMBODIMENTS" != <-> || "$NUM_EMBODIMENTS" -le 0 ]]; then
            echo "ERROR: NUM_EMBODIMENTS must be a positive integer." >&2
            exit 2
        fi
        if [[ "$NUM_EMBODIMENT_TOKENS" != <-> || "$NUM_EMBODIMENT_TOKENS" -le 0 ]]; then
            echo "ERROR: NUM_EMBODIMENT_TOKENS must be a positive integer." >&2
            exit 2
        fi
        EMBODIMENT_ARGS=(
            --model.use-embodiment-embedding
            --model.num-embodiments "$NUM_EMBODIMENTS"
            --model.num-embodiment-tokens "$NUM_EMBODIMENT_TOKENS"
        )
        ;;
    0|false|FALSE|no|NO)
        USE_EMBODIMENT_EMBEDDING=0
        EMBODIMENT_ARGS=()
        ;;
    *)
        echo "ERROR: USE_EMBODIMENT_EMBEDDING must be 0/1 or false/true." >&2
        exit 2
        ;;
esac

case "$STATE_CONDITIONING_MODE" in
    discrete_vlm) ;;
    dual)
        if [[ "$STATE_HISTORY_FRAMES" != "1" ]]; then
            echo "ERROR: STATE_CONDITIONING_MODE=dual currently requires STATE_HISTORY_FRAMES=1." >&2
            exit 2
        fi
        ;;
    *)
        echo "ERROR: STATE_CONDITIONING_MODE must be discrete_vlm or dual." >&2
        exit 2
        ;;
esac

case "$RUN_MODE" in
    overwrite) MODE_ARGS=(--overwrite) ;;
    resume) MODE_ARGS=(--no-overwrite --resume) ;;
    *)
        echo "ERROR: RUN_MODE must be overwrite or resume." >&2
        exit 2
        ;;
esac

# Env-gated EMA decay / weight decay; defaults match the recipe defaults.
# 提交方案 docs-GOAI/technical_solution.md 写定的 recipe：decay floor 0 / EMA 0.9995 / wd 0.05。
# 这三个默认值必须与文档一致 —— 否则归档进 manifest 的 training_recipe 与文档对不上，
# 续训时 _verify_training_recipe_on_resume 也会拒绝。
export PI_LR_DECAY_FLOOR_STEPS="${PI_LR_DECAY_FLOOR_STEPS:-0}"
EMA_DECAY="${EMA_DECAY:-0.9995}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WEIGHT_DECAY_ARGS=()
if [[ -n "$WEIGHT_DECAY" ]]; then
    WEIGHT_DECAY_ARGS=(--optimizer.weight-decay "$WEIGHT_DECAY")
fi

WEIGHT_PATH="${WEIGHT_PATH:-./data/pi05/pi05_pt}"
if [[ ! -f "$WEIGHT_PATH/model.safetensors" ]]; then
    echo "ERROR: pretrained weights not found: $WEIGHT_PATH/model.safetensors" >&2
    exit 1
fi

if ! pixi run -e dev python -c 'import lance; import video_lance' >/dev/null 2>&1; then
    echo "ERROR: lance/video_lance is unavailable. Install the VideoLance backend first." >&2
    exit 1
fi

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
mkdir -p "$LOG_DIR"
if (( NNODES > 1 )); then
    LOG_TAG="${EXP_NAME}-node${NODE_RANK}-${HOSTNAME:-node}"
else
    LOG_TAG="${SLURM_JOB_ID:-unknown}-${HOSTNAME:-node}-$EXP_NAME"
fi
STDOUT_LOG="$LOG_DIR/jobs-$LOG_TAG.out"
STDERR_LOG="$LOG_DIR/jobs-$LOG_TAG.err"

export TORCH_MULTIPROCESSING_START_METHOD=spawn
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-2}"
export NCCL_IB_TIME_OUT="${NCCL_IB_TIME_OUT:-22}"

if (( NNODES > 1 )); then
    JOB_TAG="${EXP_NAME}_${MASTER_ADDR}_${MASTER_PORT}"
    JOB_TAG="${JOB_TAG//[^A-Za-z0-9._-]/_}"
    export VIDEO_LANCE_CACHE_DIR="${VIDEO_LANCE_CACHE_DIR:-$HOME/.cache/video_lance/job_$JOB_TAG}"
else
    JOB_TAG="${EXP_NAME}_$$"
    export VIDEO_LANCE_CACHE_DIR="${VIDEO_LANCE_CACHE_DIR:-/tmp/video_lance_$JOB_TAG}"
fi
cleanup() {
    if (( NNODES == 1 || NODE_RANK == 0 )); then
        rm -rf "$VIDEO_LANCE_CACHE_DIR"
    fi
    rm -rf "$RUNTIME_STATS_DIR"
}
trap cleanup EXIT

if [[ "$DRY_RUN" != "1" ]]; then
    exec > >(tee -a "$STDOUT_LOG") 2> >(tee -a "$STDERR_LOG" >&2)
fi

echo "config:          $CONFIG_NAME"
echo "dataset:         $DATASET_URI"
echo "stats:           $NORM_STATS_PATH ($NORM_MODE)"
echo "policy state:    $POLICY_STATE_SCHEMA"
echo "weights:         $WEIGHT_PATH"
echo "checkpoints:     $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME"
echo "conditioning:    $TASK_CONDITIONING"
echo "task target:     $TASK_EMBEDDING_TARGET"
echo "state mode:      $STATE_CONDITIONING_MODE"
if (( ${#EMBODIMENT_ARGS[@]} > 0 )); then
    echo "embodiment:      enabled (count=$NUM_EMBODIMENTS tokens=$NUM_EMBODIMENT_TOKENS)"
fi
echo "horizon/epochs:  $ACTION_HORIZON / $NUM_EPOCHS"
echo "nodes/gpus/batch: $NNODES / $TOTAL_GPUS / $BATCH_SIZE"
echo "save interval:   $SAVE_STEP_INTERVAL"
echo "lr:              peak=$PEAK_LR decay=$DECAY_LR warmup=$WARMUP_STEPS decay_steps=$DECAY_STEPS"
echo "stdout log:      $STDOUT_LOG"
echo "stderr log:      $STDERR_LOG"

if (( NNODES > 1 )); then
    TORCHRUN_ARGS=(
        --nnodes "$NNODES"
        --nproc_per_node "$NUM_GPUS"
        --node_rank "$NODE_RANK"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
    )
else
    TORCHRUN_ARGS=(--standalone --nproc_per_node "$NUM_GPUS")
fi

TRAIN_ARGS=(
    scripts/train/train_pytorch_fsdp.py
    "$CONFIG_NAME"
    --shuffle
    --project-name "$PROJECT_NAME"
    --exp-name "$EXP_NAME"
    --pytorch-weight-path "$WEIGHT_PATH"
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
    --model.action-horizon "$ACTION_HORIZON"
    --data.policy-state-schema "$POLICY_STATE_SCHEMA"
    --model.num-tasks "$NUM_TASKS"
    "$CONDITIONING_ARGS[@]"
    --model.task-embedding-target "$TASK_EMBEDDING_TARGET"
    --model.state-conditioning-mode "$STATE_CONDITIONING_MODE"
    "$EMBODIMENT_ARGS[@]"
    --model.state-history-frames "$STATE_HISTORY_FRAMES"
    --model.state-delay-frames "$STATE_DELAY_FRAMES"
    --data.dataset-format video_lance
    --data.repo-id "$DATASET_NAME"
    --data.dataset-uri "$DATASET_URI"
    --data.asset-id "$RUNTIME_STATS_DIR"
    --data.test-ep-num "$TEST_EP_NUM"
    "$NORM_MODE_ARGS[@]"
    --num-epochs "$NUM_EPOCHS"
    --test-step-interval None
    --save-step-interval "$SAVE_STEP_INTERVAL"
    --save-epoch-interval None
    --num-workers "$NUM_WORKERS"
    --batch-size "$BATCH_SIZE"
    --lr-schedule.warmup-steps "$WARMUP_STEPS"
    --lr-schedule.peak-lr "$PEAK_LR"
    --lr-schedule.decay-steps "$DECAY_STEPS"
    --lr-schedule.decay-lr "$DECAY_LR"
    --optimizer.clip-gradient-norm 1.0
    "${WEIGHT_DECAY_ARGS[@]}"
    --ema-decay "$EMA_DECAY"
    "$MODE_ARGS[@]"
)

if [[ "$DRY_RUN" == "1" ]]; then
    print -r -- "DRY RUN: training command"
    printf "%q " pixi run -e dev torchrun "$TORCHRUN_ARGS[@]" "$TRAIN_ARGS[@]"
    printf "\n"
    exit 0
fi

pixi run -e dev torchrun "$TORCHRUN_ARGS[@]" "$TRAIN_ARGS[@]"
