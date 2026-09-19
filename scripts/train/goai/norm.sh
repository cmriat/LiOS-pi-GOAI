#!/bin/zsh

# Shared normalization-statistics entry point for GOAI VideoLance datasets.

set -euo pipefail

REPO_ROOT="${PI_REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
if [[ ! -f "$REPO_ROOT/pixi.toml" ]]; then
    echo "ERROR: repository not found: $REPO_ROOT" >&2
    exit 1
fi
cd "$REPO_ROOT"

CONFIG_NAME="${1:-${CONFIG_NAME:-}}"
DATASET_URI="${2:-${DATASET_URI:-}}"
STATS_DIR="${3:-${STATS_DIR:-}}"
NORM_MODE="${4:-${NORM_MODE:-global}}"
case "$CONFIG_NAME" in
    pi05_goai_joint|pi05_goai_ee|pi05_goai_joint_lance)
        ;;
    *)
        echo "Usage: zsh scripts/train/goai/norm.sh <pi05_goai_joint|pi05_goai_ee|pi05_goai_joint_lance> <dataset.lance> [stats_dir] [global|per_timestamp]" >&2
        exit 2
        ;;
esac
if [[ -z "$DATASET_URI" ]]; then
    echo "ERROR: DATASET_URI is required." >&2
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

# Stats output location is derived from the first dataset URI.
DATASET_BASENAME="${DATASET_URIS[1]##*/}"
DATASET_NAME="${DATASET_BASENAME%.lance}"
DATASET_PARENT="${DATASET_URIS[1]%/*}"
if [[ -z "$STATS_DIR" && ${#DATASET_URIS[@]} -gt 1 ]]; then
    echo "ERROR: multiple datasets require an explicit STATS_DIR for joint norm stats." >&2
    echo "Refusing to write joint stats into the first dataset's stats directory." >&2
    exit 2
fi
STATS_DIR="${STATS_DIR:-$DATASET_PARENT/$DATASET_NAME}"
if [[ "$STATS_DIR" != /* ]]; then
    STATS_DIR="$REPO_ROOT/$STATS_DIR"
fi
ACTION_HORIZON="${ACTION_HORIZON:-32}"
POLICY_STATE_SCHEMA="${POLICY_STATE_SCHEMA:-source}"
OVERWRITE_NORM_STATS="${OVERWRITE_NORM_STATS:-0}"
DRY_RUN="${DRY_RUN:-0}"
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.9}"

case "$NORM_MODE" in
    global)
        OUTPUT_PATH="$STATS_DIR/norm_stats.json"
        NORM_MODE_ARGS=(--data.no-use-per-timestamp-action-norm)
        ;;
    per_timestamp)
        OUTPUT_PATH="$STATS_DIR/norm_stats_pt.json"
        NORM_MODE_ARGS=(--data.use-per-timestamp-action-norm)
        ;;
    *)
        echo "ERROR: NORM_MODE must be global or per_timestamp, got: $NORM_MODE" >&2
        exit 2
        ;;
esac

case "$OVERWRITE_NORM_STATS" in
    1|true|TRUE|yes|YES)
        ;;
    0|false|FALSE|no|NO)
        if [[ -e "$OUTPUT_PATH" ]]; then
            echo "ERROR: norm stats already exist: $OUTPUT_PATH" >&2
            echo "Set OVERWRITE_NORM_STATS=1 to replace them." >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: OVERWRITE_NORM_STATS must be 0/1 or false/true." >&2
        exit 2
        ;;
esac

echo "config:          $CONFIG_NAME"
echo "dataset:         $DATASET_URI"
echo "output:          $OUTPUT_PATH"
echo "norm mode:       $NORM_MODE"
echo "action horizon:  $ACTION_HORIZON"
echo "policy state:    $POLICY_STATE_SCHEMA"

if [[ "$DRY_RUN" == "1" ]]; then
    print -r -- "DRY RUN: norm command"
    printf "%q " pixi run -e dev python scripts/train/compute_norm_stats_fast.py \
        "$CONFIG_NAME" \
        --model.action-horizon "$ACTION_HORIZON" \
        --data.policy-state-schema "$POLICY_STATE_SCHEMA" \
        --data.dataset-format video_lance \
        --data.repo-id "$DATASET_NAME" \
        --data.dataset-uri "$DATASET_URI" \
        --data.asset-id '<temporary-stats-dir>' \
        "$NORM_MODE_ARGS[@]"
    printf "\n"
    exit 0
fi

mkdir -p "$STATS_DIR"
RUNTIME_STATS_DIR="$(mktemp -d "$STATS_DIR/.norm_stats.XXXXXX")"
cleanup() {
    rm -rf "$RUNTIME_STATS_DIR"
}
trap cleanup EXIT

pixi run -e dev python scripts/train/compute_norm_stats_fast.py \
    "$CONFIG_NAME" \
    --model.action-horizon "$ACTION_HORIZON" \
    --data.policy-state-schema "$POLICY_STATE_SCHEMA" \
    --data.dataset-format video_lance \
    --data.repo-id "$DATASET_NAME" \
    --data.dataset-uri "$DATASET_URI" \
    --data.asset-id "$RUNTIME_STATS_DIR" \
    "$NORM_MODE_ARGS[@]"

GENERATED_STATS="$RUNTIME_STATS_DIR/norm_stats.json"
if [[ ! -f "$GENERATED_STATS" ]]; then
    echo "ERROR: norm_stats.json was not generated: $RUNTIME_STATS_DIR" >&2
    exit 1
fi
if [[ "$NORM_MODE" == "per_timestamp" ]]; then
    for field in per_timestamp_mean per_timestamp_std per_timestamp_q01 per_timestamp_q99; do
        if ! grep -Eq "\"$field\"[[:space:]]*:[[:space:]]*\[" "$GENERATED_STATS"; then
            echo "ERROR: generated per-timestamp stats are missing $field." >&2
            exit 1
        fi
    done
fi

mv -f -- "$GENERATED_STATS" "$OUTPUT_PATH"
echo "output: $OUTPUT_PATH"
