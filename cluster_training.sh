#!/usr/bin/env bash
#SBATCH --job-name=openpi-tienkung-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# ==============================================================================
# OpenPI π0.5 Tienkung full fine-tuning on CHESS
#
# Checkpoint strategy:
#
#   /data-sl latest valid checkpoint
#       ↓ rsync at job startup when RESUME=1
#   node-local /tmp checkpoint directory
#       ↓ Orbax writes only to local filesystem
#   completed numeric checkpoint directory
#       ↓ background rsync to .sync-<step>-<job_id>
#   persistent /data-sl checkpoint directory
#
# Resume current experiment:
#
#   RESUME=1 \
#   EXP_NAME=0719_tienkung_pick_place_full \
#   sbatch cluster_training.sh
#
# Test local checkpoint synchronization:
#
#   RESUME=1 \
#   EXP_NAME=0719_tienkung_pick_place_full \
#   NUM_TRAIN_STEPS=3100 \
#   SAVE_INTERVAL=1000 \
#   KEEP_LOCAL_TMP=1 \
#   WANDB_DISABLED=1 \
#   sbatch cluster_training.sh
#
# Formal continuation:
#
#   RESUME=1 \
#   EXP_NAME=0719_tienkung_pick_place_full \
#   NUM_TRAIN_STEPS=20000 \
#   SAVE_INTERVAL=5000 \
#   sbatch cluster_training.sh
# ==============================================================================

# ------------------------------------------------------------------------------
# Persistent paths
# ------------------------------------------------------------------------------

BASE="${BASE:-/data-sl/lsy_ws}"

REPO="${REPO:-$BASE/project/openpi-0_5}"
SANDBOX="${SANDBOX:-$BASE/sandboxes/openpi-python311}"
VENV="${VENV:-$BASE/envs/openpi05}"

DATASET_ROOT="${HF_LEROBOT_HOME:-$BASE/dataset/VLA}"
DATASET_REPO_ID="${DATASET_REPO_ID:-pick_apple/0704}"
DATASET_DIR="$DATASET_ROOT/$DATASET_REPO_ID"

WEIGHTS_DIR="${WEIGHTS_DIR:-$BASE/models/pi05_base/params}"

ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$REPO/assets}"
PERSIST_CHECKPOINT_BASE="${PERSIST_CHECKPOINT_BASE:-$BASE/checkpoints}"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$BASE/cache/openpi}"
HF_HOME="${HF_HOME:-$BASE/cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$BASE/cache/uv}"

RUNTIME_HOME="${RUNTIME_HOME:-$BASE/runtime-home/openpi}"
WANDB_DIR="${WANDB_DIR:-$BASE/logs/wandb}"

PALIGEMMA_TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"

# ------------------------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------------------------

CONFIG_NAME="${CONFIG_NAME:-pi05_tienkung_pick_place}"
EXP_NAME="${EXP_NAME:-0719_tienkung_pick_place_full}"

FSDP_DEVICES="${FSDP_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"

# A full checkpoint is about 50 GiB. Saving every 5000 steps considerably
# reduces pressure on /data-sl.
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

# Protect every scheduled local checkpoint from Orbax max_to_keep deletion.
# This is intentional because the node-local /tmp has about 1.8 TiB available.
KEEP_PERIOD="${KEEP_PERIOD:-$SAVE_INTERVAL}"

RESUME="${RESUME:-1}"
WANDB_DISABLED="${WANDB_DISABLED:-0}"

# Keep local files after the job for debugging. Normally leave this at 0.
KEEP_LOCAL_TMP="${KEEP_LOCAL_TMP:-0}"

# Background checkpoint scan interval.
SYNC_INTERVAL="${SYNC_INTERVAL:-30}"

# Require at least this much node-local free space.
MIN_LOCAL_FREE_GIB="${MIN_LOCAL_FREE_GIB:-300}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"

# Do not inherit an accidental Hugging Face mirror.
if [[ -z "${USE_HF_ENDPOINT:-}" ]]; then
    unset HF_ENDPOINT || true
fi

# ------------------------------------------------------------------------------
# Derived paths
# ------------------------------------------------------------------------------

PERSIST_RUN="$PERSIST_CHECKPOINT_BASE/$CONFIG_NAME/$EXP_NAME"

JOB_TMP="${TMPDIR:-/tmp}/openpi-${USER}-${SLURM_JOB_ID:-$$}"
LOCAL_CHECKPOINT_BASE="$JOB_TMP/checkpoints"
LOCAL_RUN="$LOCAL_CHECKPOINT_BASE/$CONFIG_NAME/$EXP_NAME"

NORM_STATS_FILE="$ASSETS_BASE_DIR/$CONFIG_NAME/$DATASET_REPO_ID/norm_stats.json"

STOP_SYNC_FILE="$JOB_TMP/.stop-checkpoint-sync"

TRAIN_PID=""
SYNC_PID=""
RECEIVED_SIGNAL=0
FINAL_SYNC_OK=0

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_dir() {
    local path="$1"
    local description="$2"

    [[ -d "$path" ]] || die "$description not found: $path"
}

require_file() {
    local path="$1"
    local description="$2"

    [[ -f "$path" ]] || die "$description not found: $path"
}

directory_not_empty() {
    local path="$1"

    find "$path" -mindepth 1 -print -quit 2>/dev/null |
        grep -q .
}

checkpoint_is_complete() {
    local checkpoint="$1"

    [[ -d "$checkpoint" ]] &&
        [[ -f "$checkpoint/_CHECKPOINT_METADATA" ]] &&
        [[ -d "$checkpoint/params" ]] &&
        [[ -d "$checkpoint/train_state" ]]
}

list_numeric_checkpoints() {
    local run_dir="$1"

    [[ -d "$run_dir" ]] || return 0

    find "$run_dir" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -printf '%f\n' 2>/dev/null |
        awk '/^[0-9]+$/ {print}' |
        sort -n
}

latest_complete_checkpoint() {
    local run_dir="$1"
    local step
    local latest=""

    while IFS= read -r step; do
        if checkpoint_is_complete "$run_dir/$step"; then
            latest="$step"
        fi
    done < <(list_numeric_checkpoints "$run_dir")

    printf '%s\n' "$latest"
}

# ------------------------------------------------------------------------------
# Persistent checkpoint synchronization
# ------------------------------------------------------------------------------

sync_one_checkpoint() {
    local step="$1"
    local source="$LOCAL_RUN/$step"
    local destination="$PERSIST_RUN/$step"
    local staging="$PERSIST_RUN/.sync-${step}-${SLURM_JOB_ID:-$$}"

    if ! checkpoint_is_complete "$source"; then
        echo "[checkpoint-sync] Step $step is not complete locally; skipping"
        return 1
    fi

    if checkpoint_is_complete "$destination"; then
        echo "[checkpoint-sync] Step $step already exists in persistent storage"
        return 0
    fi

    # A numeric destination that exists but is not complete is unsafe.
    if [[ -e "$destination" ]]; then
        echo "[checkpoint-sync] ERROR: Incomplete numeric destination exists: $destination" >&2
        return 1
    fi

    echo
    echo "[checkpoint-sync] Starting step $step"
    echo "[checkpoint-sync] Source:      $source"
    echo "[checkpoint-sync] Staging:     $staging"
    echo "[checkpoint-sync] Destination: $destination"
    echo "[checkpoint-sync] Started at:  $(date --iso-8601=seconds 2>/dev/null || date)"

    mkdir -p "$PERSIST_RUN"
    mkdir -p "$staging"

    # Keep partial data in staging if rsync fails. The next attempt can resume
    # instead of retransmitting the entire ~50 GiB checkpoint.
    if ! rsync -a \
        --delete \
        --partial \
        --info=stats2 \
        "$source/" \
        "$staging/"
    then
        echo "[checkpoint-sync] WARNING: rsync failed for step $step; will retry" >&2
        return 1
    fi

    if ! checkpoint_is_complete "$staging"; then
        echo "[checkpoint-sync] ERROR: Staged checkpoint failed structure validation: $staging" >&2
        return 1
    fi

    # Staging and destination are on the same /data-sl filesystem. Only expose
    # the numeric directory after the complete copy is present.
    if ! mv "$staging" "$destination"; then
        echo "[checkpoint-sync] ERROR: Atomic promotion failed for step $step" >&2
        return 1
    fi

    echo "[checkpoint-sync] CHECKPOINT_SYNC_OK step=$step"
    echo "[checkpoint-sync] Finished at: $(date --iso-8601=seconds 2>/dev/null || date)"
    du -sh "$destination" || true

    return 0
}

sync_wandb_id() {
    if [[ -f "$LOCAL_RUN/wandb_id.txt" ]]; then
        cp -a \
            "$LOCAL_RUN/wandb_id.txt" \
            "$PERSIST_RUN/.wandb_id.txt.${SLURM_JOB_ID:-$$}.tmp"

        mv \
            "$PERSIST_RUN/.wandb_id.txt.${SLURM_JOB_ID:-$$}.tmp" \
            "$PERSIST_RUN/wandb_id.txt"
    fi
}

sync_completed_checkpoints() {
    local path
    local step
    local status=0

    [[ -d "$LOCAL_RUN" ]] || return 0

    mkdir -p "$PERSIST_RUN"

    while IFS= read -r path; do
        step="$(basename "$path")"

        [[ "$step" =~ ^[0-9]+$ ]] || continue

        if ! sync_one_checkpoint "$step"; then
            status=1
        fi
    done < <(
        find "$LOCAL_RUN" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -print 2>/dev/null |
            sort -V
    )

    sync_wandb_id || status=1

    return "$status"
}

checkpoint_sync_loop() {
    echo "[checkpoint-sync] Background synchronization started"
    echo "[checkpoint-sync] Interval: ${SYNC_INTERVAL}s"
    echo "[checkpoint-sync] Local run: $LOCAL_RUN"
    echo "[checkpoint-sync] Persistent run: $PERSIST_RUN"

    while [[ ! -e "$STOP_SYNC_FILE" ]]; do
        sync_completed_checkpoints || true

        # Use short sleeps so shutdown does not always wait the full interval.
        local elapsed=0
        while (( elapsed < SYNC_INTERVAL )); do
            [[ -e "$STOP_SYNC_FILE" ]] && break
            sleep 1
            elapsed=$((elapsed + 1))
        done
    done

    echo "[checkpoint-sync] Background synchronization stopped"
}

# ------------------------------------------------------------------------------
# Restore latest persistent checkpoint into local storage
# ------------------------------------------------------------------------------

restore_checkpoint_to_local() {
    local latest_step

    echo
    echo "===== Restoring checkpoint to node-local storage ====="

    require_dir "$PERSIST_RUN" "Persistent experiment directory"

    latest_step="$(latest_complete_checkpoint "$PERSIST_RUN")"

    [[ -n "$latest_step" ]] ||
        die "No complete checkpoint found in $PERSIST_RUN"

    echo "Latest complete persistent checkpoint: $latest_step"
    echo "Source: $PERSIST_RUN/$latest_step"
    echo "Destination: $LOCAL_RUN/$latest_step"

    mkdir -p "$LOCAL_RUN/$latest_step"

    time rsync -a \
        --delete \
        --partial \
        --info=stats2 \
        "$PERSIST_RUN/$latest_step/" \
        "$LOCAL_RUN/$latest_step/"

    checkpoint_is_complete "$LOCAL_RUN/$latest_step" ||
        die "Restored local checkpoint failed validation: $LOCAL_RUN/$latest_step"

    if [[ -f "$PERSIST_RUN/wandb_id.txt" ]]; then
        cp -a \
            "$PERSIST_RUN/wandb_id.txt" \
            "$LOCAL_RUN/wandb_id.txt"
    fi

    echo "LOCAL_CHECKPOINT_RESTORE_OK step=$latest_step"
    du -sh "$LOCAL_RUN/$latest_step"
}

# ------------------------------------------------------------------------------
# Signal and cleanup handling
# ------------------------------------------------------------------------------

handle_signal() {
    local signal_name="$1"

    RECEIVED_SIGNAL=1

    echo
    echo "===== Received signal: $signal_name ====="

    if [[ -n "${TRAIN_PID:-}" ]] &&
        kill -0 "$TRAIN_PID" 2>/dev/null
    then
        echo "Forwarding SIGTERM to training process PID=$TRAIN_PID"
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
    fi
}

cleanup() {
    local original_exit_code=$?
    local cleanup_exit_code="$original_exit_code"

    trap - EXIT INT TERM

    echo
    echo "======================================================================"
    echo "Cleanup started"
    echo "original_exit_code=$original_exit_code"
    echo "received_signal=$RECEIVED_SIGNAL"
    echo "======================================================================"

    touch "$STOP_SYNC_FILE" 2>/dev/null || true

    if [[ -n "${SYNC_PID:-}" ]] &&
        kill -0 "$SYNC_PID" 2>/dev/null
    then
        echo "Waiting for checkpoint sync loop PID=$SYNC_PID"
        wait "$SYNC_PID" 2>/dev/null || true
    fi

    echo
    echo "===== Final checkpoint synchronization ====="

    if sync_completed_checkpoints; then
        FINAL_SYNC_OK=1
        echo "FINAL_CHECKPOINT_SYNC_OK"
    else
        FINAL_SYNC_OK=0
        echo "FINAL_CHECKPOINT_SYNC_FAILED" >&2

        if [[ "$cleanup_exit_code" -eq 0 ]]; then
            cleanup_exit_code=1
        fi
    fi

    echo
    echo "===== Local checkpoints ====="
    list_numeric_checkpoints "$LOCAL_RUN" || true

    echo
    echo "===== Persistent checkpoints ====="
    list_numeric_checkpoints "$PERSIST_RUN" || true

    echo
    echo "===== Remaining staging directories ====="
    find "$PERSIST_RUN" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name '.sync-*' \
        -print 2>/dev/null || true

    # Only remove node-local data when the final synchronization succeeded.
    if [[ "$KEEP_LOCAL_TMP" == "1" ]]; then
        echo "Keeping local temporary directory for debugging:"
        echo "$JOB_TMP"
    elif [[ "$FINAL_SYNC_OK" == "1" ]]; then
        echo "Removing local temporary directory:"
        echo "$JOB_TMP"
        rm -rf "$JOB_TMP" || true
    else
        echo "WARNING: Final synchronization failed." >&2
        echo "Local temporary data was not deliberately removed:" >&2
        echo "$JOB_TMP" >&2
        echo "Note: CHESS may still clean /tmp after the job exits." >&2
    fi

    echo
    echo "Cleanup finished with exit code $cleanup_exit_code"

    exit "$cleanup_exit_code"
}

trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT
trap cleanup EXIT

# ------------------------------------------------------------------------------
# Validate host-visible inputs
# ------------------------------------------------------------------------------

require_dir "$REPO" "OpenPI repository"
require_file "$REPO/scripts/train.py" "OpenPI training entry point"
require_file "$REPO/src/openpi/training/config.py" "OpenPI training config"

require_dir "$SANDBOX" "OpenPI Singularity sandbox"

# The venv Python may be an absolute symlink whose target only exists inside
# the container.
if [[ ! -e "$VENV/bin/python" && ! -L "$VENV/bin/python" ]]; then
    die "OpenPI virtual-environment Python entry not found: $VENV/bin/python"
fi

require_dir "$DATASET_DIR" "LeRobot dataset"
require_dir "$DATASET_DIR/meta" "LeRobot metadata directory"
require_dir "$DATASET_DIR/data" "LeRobot data directory"
require_dir "$DATASET_DIR/videos" "LeRobot videos directory"
require_file "$DATASET_DIR/meta/info.json" "LeRobot info.json"

require_dir "$WEIGHTS_DIR" "π0.5 base checkpoint params directory"
directory_not_empty "$WEIGHTS_DIR" ||
    die "π0.5 base checkpoint directory is empty: $WEIGHTS_DIR"

require_file "$NORM_STATS_FILE" "Normalization statistics"
require_file "$PALIGEMMA_TOKENIZER" "PaliGemma tokenizer"

command -v singularity >/dev/null 2>&1 ||
    die "singularity command not found"

command -v rsync >/dev/null 2>&1 ||
    die "rsync command not found"

command -v nvidia-smi >/dev/null 2>&1 ||
    die "nvidia-smi command not found"

# ------------------------------------------------------------------------------
# Create directories
# ------------------------------------------------------------------------------

mkdir -p \
    "$ASSETS_BASE_DIR" \
    "$PERSIST_CHECKPOINT_BASE" \
    "$PERSIST_RUN" \
    "$OPENPI_DATA_HOME" \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$UV_CACHE_DIR" \
    "$RUNTIME_HOME" \
    "$RUNTIME_HOME/.cache/jax" \
    "$RUNTIME_HOME/.cache/torch" \
    "$RUNTIME_HOME/.config" \
    "$WANDB_DIR" \
    "$JOB_TMP" \
    "$LOCAL_CHECKPOINT_BASE"

rm -f "$STOP_SYNC_FILE"

# ------------------------------------------------------------------------------
# Check node-local storage
# ------------------------------------------------------------------------------

AVAILABLE_KB="$(
    df -Pk "$JOB_TMP" |
        awk 'NR == 2 {print $4}'
)"

MIN_REQUIRED_KB=$((MIN_LOCAL_FREE_GIB * 1024 * 1024))

if ! [[ "$AVAILABLE_KB" =~ ^[0-9]+$ ]]; then
    die "Unable to determine available node-local storage"
fi

if (( AVAILABLE_KB < MIN_REQUIRED_KB )); then
    die "Less than ${MIN_LOCAL_FREE_GIB} GiB available in $JOB_TMP"
fi

# ------------------------------------------------------------------------------
# Protect against accidental persistent-run mixing
# ------------------------------------------------------------------------------

if [[ "$RESUME" != "1" ]]; then
    EXISTING_STEP="$(latest_complete_checkpoint "$PERSIST_RUN")"

    if [[ -n "$EXISTING_STEP" ]]; then
        die "Persistent experiment already has checkpoint $EXISTING_STEP. Use RESUME=1 or choose a new EXP_NAME."
    fi
fi

# ------------------------------------------------------------------------------
# Job diagnostics
# ------------------------------------------------------------------------------

echo "OpenPI venv Python entry:"
ls -l "$VENV/bin/python"

echo
echo "======================================================================"
echo "OpenPI π0.5 CHESS training job"
echo "======================================================================"
echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "job_id=${SLURM_JOB_ID:-local}"
echo "job_name=${SLURM_JOB_NAME:-openpi-tienkung-full}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo
echo "BASE=$BASE"
echo "REPO=$REPO"
echo "SANDBOX=$SANDBOX"
echo "VENV=$VENV"
echo
echo "CONFIG_NAME=$CONFIG_NAME"
echo "EXP_NAME=$EXP_NAME"
echo "RESUME=$RESUME"
echo "FSDP_DEVICES=$FSDP_DEVICES"
echo "BATCH_SIZE=${BATCH_SIZE:-config-default}"
echo "NUM_TRAIN_STEPS=$NUM_TRAIN_STEPS"
echo "SAVE_INTERVAL=$SAVE_INTERVAL"
echo "KEEP_PERIOD=$KEEP_PERIOD"
echo "LOG_INTERVAL=$LOG_INTERVAL"
echo
echo "DATASET_DIR=$DATASET_DIR"
echo "WEIGHTS_DIR=$WEIGHTS_DIR"
echo "NORM_STATS_FILE=$NORM_STATS_FILE"
echo "PALIGEMMA_TOKENIZER=$PALIGEMMA_TOKENIZER"
echo
echo "LOCAL_CHECKPOINT_BASE=$LOCAL_CHECKPOINT_BASE"
echo "LOCAL_RUN=$LOCAL_RUN"
echo "PERSIST_CHECKPOINT_BASE=$PERSIST_CHECKPOINT_BASE"
echo "PERSIST_RUN=$PERSIST_RUN"
echo "SYNC_INTERVAL=$SYNC_INTERVAL"
echo "KEEP_LOCAL_TMP=$KEEP_LOCAL_TMP"
echo
echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "HF_HOME=$HF_HOME"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_MODE=$WANDB_MODE"
echo "LEROBOT_VIDEO_BACKEND=$LEROBOT_VIDEO_BACKEND"
echo
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "======================================================================"

echo
echo "===== Persistent storage ====="
df -h "$BASE" || true

echo
echo "===== Node-local storage ====="
df -h "$JOB_TMP"
echo "Available node-local KiB: $AVAILABLE_KB"

echo
echo "===== Dataset ====="
du -sh "$DATASET_DIR" || true

echo
echo "===== Base weights ====="
du -sh "$WEIGHTS_DIR" || true

echo
echo "===== Host GPUs ====="
nvidia-smi

# ------------------------------------------------------------------------------
# Restore persistent checkpoint when resuming
# ------------------------------------------------------------------------------

if [[ "$RESUME" == "1" ]]; then
    restore_checkpoint_to_local
else
    rm -rf "$LOCAL_RUN"
fi

# ------------------------------------------------------------------------------
# Singularity options
# ------------------------------------------------------------------------------

SINGULARITY_ARGS=(
    exec
    --cleanenv
    --nv
    --bind "$BASE:$BASE"
    --bind "$JOB_TMP:$JOB_TMP"
    --home "$RUNTIME_HOME"
    --pwd "$REPO"
    --env "HF_LEROBOT_HOME=$DATASET_ROOT"
    --env "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
    --env "HF_HOME=$HF_HOME"
    --env "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
    --env "UV_CACHE_DIR=$UV_CACHE_DIR"
    --env "WANDB_DIR=$WANDB_DIR"
    --env "WANDB_MODE=$WANDB_MODE"
    --env "LEROBOT_VIDEO_BACKEND=$LEROBOT_VIDEO_BACKEND"
    --env "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
)

if [[ -n "${XLA_FLAGS:-}" ]]; then
    SINGULARITY_ARGS+=(--env "XLA_FLAGS=$XLA_FLAGS")
fi

# ------------------------------------------------------------------------------
# Container preflight
# ------------------------------------------------------------------------------

echo
echo "===== Container preflight ====="

singularity "${SINGULARITY_ARGS[@]}" \
    --env "EXPECTED_FSDP_DEVICES=$FSDP_DEVICES" \
    --env "EXPECTED_WEIGHTS_DIR=$WEIGHTS_DIR" \
    --env "EXPECTED_NORM_STATS=$NORM_STATS_FILE" \
    --env "EXPECTED_TOKENIZER=$PALIGEMMA_TOKENIZER" \
    "$SANDBOX" \
    "$VENV/bin/python" - <<'PY'
import os
from pathlib import Path
import sys

import jax

print("Python:", sys.version)
print("JAX:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX device count:", jax.device_count())

expected_fsdp = int(os.environ["EXPECTED_FSDP_DEVICES"])

if not any(device.platform == "gpu" for device in jax.devices()):
    raise RuntimeError(f"JAX did not detect CUDA GPUs: {jax.devices()}")

if jax.device_count() % expected_fsdp != 0:
    raise RuntimeError(
        f"JAX sees {jax.device_count()} devices, but "
        f"FSDP_DEVICES={expected_fsdp} does not divide the device count"
    )

for variable in (
    "EXPECTED_WEIGHTS_DIR",
    "EXPECTED_NORM_STATS",
    "EXPECTED_TOKENIZER",
):
    path = Path(os.environ[variable])
    if not path.exists():
        raise FileNotFoundError(f"{variable} is not visible: {path}")
    print(f"{variable}: {path}")

from openpi.training import config as training_config

cfg = training_config.get_config("pi05_tienkung_pick_place")

print("Config name:", cfg.name)
print("Model type:", cfg.model.model_type)
print("PaliGemma variant:", cfg.model.paligemma_variant)
print("Action expert variant:", cfg.model.action_expert_variant)
print("Action dimension:", cfg.model.action_dim)
print("Action horizon:", cfg.model.action_horizon)
print("Config batch size:", cfg.batch_size)
print("Config train steps:", cfg.num_train_steps)
print("EMA decay:", cfg.ema_decay)
print("Weight loader:", cfg.weight_loader)
print("Data repo ID:", cfg.data.repo_id)

print("OPENPI_PREFLIGHT_OK")
PY

# ------------------------------------------------------------------------------
# Determine and validate effective batch size
# ------------------------------------------------------------------------------

if [[ -n "$BATCH_SIZE" ]]; then
    EFFECTIVE_BATCH_SIZE="$BATCH_SIZE"
else
    EFFECTIVE_BATCH_SIZE="$(
        singularity "${SINGULARITY_ARGS[@]}" \
            "$SANDBOX" \
            "$VENV/bin/python" -c \
            "from openpi.training import config; print(config.get_config('$CONFIG_NAME').batch_size)"
    )"
fi

[[ "$EFFECTIVE_BATCH_SIZE" =~ ^[0-9]+$ ]] ||
    die "Unable to determine numeric batch size: $EFFECTIVE_BATCH_SIZE"

if (( EFFECTIVE_BATCH_SIZE % FSDP_DEVICES != 0 )); then
    die "Batch size $EFFECTIVE_BATCH_SIZE must be divisible by FSDP_DEVICES=$FSDP_DEVICES"
fi

echo
echo "Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "Per-FSDP-device batch: $((EFFECTIVE_BATCH_SIZE / FSDP_DEVICES))"

# ------------------------------------------------------------------------------
# Build training arguments
# ------------------------------------------------------------------------------

TRAIN_ARGS=(
    "$REPO/scripts/train.py"
    "$CONFIG_NAME"
    "--exp-name=$EXP_NAME"
    "--assets-base-dir=$ASSETS_BASE_DIR"
    "--checkpoint-base-dir=$LOCAL_CHECKPOINT_BASE"
    "--fsdp-devices=$FSDP_DEVICES"
    "--num-train-steps=$NUM_TRAIN_STEPS"
    "--save-interval=$SAVE_INTERVAL"
    "--keep-period=$KEEP_PERIOD"
    "--log-interval=$LOG_INTERVAL"
)

if [[ "$RESUME" == "1" ]]; then
    TRAIN_ARGS+=("--resume")
else
    TRAIN_ARGS+=("--overwrite")
fi

if [[ -n "$BATCH_SIZE" ]]; then
    TRAIN_ARGS+=("--batch-size=$BATCH_SIZE")
fi

if [[ "$WANDB_DISABLED" == "1" ]]; then
    TRAIN_ARGS+=("--no-wandb-enabled")
fi

# ------------------------------------------------------------------------------
# Start checkpoint watcher
# ------------------------------------------------------------------------------

checkpoint_sync_loop &
SYNC_PID=$!

echo
echo "Checkpoint synchronization PID: $SYNC_PID"

# ------------------------------------------------------------------------------
# Start training
# ------------------------------------------------------------------------------

echo
echo "===== Running command ====="
printf '%q ' "$VENV/bin/python" "${TRAIN_ARGS[@]}"
printf '\n===========================\n'

START_TIME="$(date +%s)"

singularity "${SINGULARITY_ARGS[@]}" \
    "$SANDBOX" \
    "$VENV/bin/python" \
    "${TRAIN_ARGS[@]}" &

TRAIN_PID=$!

echo "Training PID: $TRAIN_PID"

set +e
wait "$TRAIN_PID"
TRAIN_EXIT_CODE=$?
set -e

TRAIN_PID=""

END_TIME="$(date +%s)"

echo
echo "Training process exited"
echo "exit_code=$TRAIN_EXIT_CODE"
echo "elapsed_seconds=$((END_TIME - START_TIME))"

if [[ "$TRAIN_EXIT_CODE" -ne 0 ]]; then
    exit "$TRAIN_EXIT_CODE"
fi

echo
echo "======================================================================"
echo "TRAINING_PROCESS_COMPLETED"
echo "Final checkpoint synchronization will run during cleanup."
echo "======================================================================"
