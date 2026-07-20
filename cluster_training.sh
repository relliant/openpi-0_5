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
# Default:
#   - 1 node
#   - 8 GPUs
#   - FSDP across 8 GPUs
#   - full fine-tuning
#   - W&B offline
#
# Submit full training:
#   sbatch cluster_training.sh
#
# Smoke test:
#   EXP_NAME=smoke_test \
#   BATCH_SIZE=8 \
#   NUM_TRAIN_STEPS=5 \
#   SAVE_INTERVAL=5 \
#   LOG_INTERVAL=1 \
#   WANDB_DISABLED=1 \
#   sbatch cluster_training.sh
#
# Resume:
#   RESUME=1 \
#   EXP_NAME=0719_tienkung_pick_place_full \
#   sbatch cluster_training.sh
#
# IMPORTANT:
#   Batch size must be divisible by the number of visible JAX devices.
#   With 8 GPUs, use batch sizes such as 8, 16, 24, 32.
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
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$BASE/checkpoints}"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$BASE/cache/openpi}"
HF_HOME="${HF_HOME:-$BASE/cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$BASE/cache/uv}"

RUNTIME_HOME="${RUNTIME_HOME:-$BASE/runtime-home/openpi}"
WANDB_DIR="${WANDB_DIR:-$BASE/logs/wandb}"

# Current repo_id is pick_apple/0704, so compute_norm_stats.py writes here.
NORM_STATS_FILE="$ASSETS_BASE_DIR/pi05_tienkung_pick_place/$DATASET_REPO_ID/norm_stats.json"

# PaligemmaTokenizer always loads this asset. Compute nodes are offline, so it
# must already exist in OPENPI_DATA_HOME.
PALIGEMMA_TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"

# ------------------------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------------------------

CONFIG_NAME="${CONFIG_NAME:-pi05_tienkung_pick_place}"
EXP_NAME="${EXP_NAME:-0719_tienkung_pick_place_full}"

FSDP_DEVICES="${FSDP_DEVICES:-8}"

# Default config batch size is 16. Override with BATCH_SIZE when needed.
BATCH_SIZE="${BATCH_SIZE:-}"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
LOG_INTERVAL="${LOG_INTERVAL:-}"

RESUME="${RESUME:-0}"
WANDB_DISABLED="${WANDB_DISABLED:-0}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

# Do not set experimental XLA flags by default. If needed, pass XLA_FLAGS
# explicitly when submitting the job.
if [[ -n "${XLA_FLAGS:-}" ]]; then
    export XLA_FLAGS
fi

# Compute nodes may not have internet. Offline is the safe default.
export WANDB_MODE="${WANDB_MODE:-offline}"

# Do not inherit an accidental Hugging Face mirror unless explicitly requested.
if [[ -z "${USE_HF_ENDPOINT:-}" ]]; then
    unset HF_ENDPOINT || true
fi

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
    find "$path" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

# ------------------------------------------------------------------------------
# Validate persistent paths before starting the container
# ------------------------------------------------------------------------------

require_dir "$REPO" "OpenPI repository"
require_file "$REPO/scripts/train.py" "OpenPI training entry point"
require_file "$REPO/src/openpi/training/config.py" "OpenPI training config"

require_dir "$SANDBOX" "OpenPI Singularity sandbox"

# The venv was created inside the container. Its Python entry may be an
# absolute symlink whose target only exists inside that container.
if [[ ! -e "$VENV/bin/python" && ! -L "$VENV/bin/python" ]]; then
    die "OpenPI virtual-environment Python entry not found: $VENV/bin/python"
fi

echo "OpenPI venv Python entry:"
ls -l "$VENV/bin/python"
require_dir "$DATASET_DIR" "LeRobot dataset"
require_dir "$DATASET_DIR/meta" "LeRobot metadata directory"
require_dir "$DATASET_DIR/data" "LeRobot data directory"

# Videos are expected for this vision dataset.
require_dir "$DATASET_DIR/videos" "LeRobot videos directory"
require_file "$DATASET_DIR/meta/info.json" "LeRobot info.json"

require_dir "$WEIGHTS_DIR" "π0.5 base checkpoint params directory"
directory_not_empty "$WEIGHTS_DIR" ||
    die "π0.5 base checkpoint params directory is empty: $WEIGHTS_DIR"

require_file "$NORM_STATS_FILE" "Normalization statistics"
require_file "$PALIGEMMA_TOKENIZER" "PaliGemma tokenizer"

# ------------------------------------------------------------------------------
# Create writable persistent directories
# ------------------------------------------------------------------------------

mkdir -p \
    "$ASSETS_BASE_DIR" \
    "$CHECKPOINT_BASE_DIR" \
    "$OPENPI_DATA_HOME" \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$UV_CACHE_DIR" \
    "$RUNTIME_HOME" \
    "$RUNTIME_HOME/.cache/jax" \
    "$RUNTIME_HOME/.cache/torch" \
    "$RUNTIME_HOME/.config" \
    "$WANDB_DIR"

# ------------------------------------------------------------------------------
# Print job configuration
# ------------------------------------------------------------------------------

echo "======================================================================"
echo "OpenPI π0.5 CHESS training job"
echo "======================================================================"
echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "job_id=${SLURM_JOB_ID:-local}"
echo "job_name=${SLURM_JOB_NAME:-openpi-tienkung-full}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unset}"
echo
echo "BASE=$BASE"
echo "REPO=$REPO"
echo "SANDBOX=$SANDBOX"
echo "VENV=$VENV"
echo
echo "CONFIG_NAME=$CONFIG_NAME"
echo "EXP_NAME=$EXP_NAME"
echo "FSDP_DEVICES=$FSDP_DEVICES"
echo "BATCH_SIZE=${BATCH_SIZE:-config-default}"
echo "NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-config-default}"
echo "SAVE_INTERVAL=${SAVE_INTERVAL:-config-default}"
echo "LOG_INTERVAL=${LOG_INTERVAL:-config-default}"
echo "RESUME=$RESUME"
echo
echo "DATASET_ROOT=$DATASET_ROOT"
echo "DATASET_REPO_ID=$DATASET_REPO_ID"
echo "DATASET_DIR=$DATASET_DIR"
echo "WEIGHTS_DIR=$WEIGHTS_DIR"
echo "NORM_STATS_FILE=$NORM_STATS_FILE"
echo "PALIGEMMA_TOKENIZER=$PALIGEMMA_TOKENIZER"
echo
echo "ASSETS_BASE_DIR=$ASSETS_BASE_DIR"
echo "CHECKPOINT_BASE_DIR=$CHECKPOINT_BASE_DIR"
echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "RUNTIME_HOME=$RUNTIME_HOME"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_MODE=$WANDB_MODE"
echo
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "XLA_FLAGS=${XLA_FLAGS:-unset}"
echo "======================================================================"

echo
echo "===== Storage ====="
df -h "$BASE" "$CHECKPOINT_BASE_DIR" || true

echo
echo "===== Dataset ====="
du -sh "$DATASET_DIR" || true
ls -lh "$DATASET_DIR/meta/info.json"

echo
echo "===== Base weights ====="
du -sh "$WEIGHTS_DIR" || true

echo
echo "===== Host GPU ====="
command -v nvidia-smi >/dev/null 2>&1 ||
    die "nvidia-smi not found on compute node"
nvidia-smi

# ------------------------------------------------------------------------------
# Build CLI arguments
# ------------------------------------------------------------------------------

TRAIN_ARGS=(
    "$REPO/scripts/train.py"
    "$CONFIG_NAME"
    "--exp-name=$EXP_NAME"
    "--assets-base-dir=$ASSETS_BASE_DIR"
    "--checkpoint-base-dir=$CHECKPOINT_BASE_DIR"
    "--fsdp-devices=$FSDP_DEVICES"
)

if [[ "$RESUME" == "1" ]]; then
    TRAIN_ARGS+=("--resume")
else
    TRAIN_ARGS+=("--overwrite")
fi

if [[ -n "$BATCH_SIZE" ]]; then
    TRAIN_ARGS+=("--batch-size=$BATCH_SIZE")
fi

if [[ -n "$NUM_TRAIN_STEPS" ]]; then
    TRAIN_ARGS+=("--num-train-steps=$NUM_TRAIN_STEPS")
fi

if [[ -n "$SAVE_INTERVAL" ]]; then
    TRAIN_ARGS+=("--save-interval=$SAVE_INTERVAL")
fi

if [[ -n "$LOG_INTERVAL" ]]; then
    TRAIN_ARGS+=("--log-interval=$LOG_INTERVAL")
fi

if [[ "$WANDB_DISABLED" == "1" ]]; then
    TRAIN_ARGS+=("--no-wandb-enabled")
fi

# ------------------------------------------------------------------------------
# Shared Singularity options
# ------------------------------------------------------------------------------

SINGULARITY_ARGS=(
    exec
    --cleanenv
    --nv
    --bind "$BASE:$BASE"
    --home "$RUNTIME_HOME"
    --pwd "$REPO"
    --env "HF_LEROBOT_HOME=$DATASET_ROOT"
    --env "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
    --env "HF_HOME=$HF_HOME"
    --env "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
    --env "UV_CACHE_DIR=$UV_CACHE_DIR"
    --env "WANDB_DIR=$WANDB_DIR"
    --env "WANDB_MODE=$WANDB_MODE"
    --env "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
)

if [[ -n "${XLA_FLAGS:-}" ]]; then
    SINGULARITY_ARGS+=(--env "XLA_FLAGS=$XLA_FLAGS")
fi

# ------------------------------------------------------------------------------
# Container and GPU preflight
# ------------------------------------------------------------------------------

echo
echo "===== Container preflight ====="

singularity "${SINGULARITY_ARGS[@]}" \
    --env "EXPECTED_FSDP_DEVICES=$FSDP_DEVICES" \
    --env "EXPECTED_DATASET_DIR=$DATASET_DIR" \
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

if jax.device_count() < 1:
    raise RuntimeError("JAX did not detect any devices")

if not any(device.platform == "gpu" for device in jax.devices()):
    raise RuntimeError(f"JAX did not detect a GPU: {jax.devices()}")

if jax.device_count() % expected_fsdp != 0:
    raise RuntimeError(
        f"JAX sees {jax.device_count()} devices, but "
        f"FSDP_DEVICES={expected_fsdp} does not divide the device count"
    )

for variable in (
    "EXPECTED_DATASET_DIR",
    "EXPECTED_WEIGHTS_DIR",
    "EXPECTED_NORM_STATS",
    "EXPECTED_TOKENIZER",
):
    path = Path(os.environ[variable])
    if not path.exists():
        raise FileNotFoundError(f"{variable} is not visible in container: {path}")
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

# Determine the effective batch size and validate divisibility before the long
# JAX compilation starts.
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

if ! [[ "$EFFECTIVE_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
    die "Unable to determine numeric batch size: $EFFECTIVE_BATCH_SIZE"
fi

if (( EFFECTIVE_BATCH_SIZE % FSDP_DEVICES != 0 )); then
    die "Batch size $EFFECTIVE_BATCH_SIZE must be divisible by FSDP_DEVICES=$FSDP_DEVICES"
fi

echo
echo "Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "Per-FSDP-device batch: $((EFFECTIVE_BATCH_SIZE / FSDP_DEVICES))"

# ------------------------------------------------------------------------------
# Run training
# ------------------------------------------------------------------------------

echo
printf '===== Running command =====\n'
printf '%q ' "$VENV/bin/python" "${TRAIN_ARGS[@]}"
printf '\n===========================\n'

START_TIME="$(date +%s)"

singularity "${SINGULARITY_ARGS[@]}" \
    "$SANDBOX" \
    "$VENV/bin/python" \
    "${TRAIN_ARGS[@]}"

END_TIME="$(date +%s)"

echo
echo "======================================================================"
echo "TRAINING_COMPLETED"
echo "elapsed_seconds=$((END_TIME - START_TIME))"
echo "checkpoint_dir=$CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME"
echo "======================================================================"