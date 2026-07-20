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

# Submit with:
#   sbatch cluster_training.sh
#
# Common overrides:
#   EXP_NAME=my_run BATCH_SIZE=32 sbatch cluster_training.sh
#   RESUME=1 EXP_NAME=my_run sbatch cluster_training.sh

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CONFIG_NAME="${CONFIG_NAME:-pi05_tienkung_pick_place}"
EXP_NAME="${EXP_NAME:-0719_tienkung_pick_place_full}"

# Your LeRobot dataset should be:
#   ${HF_LEROBOT_HOME}/${repo_id in config.py}
# Current repo_id in config.py:
#   pick_apple/0704
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data-sl/lsy_ws/dataset/VLA}"

# Put large caches on shared/data storage instead of $HOME.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/data-sl/lsy_ws/cache/openpi}"
export HF_HOME="${HF_HOME:-/data-sl/lsy_ws/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export WANDB_DIR="${WANDB_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}/wandb}"

# JAX memory behavior.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_triton_gemm_any=True}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"

# Avoid inheriting a broken HuggingFace mirror unless you explicitly set it at submit time.
if [[ -z "${USE_HF_ENDPOINT:-}" ]]; then
  unset HF_ENDPOINT
fi

mkdir -p "${OPENPI_DATA_HOME}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${WANDB_DIR}"

echo "===== Slurm job ====="
echo "job_id=${SLURM_JOB_ID:-local}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "config=${CONFIG_NAME}"
echo "exp_name=${EXP_NAME}"
echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "WANDB_DIR=${WANDB_DIR}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "FSDP_DEVICES=${FSDP_DEVICES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "====================="

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found. Run uv sync before submitting, or load the correct environment in this script." >&2
  exit 1
fi

if [[ ! -d "${HF_LEROBOT_HOME}/pick_apple/0704/meta" ]]; then
  echo "ERROR: Dataset metadata not found at ${HF_LEROBOT_HOME}/pick_apple/0704/meta" >&2
  echo "Update HF_LEROBOT_HOME or repo_id in src/openpi/training/config.py." >&2
  exit 1
fi

TRAIN_ARGS=(
  scripts/train.py
  "${CONFIG_NAME}"
  "--exp-name=${EXP_NAME}"
)

if [[ "${RESUME:-0}" == "1" ]]; then
  TRAIN_ARGS+=("--resume")
else
  TRAIN_ARGS+=("--overwrite")
fi

# Optional CLI overrides supported by TrainConfig.
if [[ -n "${BATCH_SIZE:-}" ]]; then
  TRAIN_ARGS+=("--batch-size=${BATCH_SIZE}")
fi
if [[ -n "${NUM_TRAIN_STEPS:-}" ]]; then
  TRAIN_ARGS+=("--num-train-steps=${NUM_TRAIN_STEPS}")
fi
if [[ -n "${SAVE_INTERVAL:-}" ]]; then
  TRAIN_ARGS+=("--save-interval=${SAVE_INTERVAL}")
fi
if [[ -n "${LOG_INTERVAL:-}" ]]; then
  TRAIN_ARGS+=("--log-interval=${LOG_INTERVAL}")
fi
TRAIN_ARGS+=("--fsdp-devices=${FSDP_DEVICES}")
if [[ "${WANDB_DISABLED:-0}" == "1" ]]; then
  TRAIN_ARGS+=("--no-wandb-enabled")
fi

echo "Running: uv run ${TRAIN_ARGS[*]}"
uv run "${TRAIN_ARGS[@]}"
