#!/bin/bash
# Submit one 3-GPU DDP trial:
#   sbatch --export=ALL,TRIAL_NAME=three_x_batch_fixed_lambda scripts/run_3gpu_trial.slurm

#SBATCH -J tadiff-3gpu-trial
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 48:00:00
#SBATCH -o trial3gpu_%j.out
#SBATCH -e trial3gpu_%j.err

set -euo pipefail

REPO_DIR="/work/11343/rohitk59/ls6/TaDiff_Baseline/TaDiff-Net_SAILOR"
cd "$REPO_DIR"

module load gcc cuda
source /work/11343/rohitk59/ls6/TaDiff_Baseline/TaDiff-Net_SAILOR/tadiff/bin/activate

# Preflight.
which python
python -V
python - <<'PY'
import torch
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA is not available."
PY

# Common runtime defaults (override with --export=ALL,KEY=VALUE at sbatch time).
TRIAL_NAME="${TRIAL_NAME:-three_x_batch_fixed_lambda}"
MAX_STEPS="${MAX_STEPS:-5000}"
PRECISION="${PRECISION:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL_EPOCH="${VAL_INTERVAL_EPOCH:-1000}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2}"
GPU_STRATEGY="${GPU_STRATEGY:-ddp}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "$TRIAL_NAME" in
  three_x_batch_fixed_lambda)
    LAMBDA_SCHEDULE="fixed"
    SW_BATCH=32
    ACCUM=2
    ;;
  original_batch_fixed_lambda)
    # Keep effective batch close to original 1-GPU baseline.
    LAMBDA_SCHEDULE="fixed"
    SW_BATCH=11
    ACCUM=2
    ;;
  three_x_batch_time_lambda)
    LAMBDA_SCHEDULE="time_dependent"
    SW_BATCH=32
    ACCUM=2
    ;;
  *)
    echo "Unknown TRIAL_NAME: $TRIAL_NAME"
    echo "Expected one of: three_x_batch_fixed_lambda | original_batch_fixed_lambda | three_x_batch_time_lambda"
    exit 2
    ;;
esac

echo "TRIAL_NAME=$TRIAL_NAME"
echo "MAX_STEPS=$MAX_STEPS"
echo "GPU_DEVICES=$GPU_DEVICES"
echo "GPU_STRATEGY=$GPU_STRATEGY"
echo "LAMBDA_SCHEDULE=$LAMBDA_SCHEDULE"
echo "SW_BATCH=$SW_BATCH"
echo "ACCUM=$ACCUM"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "PRECISION=$PRECISION"

python train.py \
  --gpu_devices "$GPU_DEVICES" \
  --gpu_strategy "$GPU_STRATEGY" \
  --lambda_schedule "$LAMBDA_SCHEDULE" \
  --max_steps "$MAX_STEPS" \
  --num_workers "$NUM_WORKERS" \
  --precision "$PRECISION" \
  --sw_batch "$SW_BATCH" \
  --accumulate_grad_batches "$ACCUM" \
  --val_interval_epoch "$VAL_INTERVAL_EPOCH" \
  --log_interval "$LOG_INTERVAL"
