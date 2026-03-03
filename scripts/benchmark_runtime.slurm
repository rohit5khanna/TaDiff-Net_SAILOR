#!/bin/bash
# Example Slurm wrapper for runtime benchmarking.
# Submit with: sbatch scripts/benchmark_runtime.slurm

#SBATCH -J tadiff-bench
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH -t 04:00:00
#SBATCH -o bench_%j.out
#SBATCH -e bench_%j.err

set -euo pipefail

REPO_DIR="/work/11343/rohitk59/ls6/TaDiff_Baseline/TaDiff-Net_SAILOR"
cd "$REPO_DIR"

# Optional: activate your environment here.
# source /work/11343/rohitk59/ls6/TaDiff_Baseline/TaDiff-Net_SAILOR/tadiff/bin/activate

# Runtime tuning profile defaults.
export MAX_STEPS=400
export NUM_WORKERS_LIST="8 16 24"
export PROFILE_COMPILE=1
export TORCH_COMPILE_MODE=default
export LOG_INTERVAL=100
export WANDB_MODE=disabled

# Thread caps for dataloader stability/perf tuning.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Keep baseline semantics fixed for profiling.
export LAMBDA_SCHEDULE=fixed
export PRECISION=32
export SW_BATCH=32
export ACCUM=2

bash scripts/benchmark_runtime.sh

