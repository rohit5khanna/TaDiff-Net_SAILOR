#!/bin/bash
# Submit with: sbatch scripts/run_3gpu_sweep.slurm

#SBATCH -J tadiff-3gpu-sweep
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -o sweep3gpu_%j.out
#SBATCH -e sweep3gpu_%j.err

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

# Override as needed at submit time:
#   sbatch --export=ALL,MAX_STEPS=20000 scripts/run_3gpu_sweep.slurm
export MAX_STEPS="${MAX_STEPS:-5000}"
export PRECISION="${PRECISION:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export VAL_INTERVAL_EPOCH="${VAL_INTERVAL_EPOCH:-1000}"

# Baseline-safe runtime knobs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/run_3gpu_sweep.sh
