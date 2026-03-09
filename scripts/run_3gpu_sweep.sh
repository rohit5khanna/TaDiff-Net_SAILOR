#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/benchmarks/three_gpu_${DATE_TAG}}"

# Keep this modest for calibration. Override to 5000000 for full training.
MAX_STEPS="${MAX_STEPS:-5000}"
VAL_INTERVAL_EPOCH="${VAL_INTERVAL_EPOCH:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
PRECISION="${PRECISION:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2}"
GPU_STRATEGY="${GPU_STRATEGY:-ddp}"

# "Original effective batch (~64)" with 3 GPUs: 11 * 2 * 3 = 66
SW_BATCH_ORIG="${SW_BATCH_ORIG:-11}"
ACCUM_ORIG="${ACCUM_ORIG:-2}"

# "3x batch vs original 1-GPU baseline" : 32 * 2 * 3 = 192
SW_BATCH_3X="${SW_BATCH_3X:-32}"
ACCUM_3X="${ACCUM_3X:-2}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_DIR"
CSV_PATH="$OUT_DIR/summary.csv"

cat > "$CSV_PATH" <<'CSV'
trial,lambda_schedule,sw_batch,accumulate_grad_batches,status,steps,elapsed_sec,sec_per_step,log_path,time_path
CSV

{
  echo "date=$(date)"
  echo "root_dir=$ROOT_DIR"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "max_steps=$MAX_STEPS"
  echo "val_interval_epoch=$VAL_INTERVAL_EPOCH"
  echo "precision=$PRECISION"
  echo "num_workers=$NUM_WORKERS"
  echo "gpu_devices=$GPU_DEVICES"
  echo "gpu_strategy=$GPU_STRATEGY"
  echo "sw_batch_orig=$SW_BATCH_ORIG"
  echo "accum_orig=$ACCUM_ORIG"
  echo "sw_batch_3x=$SW_BATCH_3X"
  echo "accum_3x=$ACCUM_3X"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "mkl_num_threads=$MKL_NUM_THREADS"
  echo "numexpr_num_threads=$NUMEXPR_NUM_THREADS"
  echo "pytorch_cuda_alloc_conf=$PYTORCH_CUDA_ALLOC_CONF"
  echo "wandb_mode=${WANDB_MODE:-unset}"
} > "$OUT_DIR/env.txt"

# Preflight check.
python - <<'PY' > "$OUT_DIR/python_env_check.txt" 2>&1
import torch
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA is not available."
PY

run_trial() {
  local trial="$1"
  local lambda_schedule="$2"
  local sw_batch="$3"
  local accum="$4"

  local trial_dir="$OUT_DIR/$trial"
  mkdir -p "$trial_dir"
  local train_log="$trial_dir/train.log"
  local time_log="$trial_dir/time.log"

  local -a cmd=(
    python train.py
    --gpu_devices "$GPU_DEVICES"
    --gpu_strategy "$GPU_STRATEGY"
    --lambda_schedule "$lambda_schedule"
    --max_steps "$MAX_STEPS"
    --num_workers "$NUM_WORKERS"
    --precision "$PRECISION"
    --sw_batch "$sw_batch"
    --accumulate_grad_batches "$accum"
    --val_interval_epoch "$VAL_INTERVAL_EPOCH"
    --log_interval "$LOG_INTERVAL"
  )

  printf "%q " "${cmd[@]}" > "$trial_dir/cmd.shline"
  echo >> "$trial_dir/cmd.shline"

  echo "[RUN] $trial"
  echo "      $(cat "$trial_dir/cmd.shline")"

  set +e
  /usr/bin/time -p "${cmd[@]}" > "$train_log" 2> "$time_log"
  local rc=$?
  set -e

  local elapsed_sec
  elapsed_sec="$(awk '/^real /{print $2}' "$time_log" | tail -n 1)"
  local steps
  steps="$(awk -F': ' '/trainer\/global_step/{gsub(/[^0-9]/, "", $NF); if($NF!="") s=$NF} END{print s}' "$train_log")"
  if [[ -z "${steps:-}" && "$rc" -eq 0 ]]; then
    steps="$MAX_STEPS"
  fi

  local sec_per_step=""
  if [[ -n "${steps:-}" && "$steps" -gt 0 && -n "${elapsed_sec:-}" ]]; then
    sec_per_step="$(awk -v e="$elapsed_sec" -v s="$steps" 'BEGIN{printf "%.6f", e/s}')"
  fi

  local status="ok"
  if [[ "$rc" -ne 0 ]]; then
    status="fail"
  fi

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$trial" "$lambda_schedule" "$sw_batch" "$accum" "$status" "${steps:-}" \
    "${elapsed_sec:-}" "$sec_per_step" "$train_log" "$time_log" >> "$CSV_PATH"

  echo "[DONE] $trial status=$status steps=${steps:-NA} sec/step=${sec_per_step:-NA}"
}

# 1) 3x batch size, fixed lambda
run_trial "three_x_batch_fixed_lambda" "fixed" "$SW_BATCH_3X" "$ACCUM_3X"

# 2) original batch size (effective ~= 64), fixed lambda
run_trial "original_batch_fixed_lambda" "fixed" "$SW_BATCH_ORIG" "$ACCUM_ORIG"

# 3) 3x batch size, variable lambda schedule
run_trial "three_x_batch_time_lambda" "time_dependent" "$SW_BATCH_3X" "$ACCUM_3X"

echo
echo "Sweep complete."
echo "CSV: $CSV_PATH"
cat "$CSV_PATH"
