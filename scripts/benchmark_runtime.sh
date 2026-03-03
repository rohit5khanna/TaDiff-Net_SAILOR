#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/benchmarks/runtime_${DATE_TAG}}"
MAX_STEPS="${MAX_STEPS:-400}"
NUM_WORKERS_LIST="${NUM_WORKERS_LIST:-8 16 24}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
LAMBDA_SCHEDULE="${LAMBDA_SCHEDULE:-fixed}"
PRECISION="${PRECISION:-32}"
SW_BATCH="${SW_BATCH:-32}"
ACCUM="${ACCUM:-2}"
VAL_INTERVAL_EPOCH="${VAL_INTERVAL_EPOCH:-1000}"
PROFILE_COMPILE="${PROFILE_COMPILE:-0}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-default}"
ENABLE_PROGRESS_BAR="${ENABLE_PROGRESS_BAR:-0}"
GPU_MON_INTERVAL_SEC="${GPU_MON_INTERVAL_SEC:-1}"

# Runtime-only knobs: safe for paper-faithful optimization.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "$OUT_DIR"
CSV_PATH="$OUT_DIR/summary.csv"

cat > "$CSV_PATH" <<'CSV'
trial,num_workers,use_torch_compile,status,steps,elapsed_sec,sec_per_step,sec_per_microbatch,slices_per_sec,avg_gpu_util,gpu_util_samples,train_log,time_log,gpu_log
CSV

{
  echo "date=$(date)"
  echo "root_dir=$ROOT_DIR"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "max_steps=$MAX_STEPS"
  echo "num_workers_list=$NUM_WORKERS_LIST"
  echo "profile_compile=$PROFILE_COMPILE"
  echo "torch_compile_mode=$TORCH_COMPILE_MODE"
  echo "precision=$PRECISION"
  echo "sw_batch=$SW_BATCH"
  echo "accumulate_grad_batches=$ACCUM"
  echo "lambda_schedule=$LAMBDA_SCHEDULE"
  echo "log_interval=$LOG_INTERVAL"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "mkl_num_threads=$MKL_NUM_THREADS"
  echo "numexpr_num_threads=$NUMEXPR_NUM_THREADS"
  echo "wandb_mode=$WANDB_MODE"
} > "$OUT_DIR/env.txt"

run_trial() {
  local num_workers="$1"
  local use_compile="$2"
  local trial_name="$3"
  local trial_dir="$OUT_DIR/$trial_name"
  mkdir -p "$trial_dir"

  local train_log="$trial_dir/train.log"
  local time_log="$trial_dir/time.log"
  local gpu_log="$trial_dir/gpu_util.log"

  local -a cmd=(
    python train.py
    --lambda_schedule "$LAMBDA_SCHEDULE"
    --max_steps "$MAX_STEPS"
    --num_workers "$num_workers"
    --log_interval "$LOG_INTERVAL"
    --precision "$PRECISION"
    --sw_batch "$SW_BATCH"
    --accumulate_grad_batches "$ACCUM"
    --val_interval_epoch "$VAL_INTERVAL_EPOCH"
  )

  if [[ "$use_compile" == "1" ]]; then
    cmd+=(--use_torch_compile --torch_compile_mode "$TORCH_COMPILE_MODE")
  fi

  if [[ "$ENABLE_PROGRESS_BAR" == "1" ]]; then
    cmd+=(--enable_progress_bar)
  fi

  printf "%q " "${cmd[@]}" > "$trial_dir/cmd.shline"
  echo >> "$trial_dir/cmd.shline"

  echo "[RUN] $trial_name"
  echo "      $(cat "$trial_dir/cmd.shline")"

  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l "$GPU_MON_INTERVAL_SEC" > "$gpu_log" 2>/dev/null &
  local gpu_pid=$!

  local start_ts
  start_ts="$(date +%s)"
  set +e
  /usr/bin/time -p "${cmd[@]}" > "$train_log" 2> "$time_log"
  local rc=$?
  set -e
  local end_ts
  end_ts="$(date +%s)"

  kill "$gpu_pid" >/dev/null 2>&1 || true
  wait "$gpu_pid" 2>/dev/null || true

  local elapsed_sec
  elapsed_sec="$(awk '/^real /{print $2}' "$time_log" | tail -n 1)"
  if [[ -z "${elapsed_sec:-}" ]]; then
    elapsed_sec="$((end_ts - start_ts))"
  fi

  local steps
  steps="$(awk -F': ' '/trainer\/global_step/{gsub(/[^0-9]/, "", $NF); if($NF!="") s=$NF} END{print s}' "$train_log")"
  if [[ -z "${steps:-}" && "$rc" -eq 0 ]]; then
    steps="$MAX_STEPS"
  fi

  local sec_per_step=""
  local sec_per_microbatch=""
  local slices_per_sec=""
  if [[ -n "${steps:-}" && "$steps" -gt 0 ]]; then
    sec_per_step="$(awk -v e="$elapsed_sec" -v s="$steps" 'BEGIN{printf "%.6f", e/s}')"
    sec_per_microbatch="$(awk -v e="$elapsed_sec" -v s="$steps" -v a="$ACCUM" 'BEGIN{printf "%.6f", e/(s*a)}')"
    slices_per_sec="$(awk -v sb="$SW_BATCH" -v a="$ACCUM" -v sps="$sec_per_step" 'BEGIN{printf "%.6f", (sb*a)/sps}')"
  fi

  local avg_gpu_util=""
  local gpu_samples=""
  avg_gpu_util="$(awk '/^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$/ {sum+=$1; n++} END{if(n>0) printf "%.2f", sum/n}' "$gpu_log")"
  gpu_samples="$(awk '/^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$/ {n++} END{print n+0}' "$gpu_log")"

  local status="ok"
  if [[ "$rc" -ne 0 ]]; then
    status="fail"
  fi

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$trial_name" "$num_workers" "$use_compile" "$status" "${steps:-}" "$elapsed_sec" \
    "$sec_per_step" "$sec_per_microbatch" "$slices_per_sec" "$avg_gpu_util" "$gpu_samples" \
    "$train_log" "$time_log" "$gpu_log" >> "$CSV_PATH"

  echo "[DONE] $trial_name status=$status elapsed=${elapsed_sec}s steps=${steps:-NA} sec/step=${sec_per_step:-NA} avg_gpu_util=${avg_gpu_util:-NA}%"
}

for nw in $NUM_WORKERS_LIST; do
  run_trial "$nw" "0" "nw${nw}_compile0"
done

best_worker="$(awk -F, 'NR>1 && $3=="0" && $4=="ok" && $7!="" {val=$7+0; if(min=="" || val<min){min=val; w=$2}} END{print w}' "$CSV_PATH")"

if [[ "$PROFILE_COMPILE" == "1" && -n "${best_worker:-}" ]]; then
  run_trial "$best_worker" "1" "nw${best_worker}_compile1"
fi

echo
echo "Benchmark complete."
echo "CSV: $CSV_PATH"
echo
awk -F, '
  NR==1 {print; next}
  {print}
' "$CSV_PATH"

