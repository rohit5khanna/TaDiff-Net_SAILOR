#!/usr/bin/env bash
set -euo pipefail

# Submit the two production 3-GPU trials agreed for the next run:
# 1) paper-like effective batch (fixed lambda), 5M steps
# 2) paper per-GPU batch (time-dependent lambda), 1.5M steps
#
# Optional overrides:
#   BASELINE_MAX_STEPS=5000000 TIME_LAMBDA_MAX_STEPS=1718750 bash scripts/submit_3gpu_trials.sh

BASELINE_MAX_STEPS="${BASELINE_MAX_STEPS:-5000000}"
TIME_LAMBDA_MAX_STEPS="${TIME_LAMBDA_MAX_STEPS:-1500000}"

submit_one() {
  local name="$1"
  local steps="$2"
  sbatch --export=ALL,MAX_STEPS="$steps",TRIAL_NAME="$name" scripts/run_3gpu_trial.slurm
}

echo "[SUBMIT] original_batch_fixed_lambda (MAX_STEPS=$BASELINE_MAX_STEPS)"
submit_one "original_batch_fixed_lambda" "$BASELINE_MAX_STEPS"

echo "[SUBMIT] three_x_batch_time_lambda (MAX_STEPS=$TIME_LAMBDA_MAX_STEPS)"
submit_one "three_x_batch_time_lambda" "$TIME_LAMBDA_MAX_STEPS"

echo
echo "Submitted 2 separate jobs:"
echo "  - original_batch_fixed_lambda: $BASELINE_MAX_STEPS steps"
echo "  - three_x_batch_time_lambda:   $TIME_LAMBDA_MAX_STEPS steps"
