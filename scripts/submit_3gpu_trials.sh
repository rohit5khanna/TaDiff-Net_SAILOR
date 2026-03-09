#!/usr/bin/env bash
set -euo pipefail

# Submit all three 3-GPU trials as separate jobs.
# Optional override:
#   MAX_STEPS=20000 bash scripts/submit_3gpu_trials.sh

MAX_STEPS="${MAX_STEPS:-5000}"

submit_one() {
  local name="$1"
  sbatch --export=ALL,MAX_STEPS="$MAX_STEPS",TRIAL_NAME="$name" scripts/run_3gpu_trial.slurm
}

echo "[SUBMIT] three_x_batch_fixed_lambda"
submit_one "three_x_batch_fixed_lambda"

echo "[SUBMIT] original_batch_fixed_lambda"
submit_one "original_batch_fixed_lambda"

echo "[SUBMIT] three_x_batch_time_lambda"
submit_one "three_x_batch_time_lambda"

echo
echo "Submitted 3 separate jobs with MAX_STEPS=$MAX_STEPS"
