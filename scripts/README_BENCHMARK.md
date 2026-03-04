# Runtime Benchmarking Workflow

This benchmark is designed to tune throughput without changing baseline training semantics.

## What it does

`scripts/benchmark_runtime.sh` runs multiple short `train.py` trials sequentially in one allocation, then writes a CSV summary with:

- `sec_per_step`
- `sec_per_microbatch`
- `slices_per_sec`
- average GPU utilization
- startup environment self-check report (`python_env_check.txt`)

Default sweep:

- `num_workers = 8, 16, 24`
- fixed lambda schedule
- precision 32
- `sw_batch=32`, `accumulate_grad_batches=2`

Optionally, it runs a `torch.compile` trial on the best worker setting.

## Run inside an active allocation

```bash
bash scripts/benchmark_runtime.sh
```

## Run via Slurm

```bash
sbatch scripts/benchmark_runtime.slurm
```

## Key environment overrides

```bash
export MAX_STEPS=500
export NUM_WORKERS_LIST="8 12 16 24"
export PROFILE_COMPILE=1
export TORCH_COMPILE_MODE=default
export LOG_INTERVAL=100
export OUT_DIR=/path/to/benchmark_outputs
# Optional: force specific GPU id/UUID for utilization monitoring.
export GPU_MON_ID=0
```

## Output

CSV path:

`benchmarks/runtime_<timestamp>/summary.csv`

Env check path:

`benchmarks/runtime_<timestamp>/python_env_check.txt`

Use the row with lowest `sec_per_step` among `status=ok` trials for your next confirmation run.
