import argparse


def load_args(cfg):
    parser = argparse.ArgumentParser(description='TaDiff-Net Training')

    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--gpu_devices", type=str, default=cfg.gpu_devices)

    ## MODE
    parser.add_argument('--do_train_only', default=False, action='store_true')
    parser.add_argument('--do_test_only', default=False, action='store_true')
    parser.add_argument('--resume_from_ckpt', default=False, action='store_true',
                        help='Resume training from last checkpoint')

    ## TRAIN
    parser.add_argument("--max_epochs", type=int, default=cfg.max_epochs)
    parser.add_argument("--max_steps", type=int, default=cfg.max_steps)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--sw_batch", type=int, default=cfg.sw_batch,
                        help='Number of 2D slices per batch (default: 16)')
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)

    parser.add_argument("--precision", type=str, default=cfg.precision)  # 16-mixed, 32
    parser.add_argument("--accumulate_grad_batches", type=int, default=cfg.accumulate_grad_batches)
    parser.add_argument("--gpu_strategy", type=str, default=cfg.gpu_strategy,
                        help='Training strategy: auto, ddp, etc.')
    parser.add_argument("--log_interval", type=int, default=cfg.log_interval,
                        help='Log metrics every N optimizer steps')
    parser.add_argument("--enable_progress_bar", default=cfg.enable_progress_bar, action='store_true',
                        help='Enable Rich progress bar output')
    parser.add_argument("--use_torch_compile", default=cfg.use_torch_compile, action='store_true',
                        help='Enable torch.compile optimization (PyTorch 2.x)')
    parser.add_argument("--torch_compile_mode", type=str, default=cfg.torch_compile_mode,
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help='Compilation mode passed to torch.compile')

    ## VALIDATION
    parser.add_argument("--val_interval_epoch", type=int, default=cfg.val_interval_epoch)

    ## LAMBDA SCHEDULE
    parser.add_argument("--lambda_schedule", type=str, default=cfg.lambda_schedule,
                        choices=["fixed", "time_dependent"],
                        help='Lambda schedule for aux_loss: "fixed" (constant 0.01) or "time_dependent" (lambda(t) = 0.01 * alphabar_t^2)')

    args = parser.parse_args()
    return args
