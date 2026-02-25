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

    ## VALIDATION
    parser.add_argument("--val_interval_epoch", type=int, default=cfg.val_interval_epoch)

    args = parser.parse_args()
    return args
