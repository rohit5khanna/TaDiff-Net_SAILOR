"""
TaDiff-Net Training Script

This script trains the TaDiff diffusion model for longitudinal MRI generation
and tumor growth prediction, using PyTorch Lightning.

Training follows the paper's specifications (Section III-B):
  - Optimizer: AdamW with lr=2.5e-4, weight_decay=3e-5
  - LR schedule: Warmup (1000 steps) + cosine decay
  - Diffusion: T=1000, linear beta schedule (1e-4 to 2e-2)
  - Joint loss: L = L_diffusion + 0.01 * L_seg  (Eq. 16)
  - Gradient clipping: max_norm=1.5

Usage:
    # Train with default config
    python train.py

    # Train with custom settings
    python train.py --lr 2.5e-4 --max_steps 5000000 --gpu_devices '0'

    # Resume from checkpoint
    python train.py --resume_from_ckpt

    # Train on multiple GPUs
    python train.py --gpu_devices '0, 1' --gpu_strategy ddp

Example (single GPU, quick test):
    python train.py --max_steps 1000 --val_interval_epoch 5

Author: Rohit Khanna (Cornell ORIE)
"""

import os
import sys
import torch
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from config.cfg_tadiff_net import config as cfg
from config.arg_parse import load_args
from src.tadiff_model import Tadiff_model, MyCallback
from src.data.train_data_module import TaDiffDataModule


def main():
    """
    Main training function.

    Steps:
      1. Parse arguments and merge with config
      2. Set random seed for reproducibility
      3. Create DataModule (loads data, creates train/val dataloaders)
      4. Create model
      5. Configure callbacks (checkpointing, LR monitoring, validation viz)
      6. Create Lightning Trainer
      7. Start training (with optional checkpoint resumption)
    """

    # ---- Step 1: Parse CLI args and merge with config ----
    args = load_args(cfg)
    # Override config with any CLI arguments
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            cfg[key] = value

    print("=" * 60)
    print("TaDiff-Net Training")
    print("=" * 60)
    print(f"Model channels: {cfg.model_channels}")
    print(f"Channel mult:   {cfg.channel_mult}")
    print(f"Learning rate:  {cfg.lr}")
    print(f"Max steps:      {cfg.max_steps}")
    print(f"Warmup steps:   {cfg.warmup_steps}")
    print(f"Batch size:     {cfg.sw_batch} (per GPU)")
    print(f"Grad accum:     {cfg.accumulate_grad_batches}")
    print(f"GPU devices:    {cfg.gpu_devices}")
    print(f"Precision:      {cfg.precision}")
    print(f"aux_loss_w:     {cfg.aux_loss_w}")
    print("=" * 60)

    # ---- Step 2: Set seed ----
    seed_everything(cfg.seed, workers=True)

    # ---- Step 3: Create DataModule ----
    # This loads all patient .npy files and creates slice-based datasets
    dm = TaDiffDataModule(cfg)

    # ---- Step 4: Create model ----
    model = Tadiff_model(cfg)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)

    # ---- Step 5: Configure callbacks ----
    callbacks = []

    # 5a. ModelCheckpoint: saves best + last checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.logdir,
        filename=cfg.ckpt_filename,
        monitor=cfg.ckpt_monitor,      # "val_loss"
        mode=cfg.ckpt_mode,            # "min"
        save_top_k=cfg.ckpt_save_top_k,  # keep top 3
        save_last=cfg.ckpt_save_last,    # always save last
        verbose=True,
        every_n_epochs=cfg.val_interval_epoch,  # only check when validation runs
    )
    callbacks.append(checkpoint_callback)

    # 5b. LearningRateMonitor: logs LR to the logger
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # 5c. Progress bar
    callbacks.append(RichProgressBar())

    # 5d. MyCallback for validation visualization
    # NOTE: MyCallback requires a sample batch to initialize.
    # We set it up after the datamodule is ready.
    # It will be added during training via a hook or manually.

    # ---- Step 6: Configure logger ----
    # Try wandb first, fall back to TensorBoard
    try:
        logger = WandbLogger(
            project=f"TaDiff_{cfg.network}",
            entity=cfg.wandb_entity,
            save_dir=cfg.logdir,
            log_model=False,  # don't upload checkpoints to wandb
        )
        print("Using Weights & Biases logger")
    except Exception as e:
        print(f"WandB not available ({e}), falling back to TensorBoard")
        logger = TensorBoardLogger(
            save_dir=cfg.logdir,
            name="tadiff_logs",
        )

    # ---- Step 7: Configure GPU devices ----
    if isinstance(cfg.gpu_devices, str):
        # Parse "0, 1" → [0, 1]
        devices = [int(d.strip()) for d in cfg.gpu_devices.split(',') if d.strip()]
    else:
        devices = cfg.gpu_devices

    # Determine strategy
    strategy = cfg.gpu_strategy
    if isinstance(devices, list) and len(devices) == 1:
        strategy = "auto"  # single GPU doesn't need DDP

    # ---- Step 8: Create Trainer ----
    trainer_kwargs = dict(
        max_epochs=cfg.max_epochs if cfg.max_epochs > 0 else None,
        max_steps=cfg.max_steps if cfg.max_epochs <= 0 else -1,
        accelerator=cfg.gpu_accelerator,
        devices=devices,
        strategy=strategy,
        precision=cfg.precision,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gradient_clip_val=cfg.grad_clip,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.log_interval,
        check_val_every_n_epoch=cfg.val_interval_epoch,
        deterministic=False,  # True would be slower
        enable_progress_bar=True,
    )

    # Handle checkpoint resumption
    ckpt_path = None
    if cfg.resume_from_ckpt:
        # Look for 'last.ckpt' in the logdir
        last_ckpt = os.path.join(cfg.logdir, 'last.ckpt')
        if os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
            print(f"Resuming from checkpoint: {ckpt_path}")
        elif cfg.ckpt_best_or_last is not None:
            ckpt_path = cfg.ckpt_best_or_last
            print(f"Resuming from checkpoint: {ckpt_path}")
        else:
            print("WARNING: resume_from_ckpt=True but no checkpoint found. Training from scratch.")

    trainer = Trainer(**trainer_kwargs)

    # ---- Step 9: Start training ----
    print("\nStarting training...")
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    # ---- Step 10: Print final results ----
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score}")
    print("=" * 60)


if __name__ == '__main__':
    main()
