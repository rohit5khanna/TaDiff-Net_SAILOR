"""
Training DataModule for TaDiff-Net.

This module handles loading preprocessed SAILOR .npy data and sampling
2D slices for training the 2D TaDiff diffusion model.

Data flow:
  1. Load 3D .npy files per patient (image, label, days, treatment)
  2. Sample random 2D axial slices from 3D volumes
  3. Construct 4-session windows from available timepoints
  4. Output batches matching what get_loss() expects:
       - image:      (B, 4, 3, H, W)   4 sessions, 3 modalities (T1, T1c, FLAIR)
       - label:      (B, 4, H, W)      4 sessions, binary tumor masks
       - days:       (B, 4)            cumulative days for each session
       - treatments: (B, 4)            treatment code for each session

Author: Rohit Khanna (Cornell ORIE)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from typing import List, Optional, Dict, Tuple


class TaDiffSliceDataset(Dataset):
    """
    Dataset that loads 3D patient volumes and returns random 2D slices.

    Each __getitem__ call:
      1. Picks a random patient
      2. Picks a random 4-session window from that patient's timepoints
      3. Picks a random axial slice (z-index) that contains tumor
      4. Returns the 2D data in the format expected by get_loss()

    Args:
        data_dir:       Path to directory containing preprocessed .npy files
        patient_ids:    List of patient IDs to include (e.g., ['sub-01', 'sub-02', ...])
        n_repeat:       How many times to "repeat" the dataset per epoch (default: 10)
                        This artificially inflates epoch length for better LR scheduling.
        min_tumor_voxels: Minimum tumor voxels in a slice to be considered valid (default: 10)
        image_size:     Expected spatial size (H, W) for padding/cropping (default: 192)
        augment:        Whether to apply data augmentation (default: False for now)
    """

    def __init__(
        self,
        data_dir: str,
        patient_ids: List[str],
        n_repeat: int = 10,
        min_tumor_voxels: int = 10,
        image_size: int = 192,
        augment: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.n_repeat = n_repeat
        self.min_tumor_voxels = min_tumor_voxels
        self.image_size = image_size
        self.augment = augment

        # Load all patient data into memory
        # For SAILOR (25 patients, ~240x240x155 per volume), this is feasible
        self.patients = []
        self.valid_slices = []  # List of (patient_idx, slice_idx) tuples

        print(f"Loading {len(patient_ids)} patients from {data_dir}...")
        for pid in patient_ids:
            patient_data = self._load_patient(pid)
            if patient_data is not None:
                patient_idx = len(self.patients)
                self.patients.append(patient_data)

                # Pre-compute valid slice indices (slices with tumor)
                # label shape: (T, H, W, D)
                label = patient_data['label']
                # Sum across all sessions, H, W to get tumor volume per slice
                tumor_per_slice = np.sum(label > 0, axis=(0, 1, 2))  # shape (D,)
                valid_z = np.where(tumor_per_slice >= min_tumor_voxels)[0]

                for z_idx in valid_z:
                    self.valid_slices.append((patient_idx, int(z_idx)))

                n_sessions = label.shape[0]
                print(f"  {pid}: {n_sessions} sessions, {len(valid_z)} valid slices, "
                      f"volume shape {label.shape}")

        print(f"Total: {len(self.patients)} patients, {len(self.valid_slices)} valid slices")
        print(f"Effective dataset size: {len(self)} (with n_repeat={n_repeat})")

    def _load_patient(self, patient_id: str) -> Optional[Dict]:
        """
        Load a single patient's preprocessed .npy files.

        Returns dict with keys: image, label, days, treatment, patient_id
        or None if files are missing/corrupted.
        """
        try:
            image_path = os.path.join(self.data_dir, f'{patient_id}_image.npy')
            label_path = os.path.join(self.data_dir, f'{patient_id}_label.npy')
            days_path = os.path.join(self.data_dir, f'{patient_id}_days.npy')
            treat_path = os.path.join(self.data_dir, f'{patient_id}_treatment.npy')

            # Check all files exist
            for path in [image_path, label_path, days_path, treat_path]:
                if not os.path.exists(path):
                    print(f"  WARNING: Missing file {path}, skipping {patient_id}")
                    return None

            image = np.load(image_path)    # (M*T, H, W, D) where M=4 modalities
            label = np.load(label_path)    # (T, H, W, D)
            days = np.load(days_path)      # (T,)
            treatment = np.load(treat_path)  # (T,)

            n_sessions = label.shape[0]
            n_modalities = 4  # T1, T1c, FLAIR, T2

            # Reshape image: (M*T, H, W, D) → (T, M, H, W, D)
            assert image.shape[0] == n_modalities * n_sessions, \
                f"Image shape mismatch for {patient_id}: {image.shape[0]} != {n_modalities}*{n_sessions}"
            image = image.reshape(n_modalities, n_sessions, *image.shape[1:])  # (M, T, H, W, D)
            image = np.transpose(image, (1, 0, 2, 3, 4))  # (T, M, H, W, D)

            # Drop T2 (index 3), keep T1, T1c, FLAIR (indices 0, 1, 2)
            # The model only uses 3 modalities as input per the paper
            image = image[:, :3, :, :, :]  # (T, 3, H, W, D)

            # Validate
            if n_sessions < 2:
                print(f"  WARNING: {patient_id} has only {n_sessions} session(s), skipping")
                return None

            return {
                'image': image.astype(np.float32),
                'label': (label > 0).astype(np.float32),  # binary mask
                'days': days.astype(np.float32),
                'treatment': treatment.astype(np.float32),
                'patient_id': patient_id,
                'n_sessions': n_sessions,
            }

        except Exception as e:
            print(f"  ERROR loading {patient_id}: {e}")
            return None

    def __len__(self) -> int:
        """
        Dataset length = number of valid slices * n_repeat.

        n_repeat inflates the epoch so that:
          - The LR scheduler sees enough steps per epoch
          - Different random 4-session windows are sampled each time
        """
        return len(self.valid_slices) * self.n_repeat

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Sample a single training example.

        Steps:
          1. Map idx to a (patient, slice) pair
          2. Sample a random 4-session window
          3. Extract the 2D slice
          4. Return formatted tensors
        """
        # Map repeated index back to valid slice index
        real_idx = idx % len(self.valid_slices)
        patient_idx, z_idx = self.valid_slices[real_idx]
        patient = self.patients[patient_idx]

        n_sessions = patient['n_sessions']

        # --- Step 2: Sample a 4-session window ---
        # The model always expects 4 sessions as input:
        #   sessions[0:3] = conditioning (past 3 time points)
        #   sessions[3]   = target (to be predicted)
        #
        # If patient has >= 4 sessions, randomly pick a contiguous window.
        # If patient has < 4 sessions, pad by repeating the first session.

        if n_sessions >= 4:
            # Random start index for 4-session window
            max_start = n_sessions - 4
            start = np.random.randint(0, max_start + 1)
            session_indices = list(range(start, start + 4))
        elif n_sessions == 3:
            # Pad: repeat first session, then use all 3
            session_indices = [0, 0, 1, 2]
        elif n_sessions == 2:
            # Pad: repeat first session twice
            session_indices = [0, 0, 0, 1]
        else:
            # Should not happen (filtered in _load_patient), but just in case
            session_indices = [0, 0, 0, 0]

        # --- Step 3: Extract 2D slice ---
        # image shape: (T, 3, H, W, D)
        # label shape: (T, H, W, D)

        # Extract sessions and slice
        img_slice = patient['image'][session_indices, :, :, :, z_idx]  # (4, 3, H, W)
        lbl_slice = patient['label'][session_indices, :, :, z_idx]     # (4, H, W)
        day_values = patient['days'][session_indices]                    # (4,)
        treat_values = patient['treatment'][session_indices]             # (4,)

        # --- Step 4: Center crop / pad to image_size ---
        h, w = img_slice.shape[2], img_slice.shape[3]
        target_h, target_w = self.image_size, self.image_size

        if h > target_h or w > target_w:
            # Center crop
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            img_slice = img_slice[:, :, start_h:start_h+target_h, start_w:start_w+target_w]
            lbl_slice = lbl_slice[:, start_h:start_h+target_h, start_w:start_w+target_w]
        elif h < target_h or w < target_w:
            # Pad with zeros
            pad_h = target_h - h
            pad_w = target_w - w
            img_slice = np.pad(img_slice,
                             ((0,0), (0,0), (pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2)),
                             mode='constant', constant_values=0)
            lbl_slice = np.pad(lbl_slice,
                             ((0,0), (pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2)),
                             mode='constant', constant_values=0)

        # Convert to tensors
        return {
            'image': torch.from_numpy(img_slice),        # (4, 3, H, W)
            'label': torch.from_numpy(lbl_slice),        # (4, H, W)
            'days': torch.from_numpy(day_values),         # (4,)
            'treatments': torch.from_numpy(treat_values), # (4,)
        }


class TaDiffDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for TaDiff training.

    This wraps TaDiffSliceDataset and provides train/val dataloaders
    that the Lightning Trainer expects.

    Usage in train.py:
        from src.data.train_data_module import TaDiffDataModule

        dm = TaDiffDataModule(config)
        trainer.fit(model, datamodule=dm)

    Args:
        config: The Munch config object from cfg_tadiff_net.py
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Called by Lightning before training/validation begins.
        Creates train and val datasets.
        """
        data_dir = self.cfg.data_dir.get('sailor', './data/sailor')

        if stage == 'fit' or stage is None:
            # Training dataset
            self.train_dataset = TaDiffSliceDataset(
                data_dir=data_dir,
                patient_ids=self.cfg.train_patients,
                n_repeat=self.cfg.n_repeat_tr,
                image_size=self.cfg.image_size,
                augment=True,
            )

            # Validation dataset
            self.val_dataset = TaDiffSliceDataset(
                data_dir=data_dir,
                patient_ids=self.cfg.val_patients,
                n_repeat=self.cfg.n_repeat_val,
                image_size=self.cfg.image_size,
                augment=False,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader.

        batch_size here refers to the number of 2D slices per batch.
        The paper says effective batch = 64, achieved via:
          batch_size_per_gpu * n_gpus * accumulate_grad_batches

        With default config: sw_batch=16, 1 GPU, accum=2 → effective = 32
        Adjust based on your GPU memory.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.sw_batch,  # 16 slices per batch
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,  # avoid small last batch issues with DDP
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.sw_batch,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )
