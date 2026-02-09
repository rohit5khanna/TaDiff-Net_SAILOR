"""
SAILOR Dataset Preprocessing for TaDiff-Net
============================================

Preprocesses raw SAILOR (brain tumor MRI) NIfTI data into the .npy format
expected by TaDiff-Net for training.

This is a fixed version of the original preproc_prepare_data.py from the
TaDiff-Net repo. Changes from the original:
  1. Removed dependency on missing `reorient_nii` package — replaced with
     nibabel's built-in `as_closest_canonical` + `ornt_transform` for
     reorientation to PLI axes.
  2. Fixed double-path-join bug in `get_session_list()` (line 143 of original
     joined sailor_raw_path twice, producing an invalid path).
  3. Added 1mm isotropic resampling via MONAI's `Spacing` transform, applied
     correctly on the NIfTI object (with affine) *before* extracting to numpy.
  4. Added CLI arguments so paths don't need to be hardcoded.
  5. Added per-patient validation checks after saving.

All preprocessing *operations* (z-score normalization, 0.2% outlier clipping,
mask merging with enhancing-tumor priority, treatment encoding) are identical
to the original.

Usage:
    python preprocess_sailor.py \
        --raw_dir  /path/to/sailor_raw \
        --out_dir  /path/to/output/sailor_npy \
        --csv_file /path/to/sailor_raw/sailor_info.csv

    If --csv_file is omitted it defaults to <raw_dir>/sailor_info.csv.
    If --out_dir  is omitted it defaults to ./sailor_npy.

Requirements:
    pip install numpy pandas nibabel monai torch

Author: Rohit Khanna (adapted from Qinghui Liu / TaDiff-Net)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from monai.transforms import Spacing

# ─────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────

# Segmentation label values
BG = 0         # background
EDEMA = 1      # peritumoral edema
NECROTIC = 2   # necrotic core (reserved, not used in SAILOR)
ENHANCING = 3  # enhancing tumor

# Expected NIfTI filenames inside each session folder
KEY_FILENAMES = {
    "edema_mask": "EdemaMask-CL.nii.gz",
    "et_mask":    "ContrastEnhancedMask-CL.nii.gz",
    "t1":         "T1.nii.gz",
    "t1c":        "T1c.nii.gz",
    "flair":      "Flair.nii.gz",
    "t2":         "T2.nii.gz",
}

# All 27 patients in the raw SAILOR dataset
ALL_PATIENT_IDS = [f"sub-{i:02d}" for i in range(1, 28)]

# 25 patients that passed quality-control (sub-24 excluded)
VALID_PATIENT_IDS = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07",
    "sub-08", "sub-09", "sub-10", "sub-11", "sub-12", "sub-13", "sub-14",
    "sub-15", "sub-16", "sub-17", "sub-18", "sub-19", "sub-20", "sub-21",
    "sub-22", "sub-23", "sub-25", "sub-26", "sub-27",
]

# Target voxel spacing in mm (isotropic 1mm)
TARGET_SPACING = (1.0, 1.0, 1.0)

# Target orientation axes code
TARGET_ORIENTATION = "PLI"  # Posterior-Left-Inferior


# ─────────────────────────────────────────────────────────────────────
#  NIfTI I/O helpers
# ─────────────────────────────────────────────────────────────────────

def reorient_nifti(img, target_axcodes="PLI"):
    """
    Reorient a nibabel image to the desired axis-code orientation.

    Uses nibabel's orientation utilities (no external package needed).

    Args:
        img:            nibabel.Nifti1Image
        target_axcodes: 3-character axis-code string, e.g. "PLI", "RAS"

    Returns:
        nibabel.Nifti1Image in the requested orientation
    """
    current_ornt = nib.orientations.io_orientation(img.affine)
    target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
    transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
    return img.as_reoriented(transform)


def resample_to_1mm(img, mode="bilinear"):
    """
    Resample a nibabel image to 1×1×1 mm isotropic spacing using MONAI.

    Correctly passes the affine so MONAI knows the source voxel sizes.

    Args:
        img:  nibabel.Nifti1Image (already reoriented)
        mode: interpolation mode — "bilinear" for images, "nearest" for masks

    Returns:
        np.ndarray of the resampled volume (H, W, D)
    """
    data = img.get_fdata().astype(np.float32)
    affine = img.affine

    # MONAI Spacing expects a channel-first tensor: (C, H, W, D)
    data_tensor = torch.from_numpy(data).unsqueeze(0)

    spacer = Spacing(pixdim=TARGET_SPACING, mode=mode)
    resampled = spacer(data_tensor, affine=affine)

    return resampled.squeeze(0).numpy()


def read_nii(path, orientation_axcode=TARGET_ORIENTATION,
             non_zero_norm=False, clip_percent=0.1,
             resample=True, is_mask=False):
    """
    Load a NIfTI file, reorient, resample to 1mm, and optionally normalize.

    Args:
        path:               str  – path to .nii.gz file
        orientation_axcode: str  – target orientation (default "PLI")
        non_zero_norm:      bool – z-score normalize non-zero voxels then scale to [0,1]
        clip_percent:       float – percentile for outlier clipping (0–0.5)
        resample:           bool – resample to 1mm isotropic
        is_mask:            bool – if True, use nearest-neighbor interpolation
                                   during resampling (preserves integer labels)

    Returns:
        np.ndarray (H, W, D)
    """
    img = nib.load(path)
    img = reorient_nifti(img, orientation_axcode)

    if resample:
        mode = "nearest" if is_mask else "bilinear"
        img_data = resample_to_1mm(img, mode=mode)
    else:
        img_data = img.get_fdata().astype(np.float32)

    if non_zero_norm:
        img_data = nonzero_norm_image(img_data, clip_percent=clip_percent)

    return img_data


# ─────────────────────────────────────────────────────────────────────
#  Intensity normalization (identical to original)
# ─────────────────────────────────────────────────────────────────────

def nonzero_norm_image(image, clip_percent=0.1):
    """
    Normalize image intensities using only non-zero voxels.

    Steps:
        1. Clip outlier intensities at the given percentiles
        2. Z-score normalize (mean=0, std=1) on non-zero voxels
        3. Min-max scale entire image to [0, 1]

    Args:
        image:        np.ndarray – 3-D volume
        clip_percent: float      – bottom/top percentile to clip (default 0.1)

    Returns:
        np.ndarray in [0, 1]
    """
    assert 0 <= clip_percent <= 0.5, (
        f"clip_percent must be in [0, 0.5], got {clip_percent}"
    )

    nz_mask = image > 0

    if image[nz_mask].size == 0:
        print(f"  WARNING: image has no non-zero values. "
              f"shape={image.shape}, min={image.min():.4f}, max={image.max():.4f}")
        return image

    # 1. Clip outliers
    if clip_percent > 0:
        lo = np.percentile(image[nz_mask], clip_percent)
        hi = np.percentile(image[nz_mask], 100 - clip_percent)
        image[nz_mask & (image < lo)] = lo
        image[nz_mask & (image > hi)] = hi

    # 2. Z-score on non-zero voxels
    nz_vals = image[nz_mask]
    mu = np.mean(nz_vals)
    sigma = np.std(nz_vals)
    assert sigma != 0.0, f"Image std is zero (constant-valued image)"
    image = (image - mu) / sigma

    # 3. Scale to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    return image


# ─────────────────────────────────────────────────────────────────────
#  Metadata loading
# ─────────────────────────────────────────────────────────────────────

def get_session_list(csv_path):
    """
    Parse sailor_info.csv to get per-patient interval days and treatment arrays.

    Args:
        csv_path: str – absolute path to sailor_info.csv

    Returns:
        (interval_dict, treatment_dict) where each maps patient_id to np.array
            interval_dict:  [0, d1, d2, ...] raw intervals with 0 prepended
            treatment_dict: [0,0,0,0,1,1,...] 0=CRT for sessions 0-3, 1=TMZ after
    """
    df = pd.read_csv(csv_path)
    df.set_index("patients", inplace=True)

    interval_dict = {}
    treatment_dict = {}

    for patient_id in ALL_PATIENT_IDS:
        if patient_id not in df.index:
            print(f"  WARNING: {patient_id} not found in CSV, skipping")
            continue

        raw = df.loc[patient_id, "interval_days"]           # e.g. "[13, 15, 14, 34, 28]"
        intervals = np.array([int(s) for s in raw.strip("[]").split(",")])

        # Prepend 0 for the baseline session
        interval_dict[patient_id] = np.insert(intervals, 0, 0)

        # Treatment: first 4 sessions = CRT (0), rest = TMZ (1)
        n_sessions = len(intervals) + 1
        treatment_dict[patient_id] = np.array(
            [0 if i <= 3 else 1 for i in range(n_sessions)]
        )

    return interval_dict, treatment_dict


# ─────────────────────────────────────────────────────────────────────
#  File discovery
# ─────────────────────────────────────────────────────────────────────

def get_file_dict(patient_ids, raw_dir):
    """
    Build a nested dict of NIfTI paths for every patient and session.

    Args:
        patient_ids: list of str, e.g. ["sub-01", "sub-02", ...]
        raw_dir:     str – root directory containing sub-XX folders

    Returns:
        {patient_id: {session_id: {modality_key: filepath}}}
    """
    file_dict = {}

    for pid in patient_ids:
        patient_path = os.path.join(raw_dir, pid)
        if not os.path.isdir(patient_path):
            print(f"  WARNING: directory not found for {pid}, skipping")
            continue

        # Collect session directories (ses-01, ses-02, ...)
        session_ids = sorted([
            entry.name for entry in os.scandir(patient_path)
            if entry.is_dir() and entry.name.startswith("ses-")
        ])

        if len(session_ids) == 0:
            print(f"  WARNING: no session dirs found for {pid}, skipping")
            continue

        sessions = {}
        for sid in session_ids:
            files = {}
            missing = []
            for key, fname in KEY_FILENAMES.items():
                fpath = os.path.join(patient_path, sid, fname)
                if not os.path.isfile(fpath):
                    missing.append(fname)
                files[key] = fpath

            if missing:
                print(f"  WARNING: {pid}/{sid} missing files: {missing}")

            sessions[sid] = files

        file_dict[pid] = sessions

    return file_dict


# ─────────────────────────────────────────────────────────────────────
#  Main processing
# ─────────────────────────────────────────────────────────────────────

def save_session_data(file_dict, save_path, csv_path, raw_dir):
    """
    Process every patient in file_dict and save four .npy files each.

    Output per patient:
        {pid}_image.npy      (M*T, H, W, D)  float32  – 4 modalities × T sessions
        {pid}_label.npy      (T,   H, W, D)  int8     – merged segmentation masks
        {pid}_days.npy       (T,)             int64    – cumulative days from baseline
        {pid}_treatment.npy  (T,)             int64    – 0=CRT, 1=TMZ

    Args:
        file_dict: from get_file_dict()
        save_path: str – output directory
        csv_path:  str – path to sailor_info.csv
        raw_dir:   str – raw data root (only used for logging)
    """
    os.makedirs(save_path, exist_ok=True)

    interval_dict, treatment_dict = get_session_list(csv_path)
    patient_ids = list(file_dict.keys())

    print(f"\nProcessing {len(patient_ids)} patients …")
    print(f"  Raw data dir : {raw_dir}")
    print(f"  CSV metadata : {csv_path}")
    print(f"  Output dir   : {save_path}\n")

    for patient_id in patient_ids:
        print(f"── {patient_id} ──")

        session_ids = sorted(file_dict[patient_id].keys())
        n_sessions = len(session_ids)
        print(f"   Sessions: {n_sessions}  ({session_ids[0]} … {session_ids[-1]})")

        images = []   # will have 4*T entries
        labels = []   # will have T entries

        modalities = ["t1", "t1c", "flair", "t2"]

        for mod_idx, modality in enumerate(modalities):
            for sess_idx, session_id in enumerate(session_ids):
                img_path = file_dict[patient_id][session_id][modality]
                img = read_nii(img_path, non_zero_norm=True, clip_percent=0.2)
                images.append(img)

                # Build segmentation mask once per session (during T1 pass)
                if mod_idx == 0:
                    merged = np.zeros_like(img)

                    # Edema (label 1)
                    edema_path = file_dict[patient_id][session_id]["edema_mask"]
                    edema = read_nii(edema_path, non_zero_norm=False, is_mask=True)
                    merged[edema > 0] = EDEMA

                    # Enhancing tumor (label 3) overwrites edema
                    et_path = file_dict[patient_id][session_id]["et_mask"]
                    et = read_nii(et_path, non_zero_norm=False, is_mask=True)
                    merged[et > 0] = ENHANCING

                    labels.append(merged.astype(np.int8))

        # Stack into arrays
        image_arr = np.stack(images, axis=0).astype(np.float32)   # (4*T, H, W, D)
        label_arr = np.stack(labels, axis=0).astype(np.int8)      # (T, H, W, D)

        # Cumulative days from baseline
        if patient_id not in interval_dict:
            print(f"   WARNING: {patient_id} not in CSV, skipping days/treatment")
            continue

        day_intervals = interval_dict[patient_id]  # [0, d1, d2, ...]
        days = np.cumsum(day_intervals)             # [0, d1, d1+d2, ...]
        treatment = treatment_dict[patient_id]

        # ── Sanity checks ──────────────────────────────────────────
        assert image_arr.shape[0] == 4 * n_sessions, (
            f"Image stack has {image_arr.shape[0]} slabs but expected "
            f"4 modalities × {n_sessions} sessions = {4*n_sessions}"
        )
        assert label_arr.shape[0] == n_sessions, (
            f"Label stack has {label_arr.shape[0]} but expected {n_sessions}"
        )

        if len(days) != n_sessions:
            csv_n = len(days)
            print(f"   *** MISMATCH: CSV gives {csv_n} time points but "
                  f"found {n_sessions} session dirs on disk ***")
            if csv_n > n_sessions:
                # More intervals than session dirs — truncate to match dirs
                print(f"       → Truncating days/treatment to first {n_sessions} entries")
                days = days[:n_sessions]
                treatment = treatment[:n_sessions]
            else:
                # Fewer intervals than session dirs — CSV is incomplete.
                # This affects 7 patients (sub-05, -08, -12, -15, -20, -22, -25).
                # We can only produce correct days for the sessions covered by
                # the CSV. For the remaining sessions we extrapolate using the
                # mean interval, and mark treatment as TMZ (session index > 3).
                mean_gap = int(np.mean(day_intervals[1:])) if len(day_intervals) > 1 else 90
                extra = n_sessions - csv_n
                print(f"       → CSV only covers {csv_n} of {n_sessions} sessions")
                print(f"       → Extrapolating {extra} sessions with mean interval = {mean_gap} days")
                last_day = days[-1]
                extra_days = np.array([last_day + mean_gap * (k + 1) for k in range(extra)])
                days = np.concatenate([days, extra_days])
                extra_treat = np.array([0 if (csv_n + k) <= 3 else 1 for k in range(extra)])
                treatment = np.concatenate([treatment, extra_treat])

        assert not np.isnan(image_arr).any(), f"NaN in images for {patient_id}"
        assert not np.isinf(image_arr).any(), f"Inf in images for {patient_id}"
        assert image_arr.min() >= 0.0 and image_arr.max() <= 1.0, (
            f"Image range [{image_arr.min():.4f}, {image_arr.max():.4f}] "
            f"outside [0, 1] for {patient_id}"
        )

        # ── Save ────────────────────────────────────────────────────
        np.save(os.path.join(save_path, f"{patient_id}_image.npy"), image_arr)
        np.save(os.path.join(save_path, f"{patient_id}_label.npy"), label_arr)
        np.save(os.path.join(save_path, f"{patient_id}_days.npy"), days)
        np.save(os.path.join(save_path, f"{patient_id}_treatment.npy"), treatment)

        print(f"   image     : {image_arr.shape}  [{image_arr.min():.3f} – {image_arr.max():.3f}]")
        print(f"   label     : {label_arr.shape}  unique={np.unique(label_arr)}")
        print(f"   days      : {days}")
        print(f"   treatment : {treatment}")
        print()

    print(f"Done. Preprocessed data saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess SAILOR MRI data for TaDiff-Net training."
    )
    p.add_argument(
        "--raw_dir", type=str, required=True,
        help="Root directory of the raw SAILOR dataset (contains sub-XX folders "
             "and sailor_info.csv)."
    )
    p.add_argument(
        "--out_dir", type=str, default="./sailor_npy",
        help="Output directory for .npy files (default: ./sailor_npy)."
    )
    p.add_argument(
        "--csv_file", type=str, default=None,
        help="Path to sailor_info.csv. Default: <raw_dir>/sailor_info.csv."
    )
    p.add_argument(
        "--patients", type=str, nargs="*", default=None,
        help="Optional: specific patient IDs to process (e.g. sub-01 sub-17). "
             "Default: all 25 valid patients."
    )
    return p.parse_args()


def main():
    args = parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    csv_path = args.csv_file if args.csv_file else os.path.join(raw_dir, "sailor_info.csv")
    patient_ids = args.patients if args.patients else VALID_PATIENT_IDS

    # Validate inputs
    if not os.path.isdir(raw_dir):
        sys.exit(f"ERROR: raw_dir does not exist: {raw_dir}")
    if not os.path.isfile(csv_path):
        sys.exit(f"ERROR: CSV file not found: {csv_path}")

    print("=" * 60)
    print("  SAILOR Preprocessing for TaDiff-Net")
    print("=" * 60)
    print(f"  Raw data  : {raw_dir}")
    print(f"  CSV file  : {csv_path}")
    print(f"  Output    : {out_dir}")
    print(f"  Patients  : {len(patient_ids)}")
    print("=" * 60)

    # Discover files
    file_dict = get_file_dict(patient_ids, raw_dir)
    if len(file_dict) == 0:
        sys.exit("ERROR: no patient data found. Check --raw_dir path.")

    # Process and save
    save_session_data(file_dict, out_dir, csv_path, raw_dir)


if __name__ == "__main__":
    main()
