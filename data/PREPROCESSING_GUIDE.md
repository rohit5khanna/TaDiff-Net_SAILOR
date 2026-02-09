# SAILOR Data Preprocessing Guide for TaDiff-Net

## Overview

This document explains how to run `preprocess_sailor.py` to convert the raw
SAILOR MRI dataset into the `.npy` format that TaDiff-Net expects for training.

This script is a **bug-fixed** version of the original `preproc_prepare_data.py`
from the TaDiff-Net GitHub repo. The preprocessing operations (normalization,
mask merging, treatment encoding) are identical to what the TaDiff authors used.
Only plumbing bugs and the missing resampling step have been addressed.

---

## What changed vs the original `preproc_prepare_data.py`

| # | Issue in original | Fix applied |
|---|---|---|
| 1 | `from reorient_nii import reorient` — package not included in repo and not on PyPI | Replaced with nibabel's built-in `io_orientation` / `ornt_transform` / `as_reoriented`. Achieves the same PLI reorientation, zero external deps. |
| 2 | `get_session_list()` line 143: `pd.read_csv(os.path.join(sailor_raw_path, file_csv))` where `file_csv` already equals `os.path.join(sailor_raw_path, "sailor_info.csv")` — produces an invalid double path like `/data/raw//data/raw/sailor_info.csv` | Function now takes a single absolute `csv_path` argument. No path joining inside. |
| 3 | No 1mm isotropic resampling (paper states scans were "resampled to isotropic 1mm resolution") | Added via MONAI `Spacing(pixdim=(1,1,1))` applied to the NIfTI object *with its affine*, before extracting to numpy. This is the correct way — applying `Spacing` to raw numpy (as in the earlier attempt) silently fails because MONAI doesn't know the source voxel size without the affine. |
| 4 | Paths hardcoded at module level | Replaced with CLI arguments (`--raw_dir`, `--out_dir`, `--csv_file`). |
| 5 | No validation after processing | Added assertion checks for shape consistency, NaN/Inf, value range [0,1], and days/session count mismatches. |
| 6 | `get_file_dict` picks up non-session dirs (e.g. if a patient folder contains other subdirs) | Added `entry.name.startswith("ses-")` filter. |

**Nothing changed** in the actual preprocessing logic:
- Same z-score normalization on non-zero voxels
- Same 0.2% outlier clipping
- Same min-max scaling to [0, 1]
- Same mask merging (edema=1, enhancing=3, enhancing overwrites edema)
- Same treatment encoding (sessions 0-3 = CRT=0, sessions 4+ = TMZ=1)
- Same output array layout (image, label, days, treatment)

---

## Prerequisites

### Python packages

```bash
pip install numpy pandas nibabel monai torch
```

No other packages are needed. Specifically, you do **not** need `reorient_nii`.

### Data

Download the raw SAILOR dataset from EBRAINS:
https://search.kg.ebrains.eu/instances/cae85bcb-8526-442d-b0d8-a866425efff8

After downloading you should have a directory that looks like:

```
sailor_raw/
├── sailor_info.csv
├── sub-01/
│   ├── ses-01/
│   │   ├── T1.nii.gz
│   │   ├── T1c.nii.gz
│   │   ├── Flair.nii.gz
│   │   ├── T2.nii.gz
│   │   ├── EdemaMask-CL.nii.gz
│   │   └── ContrastEnhancedMask-CL.nii.gz
│   ├── ses-02/
│   │   └── ...
│   └── ...
├── sub-02/
│   └── ...
...
└── sub-27/
```

The `sailor_info.csv` file is **included in the TaDiff-Net repo** at
`TaDiff-Net/data/sailor_info.csv`. If your raw data download does not include
it, copy it from the repo into the raw data directory, or pass its location
explicitly with `--csv_file`.

---

## How to run

### Full dataset (all 25 valid patients)

```bash
python preprocess_sailor.py \
    --raw_dir  /path/to/sailor_raw \
    --out_dir  /path/to/sailor_npy
```

### Specific patients only (for testing)

```bash
python preprocess_sailor.py \
    --raw_dir  /path/to/sailor_raw \
    --out_dir  /path/to/sailor_npy \
    --patients sub-17
```

### Custom CSV location

```bash
python preprocess_sailor.py \
    --raw_dir  /path/to/sailor_raw \
    --out_dir  /path/to/sailor_npy \
    --csv_file /some/other/path/sailor_info.csv
```

---

## What the script does step by step

For each patient:

1. **Discover session folders** — scans `<raw_dir>/sub-XX/` for directories
   named `ses-*`, sorted alphabetically. Warns if any expected NIfTI files are
   missing.

2. **For each of 4 modalities (T1, T1c, FLAIR, T2) × each session:**
   - Load the `.nii.gz` file with nibabel
   - **Reorient** to PLI (Posterior-Left-Inferior) axes using nibabel's
     orientation transforms
   - **Resample** to 1×1×1 mm isotropic spacing using MONAI's `Spacing`
     transform (applied with the NIfTI affine so source voxel sizes are known)
   - **Normalize** non-zero voxels:
     - Clip top/bottom 0.2 percentile intensities
     - Z-score normalize (mean=0, std=1) on non-zero voxels only
     - Min-max scale the entire volume to [0, 1]

3. **For each session (once, during the T1 pass):**
   - Load the edema mask → set label=1 where mask > 0
   - Load the enhancing tumor mask → set label=3 where mask > 0
     (this overwrites any edema voxels that overlap with enhancing tumor)

4. **Load metadata from CSV:**
   - Parse inter-session intervals and compute cumulative days from baseline
   - Assign treatment codes: sessions 0–3 = CRT (0), sessions 4+ = TMZ (1)

5. **Validate and save** four `.npy` files per patient.

---

## Output format

For a patient with T sessions, the output is:

| File | Shape | Dtype | Description |
|---|---|---|---|
| `sub-XX_image.npy` | `(4*T, H, W, D)` | float32 | Stacked modalities: first T slabs are T1 (all sessions), next T are T1c, next T are FLAIR, last T are T2. Values in [0, 1]. |
| `sub-XX_label.npy` | `(T, H, W, D)` | int8 | Merged segmentation masks. 0=background, 1=edema, 3=enhancing tumor. |
| `sub-XX_days.npy` | `(T,)` | int64 | Cumulative days from baseline. E.g. `[0, 13, 28, 42, 76, 104]`. |
| `sub-XX_treatment.npy` | `(T,)` | int64 | 0=CRT, 1=TMZ. E.g. `[0, 0, 0, 0, 1, 1]`. |

**Image stacking order:** The image array is stacked modality-first, meaning:
```
index 0          … T-1   : T1 sessions 1…T
index T          … 2T-1  : T1c sessions 1…T
index 2T         … 3T-1  : FLAIR sessions 1…T
index 3T         … 4T-1  : T2 sessions 1…T
```

Example for sub-01 (6 sessions):
- `sub-01_image.npy` → shape `(24, H, W, D)` = 4 modalities × 6 sessions
- `sub-01_label.npy` → shape `(6, H, W, D)`
- `sub-01_days.npy` → `[0, 13, 28, 42, 76, 104]`
- `sub-01_treatment.npy` → `[0, 0, 0, 0, 1, 1]`

---

## Verifying your output

After preprocessing, run this sanity check:

```python
import numpy as np

pid = "sub-17"
out = "./sailor_npy"

img = np.load(f"{out}/{pid}_image.npy")
lbl = np.load(f"{out}/{pid}_label.npy")
days = np.load(f"{out}/{pid}_days.npy")
treat = np.load(f"{out}/{pid}_treatment.npy")

n_sessions = lbl.shape[0]

print(f"Image : {img.shape}  (expect ({4*n_sessions}, H, W, D))")
print(f"Label : {lbl.shape}  (expect ({n_sessions}, H, W, D))")
print(f"Days  : {days}")
print(f"Treat : {treat}")
print(f"Image range : [{img.min():.4f}, {img.max():.4f}]  (expect [0, 1])")
print(f"Label values: {np.unique(lbl)}  (expect subset of [0, 1, 3])")
print(f"Any NaN : {np.isnan(img).any()}")
print(f"Any Inf : {np.isinf(img).any()}")

assert img.shape[0] == 4 * n_sessions
assert img.min() >= 0.0 and img.max() <= 1.0
assert set(np.unique(lbl)).issubset({0, 1, 2, 3})
assert len(days) == n_sessions
assert len(treat) == n_sessions
print("\nAll checks passed.")
```

---

## Runtime expectations

- **Per patient:** 2–10 minutes depending on number of sessions and volume size.
  The 1mm resampling is the slowest step.
- **Full dataset (25 patients, ~225 sessions):** roughly 1.5–4 hours on a
  modern CPU. No GPU needed for preprocessing.
- **Disk space:** expect ~30–60 GB for the full output (varies with spatial
  dimensions after resampling).

---

## Known data issue: incomplete intervals in `sailor_info.csv`

The `interval_days` column in `sailor_info.csv` is **incomplete for 7 of 27
patients**. The number of intervals listed is fewer than the number of session
directories on disk:

| Patient | `num_ses` in CSV | Intervals listed | Sessions implied | Missing |
|---------|-----------------|-----------------|-----------------|---------|
| sub-05  | 5               | 3               | 4               | 1       |
| sub-08  | 7               | 5               | 6               | 1       |
| sub-12  | 10              | 8               | 9               | 1       |
| sub-15  | 19              | 6               | 7               | **12**  |
| sub-20  | 10              | 8               | 9               | 1       |
| sub-22  | 6               | 4               | 5               | 1       |
| sub-25  | 11              | 8               | 9               | 2       |

**How the script handles this:** The script counts session directories on disk
(the ground truth for how many sessions exist). When more sessions exist than
the CSV covers, it extrapolates the missing days using the mean interval from
the known sessions and assigns TMZ (1) as the treatment for any session with
index > 3. A clear warning is printed for each affected patient.

**sub-15 is the most extreme case** — the CSV only covers 7 of its 19 sessions.
The extrapolated days for sessions 8–19 are approximations. If exact treatment
days for sub-15 are available from another source (e.g. the `intervals-days.txt`
file in the raw patient directory), consider updating `sailor_info.csv` before
running preprocessing.

This is a property of the CSV that shipped with the dataset/repo — not a bug
in the preprocessing script.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError` on a `.nii.gz` | Missing modality for that session | The script warns but continues. Check if the session should be excluded. |
| `AssertionError: Image std is zero` | A modality volume is constant-valued (corrupted) | Exclude that patient or session. |
| `WARNING: CSV has N time entries but found M session dirs` | Mismatch between `sailor_info.csv` intervals and actual session folders | The script truncates/pads days to match. Verify the raw data is complete. |
| Output shapes look wrong | Possible extra non-session directories in a patient folder | The script filters for dirs starting with `ses-`. Check for stray folders. |
| MONAI `Spacing` errors | MONAI version incompatibility | Tested with monai>=1.0. Run `pip install --upgrade monai`. |
