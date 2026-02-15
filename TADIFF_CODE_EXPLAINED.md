# TaDiff-Net: Complete Code Explanation with Paper References

**A learning-oriented guide to every file in the TaDiff codebase**

This document explains the entire TaDiff-Net codebase in detail, mapping every code component back to the paper: *"Treatment-aware Diffusion Probabilistic Model for Longitudinal MRI Generation and Diffuse Glioma Growth Prediction"* by Liu et al. (IEEE Transactions on Medical Imaging, 2025).

The goal is twofold: (1) understand what the code does and why, and (2) learn Python/PyTorch coding patterns along the way.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Repository Structure](#2-repository-structure)
3. [Configuration Files](#3-configuration-files)
   - 3.1 [cfg_tadiff_net.py — Model & Training Config](#31-cfg_tadiff_netpy)
   - 3.2 [test_config.py — Testing Config](#32-test_configpy)
4. [Data Pipeline](#4-data-pipeline)
   - 4.1 [preprocess_sailor.py — Raw NIfTI to .npy](#41-preprocess_sailorpy)
   - 4.2 [data_loader.py — .npy to Training Batches](#42-data_loaderpy)
5. [The Neural Network Architecture](#5-the-neural-network-architecture)
   - 5.1 [utils.py — Building Blocks](#51-utilspy)
   - 5.2 [tadiff_unet_arch.py — The TaDiff U-Net](#52-tadiff_unet_archpy)
6. [The Diffusion Process](#6-the-diffusion-process)
   - 6.1 [diffusion.py — Forward & Reverse Diffusion](#61-diffusionpy)
7. [Training](#7-training)
   - 7.1 [tadiff_model.py — The Training Loop](#71-tadiff_modelpy)
8. [Testing & Evaluation](#8-testing--evaluation)
   - 8.1 [test.py — Testing Pipeline](#81-testpy)
   - 8.2 [metrics.py — Evaluation Metrics](#82-metricspy)
9. [Inference](#9-inference)
   - 9.1 [inference.py — Prediction Without Ground Truth](#91-inferencepy)
10. [End-to-End Data Flow](#10-end-to-end-data-flow)
11. [Key Python & PyTorch Patterns Explained](#11-key-python--pytorch-patterns-explained)

---

## 1. The Big Picture

### What the paper proposes

TaDiff is a **conditional diffusion probabilistic model** that:
- Takes as input: 3 past MRI sessions (each with T1, T1c, FLAIR modalities) + treatment info + target time point
- Produces as output: a future MRI scan + tumor segmentation mask at the target time point

This is described in the paper's Section III and visualized in **Figure 1** and **Figure 2**.

### What makes TaDiff special vs regular diffusion models

| Feature | Standard DDPM | TaDiff |
|---------|--------------|--------|
| Input | Single image + noise | 3 historical MRIs + noise (at target position) |
| Conditioning | Timestep only | Timestep + treatment days + treatment type |
| Output | Denoised image only | Denoised image + segmentation masks |
| Loss | MSE on noise | Weighted MSE on noise + Dice loss on masks |

### How the code maps to the paper

| Paper Component | Code File |
|----------------|-----------|
| Section III-A (Problem Setting) | `config/cfg_tadiff_net.py`, `data/preprocess_sailor.py` |
| Section III-B (Network Architecture) | `src/net/tadiff_unet_arch.py` |
| Section II (Diffusion Theory) | `src/net/diffusion.py` |
| Algorithm 1 (Training) | `src/tadiff_model.py` → `get_loss()` |
| Algorithm 2 (Sampling/Inference) | `src/net/diffusion.py` → `TaDiff_inverse()` |
| Section III-C (Joint Loss) | `src/tadiff_model.py` → `get_loss()` lines 181-209 |
| Section IV-A (Dataset & Metrics) | `src/evaluation/metrics.py`, `data/preprocess_sailor.py` |
| Section IV-B (Implementation) | `config/cfg_tadiff_net.py` |

---

## 2. Repository Structure

```
TaDiff-Net/
├── config/
│   ├── cfg_tadiff_net.py      # All hyperparameters (model, training, system)
│   ├── test_config.py          # Test-time configuration
│   └── arg_parse.py            # CLI argument parsing
├── data/
│   ├── preprocess_sailor.py    # Fixed preprocessing script (our addition)
│   ├── preproc_prepare_data.py # Original preprocessing (has bugs)
│   ├── sailor_info.csv         # Patient metadata (ages, days, treatments)
│   ├── PREPROCESSING_GUIDE.md  # Preprocessing documentation (our addition)
│   └── sailor/                 # Expected structure for raw NIfTI data
├── src/
│   ├── tadiff_model.py         # PyTorch Lightning training module
│   ├── net/
│   │   ├── tadiff_unet_arch.py # The TaDiff U-Net architecture
│   │   ├── diffusion.py        # Gaussian diffusion (forward + reverse)
│   │   ├── utils.py            # Utility layers (embeddings, norms, etc.)
│   │   └── ssim.py             # SSIM loss implementation
│   ├── data/
│   │   └── data_loader.py      # MONAI data loading and transforms
│   ├── evaluation/
│   │   ├── metrics.py          # Dice, PSNR, SSIM, MAE, RAVD
│   │   └── ssim.py             # SSIM for evaluation
│   ├── utils/
│   │   └── image_processing.py # Image manipulation helpers
│   └── visualization/
│       └── visualizer.py       # Plotting, overlays, uncertainty maps
├── test.py                     # Testing pipeline
└── inference.py                # Inference without ground truth
```

---

## 3. Configuration Files

### 3.1 `cfg_tadiff_net.py`

**Purpose:** Central location for ALL hyperparameters. Rather than scattering numbers through the code, everything is defined here and accessed as `config.lr`, `config.batch_size`, etc.

**Paper Reference:** Section IV-B (Implementation and Training Details)

```python
# ── Model Architecture ──
image_size = 192          # Paper: "cropped to a patch size of 192x192"
in_channels = 13          # 4 sessions x 3 modalities + 1 mask channel = 13
out_channels = 7          # 3 modality predictions + 4 mask predictions = 7
model_channels = 32       # Paper: "64, 128, 256, and 512 channels"
                          # base=32, mult=(1,2,3,4) → 32, 64, 96, 128
num_res_blocks = 2        # 2 residual blocks per resolution level
channel_mult = (1,2,3,4)  # Channel multiplier at each U-Net level
attention_resolutions = [8,4]  # Add attention at 8x and 4x downsampled resolutions
num_heads = 4             # 4 attention heads
```

**Coding pattern — `DefaultMunch`:**
```python
from munch import DefaultMunch
config = DefaultMunch.fromDict(config)
```
`Munch` is a dictionary subclass that lets you access keys as attributes. So instead of `config['lr']` you write `config.lr`. This is purely for convenience and readability.

**Key hyperparameters and what they control:**

| Parameter | Value | What it does |
|-----------|-------|-------------|
| `max_T` | 1000 | Number of diffusion steps. Paper Section II: T=1000 |
| `ddpm_schedule` | `'linear'` | Beta schedule type. Paper: linear from 1e-4 to 2e-2 |
| `lr` | 5e-3 | Learning rate. Paper: "initial learning rate 2.5e-4" (with warmup) |
| `batch_size` | 1 | Batch size per GPU |
| `sw_batch` | 16 | Sliding window batch size (16 slices processed at once) |
| `accumulate_grad_batches` | 4 | Simulates batch_size=4 without extra GPU memory |
| `n_repeat_tr` | 10 | Repeats training data 10x per epoch (data augmentation) |
| `aux_loss_w` | 0.01 | Lambda in Eq. 16: weight for segmentation loss |

**Why `in_channels = 13`:** The model receives 4 sessions of data. Each session has 3 MRI modalities (T1, T1c, FLAIR — T2 is dropped). Plus, there's 1 channel for the mask input. That's 4x3 + 1 = 13. But wait, the code actually says `in_channels-1` when creating the model (line 26 of tadiff_model.py), so the actual U-Net input is 12 channels = 4 sessions x 3 modalities.

### 3.2 `test_config.py`

**Purpose:** Configuration specifically for the testing/evaluation phase.

**Coding pattern — `@dataclass`:**
```python
@dataclass
class TestConfig:
    diffusion_steps: int = 600
    num_samples: int = 5
    min_tumor_size: int = 20
```

A `dataclass` is Python's way of creating a class that's mainly used to hold data. Instead of writing `__init__`, `__repr__`, etc., Python auto-generates them from the type annotations. Think of it as a structured, typed configuration object.

Key test parameters:
- `diffusion_steps = 600` — Use 600 reverse steps at test time (not the full 1000 used in training)
- `num_samples = 5` — Generate 5 predictions per slice for ensemble averaging and uncertainty
- `target_session_idx = 3` — Predict the 4th session (index 3) given the first 3
- `min_tumor_size = 20` — Skip slices with fewer than 20 tumor voxels

---

## 4. Data Pipeline

### 4.1 `preprocess_sailor.py`

**Purpose:** Convert raw NIfTI (.nii.gz) brain MRI files into .npy arrays that the model can load.

**Paper Reference:** Section IV-A: "Scans were skull-stripped, registered to a common space defined by the T1, and resampled to isotropic 1 mm resolution. For intensity normalization, we utilized z-score normalization on a per-channel basis."

**The pipeline (what happens to each MRI scan):**

```
Raw NIfTI file on disk
    │
    ▼
1. nib.load(path)              ← Load the file
    │                             Returns a nibabel image with data + affine matrix
    │                             The affine tells us voxel sizes and orientation
    ▼
2. reorient_nifti(img, "PLI")  ← Standardize orientation
    │                             All scanners store images differently
    │                             PLI = Posterior-Left-Inferior (a convention)
    ▼
3. resample_to_1mm(img)        ← Make all voxels exactly 1mm x 1mm x 1mm
    │                             Uses MONAI's Spacing transform WITH the affine
    │                             (without affine, MONAI can't know original voxel sizes)
    │                             Uses "bilinear" for images, "nearest" for masks
    ▼
4. nonzero_norm_image(data)    ← Normalize intensity to [0, 1]
    │                             Step a: Clip outliers at 0.2th and 99.8th percentile
    │                             Step b: Z-score (subtract mean, divide by std) on non-zero voxels
    │                             Step c: Min-max scale to [0, 1]
    ▼
5. Stack & Save                ← Combine all modalities and sessions into .npy files
```

**Why each step matters:**

**Step 2 — Orientation:** Different MRI scanners store slices in different orders. One might go front-to-back, another right-to-left. If you don't standardize, the neural network sees the same brain from different "camera angles" and gets confused.

**Step 3 — Resampling:** One patient's scan might have 0.5mm voxels (very detailed) while another has 2mm voxels (coarse). Resampling to 1mm makes every voxel represent the same physical volume of brain tissue.

**Step 4 — Normalization:** MRI intensities are arbitrary — one scanner might produce values 0-4000, another 0-12000, for the same tissue. Normalization puts all scans on a common [0, 1] scale so the network sees consistent inputs.

**Mask handling (critical detail):**
```python
# For images: smooth interpolation
mode = "bilinear"  # Blends neighboring voxels smoothly

# For masks: nearest-neighbor interpolation
mode = "nearest"   # Picks the closest integer label
```
Masks contain discrete labels {0, 1, 3}. If you used bilinear interpolation, you'd get fractional values like 0.7 or 2.1, which are meaningless. Nearest-neighbor preserves the discrete integer labels.

**Mask merging logic:**
```python
merged = np.zeros_like(img)        # Start with all background (0)
merged[edema > 0] = EDEMA          # Set edema regions to 1
merged[et > 0] = ENHANCING         # Set enhancing tumor to 3
                                   # This OVERWRITES edema where they overlap
```
The enhancing tumor core sits inside the edema region. By writing enhancing last, it takes priority in overlapping areas. This matches the clinical hierarchy: enhancing tumor is more specific/important than surrounding edema.

**Output format (4 files per patient):**

| File | Shape | Type | What it contains |
|------|-------|------|-----------------|
| `sub-XX_image.npy` | (4*T, H, W, D) | float32 | 4 modalities x T sessions, normalized to [0,1] |
| `sub-XX_label.npy` | (T, H, W, D) | int8 | Merged segmentation masks (0=bg, 1=edema, 3=enhancing) |
| `sub-XX_days.npy` | (T,) | int64 | Cumulative days from baseline [0, 36, 64, 127, ...] |
| `sub-XX_treatment.npy` | (T,) | int64 | Treatment type per session (0=CRT, 1=TMZ) |

### 4.2 `data_loader.py`

**Purpose:** Load the .npy files and apply spatial transforms to create training batches.

**Paper Reference:** Section IV-B: "images were cropped to a patch size of 192x192"

```python
val_transforms = Compose([
    LoadImaged(keys=npz_keys, image_only=True),      # Load .npy files into memory
    CropForegroundd(keys=["image", "label"],          # Remove empty space around the brain
                    source_key="image"),
    CenterSpatialCropd(keys=["image", "label"],       # Crop to 192x192x192 cube
                       roi_size=[192, 192, 192]),
    SpatialPadd(keys=["image", "label"],              # Pad if brain is smaller than 192^3
                spatial_size=(192, 192, 192)),
    MergeMultiLabels(keys=["label"]),                 # Binarize: all labels > 0 become 1
])
```

**Coding pattern — MONAI transforms:**
MONAI uses a dictionary-based transform system. Each transform gets a dictionary like `{"image": array, "label": array, "days": array}` and modifies specific keys. The `d` suffix (like `CropForegroundd`) means "dictionary transform."

**`MergeMultiLabels`:**
```python
class MergeMultiLabels(MapTransform):
    def __call__(self, data):
        for key in self.key_iterator(data):
            data[key] = data[key] > 0  # Everything non-zero becomes 1 (True)
        return data
```
This merges all tumor sub-types (edema=1, enhancing=3) into a single binary mask (tumor=1, background=0). However, note that this is the *validation/test* transform. During training, the model actually works with the multi-class labels through its 4-channel mask output.

**`CacheDataset`:**
```python
test_dataset = CacheDataset(data=test_file_list, transform=val_transforms)
```
`CacheDataset` is MONAI's smart data loader. The first time it loads a patient, it applies all transforms and caches the result in RAM. Subsequent accesses are instant. This is crucial because medical image transforms (cropping 3D volumes, etc.) are slow.

---

## 5. The Neural Network Architecture

### 5.1 `utils.py` — Building Blocks

**Purpose:** Utility layers used throughout the network. Think of these as the LEGO bricks that the U-Net is assembled from.

#### `timestep_embedding` (Paper: Section III-B, "sinusoidal embedding [31]")

```python
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    return embedding
```

**What it does:** Converts a scalar number (like diffusion timestep=500, or day=127) into a high-dimensional vector. This is the same idea as positional encoding from the Transformer paper ("Attention is All You Need").

**Why it's needed:** Neural networks work with vectors, not single numbers. A single number like "500" doesn't give the network enough information to work with. The sinusoidal embedding creates a unique, high-dimensional fingerprint for each number, where nearby numbers have similar (but not identical) embeddings.

**Analogy:** It's like converting a radio frequency (a single number) into an audio waveform (a rich signal). The number 500 and the number 501 will produce similar waveforms, but 500 and 10 will produce very different ones.

**How the math works:**
1. Create a set of frequencies: `freqs = [1, 0.63, 0.40, 0.25, ...]` (exponentially decreasing)
2. Multiply the timestep by each frequency: `args = timestep * freqs`
3. Take cos and sin of each: `[cos(500*1), cos(500*0.63), ..., sin(500*1), sin(500*0.63), ...]`
4. The result is a 32-dimensional vector (when `dim=32`)

#### `FourierFeatures`

```python
class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=0.2):
        self.weight = nn.Parameter(th.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return th.cat([f.cos(), f.sin()], dim=-1)
```

Similar to sinusoidal embeddings, but with **learnable** frequencies (stored in `self.weight`). Used for the treatment embedding, because treatment codes (0 for CRT, 1 for TMZ) are categorical, not continuous like time.

**Coding pattern — `nn.Parameter`:** Wrapping a tensor in `nn.Parameter` tells PyTorch: "this is a learnable weight — please update it during backpropagation." Without this wrapper, the tensor would be treated as a constant.

#### `zero_module`

```python
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module
```

Initializes a layer's weights to all zeros. This is used for the final output convolution of the U-Net and in residual connections. The idea: at the start of training, the residual branch contributes nothing (all zeros), so the network initially behaves like a simpler model. As training progresses, these weights gradually learn to add useful modifications.

#### `GroupNorm32` and `normalization`

```python
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels, g=32):
    return GroupNorm32(g, channels)
```

**What is Group Normalization?** It normalizes features within groups of channels. If you have 128 channels and g=32 groups, each group of 4 channels is normalized together. This stabilizes training by keeping activations from exploding or vanishing.

**Why `float()` then back?** When using mixed precision (float16 for speed), normalization needs float32 for numerical stability. The `.float()` call temporarily upcasts, normalizes, then `.type(x.dtype)` casts back.

### 5.2 `tadiff_unet_arch.py` — The TaDiff U-Net

**Purpose:** The main neural network. This is the `mu_theta` from the paper (Eq. 10) — the network that predicts the noise `epsilon` and segmentation masks.

**Paper Reference:** Section III-B (Treatment-aware Diffusion Network), Figure 2

#### Architecture Overview

The TaDiff U-Net follows the classic encoder-bottleneck-decoder structure with skip connections, but adds three parallel conditioning streams:

```
Input (12 channels: 4 sessions x 3 modalities)
    │
    ├── time_embed:   diffusion timestep t → 32-dim vector
    ├── days_embed:   treatment days [d1,d2,d3,dt] → 4x 32-dim vectors
    └── treats_embed: treatment codes [tau1,tau2,tau3,taut] → 4x 32-dim vectors
    │
    ▼
    Combine into treat_day_diff (128-dim conditioning vector)
    │
    ▼
┌─ Encoder (input_blocks) ─────────────────────────────┐
│  Level 0: 32 ch  → ResBlock → ResBlock → Downsample  │ ← conditioned by treat_day_diff
│  Level 1: 64 ch  → ResBlock → ResBlock → Downsample  │
│  Level 2: 96 ch  → ResBlock+Attn → ResBlock+Attn → ↓│
│  Level 3: 128 ch → ResBlock+Attn → ResBlock+Attn     │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌─ Bottleneck (middle_block) ───────────────────────────┐
│  ResBlock → AttentionBlock → ResBlock                  │ ← conditioned by middle_emb
└───────────────────────────────────────────────────────┘    (time + target treatment-day)
    │
    ▼
┌─ Decoder (output_blocks) ─────────────────────────────┐
│  Level 3: concat skip → ResBlock+Attn → Upsample      │ ← conditioned by treat_day_diff
│  Level 2: concat skip → ResBlock+Attn → Upsample      │
│  Level 1: concat skip → ResBlock → Upsample           │
│  Level 0: concat skip → ResBlock                      │
└───────────────────────────────────────────────────────┘
    │
    ▼
Output (7 channels: 4 mask channels + 3 image channels)
```

#### The Three Embedding Streams (Paper: Section III-B)

The paper says: "we utilize two separate embedding and MLP layers for injecting the paired treatment and day information"

```python
# Stream 1: Diffusion timestep
self.time_embed = nn.Sequential(
    linear(model_channels, model_channels * 2),   # 32 → 64
    nn.SiLU(),                                      # Activation function
    linear(model_channels * 2, model_channels),    # 64 → 32
)

# Stream 2: Treatment days
self.days_embed = nn.Sequential(
    linear(model_channels, model_channels * 2),    # 32 → 64
    nn.SiLU(),
    linear(model_channels * 2, model_channels),    # 64 → 32
)

# Stream 3: Treatment codes (with Fourier features)
self.treats_embed = nn.Sequential(
    FourierFeatures(1, model_channels),            # 1 → 32 (learnable frequencies)
    linear(model_channels, model_channels * 2),    # 32 → 64
    nn.SiLU(),
    linear(model_channels * 2, model_channels),    # 64 → 32
)
```

**Why separate streams?** Each type of information (time, days, treatment) lives on a different "scale." Diffusion timesteps range from 1-1000, treatment days from 0-700+, and treatment codes are just 0 or 1. Separate MLPs let each stream learn appropriate representations.

**Why `SiLU` (also called Swish)?** `SiLU(x) = x * sigmoid(x)`. It's a smooth activation function that's become standard in diffusion models. Unlike ReLU, it doesn't have a hard cutoff at zero, which helps gradient flow.

#### The Forward Pass — Treatment-Day Conditioning

This is where TaDiff's novelty lives. Here's the `forward()` method explained step by step:

```python
def forward(self, x, timesteps, intv_t=None, treat_code=None, i_tg=None):
```

**Step 1: Embed the diffusion timestep**
```python
emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
# timesteps = [500]  →  sinusoidal_embedding  →  [32-dim vector]  →  MLP  →  [32-dim vector]
```

**Step 2: Embed all treatment days**
```python
d_embs = [timestep_embedding(days, self.model_channels) for days in intv_t]
# intv_t = [36, 64, 127, 225]  →  4 sinusoidal embeddings  →  4x [32-dim vectors]
days_feat = [self.days_embed(d) for d in d_embs]
# Each goes through the days MLP  →  4x [32-dim vectors]
```

**Step 3: Embed all treatment codes**
```python
treat_feat = [self.treats_embed((t[:, None] + 1)*10) for t in treat_code]
# treat_code = [0, 1, 1, 1] (CRT, TMZ, TMZ, TMZ)
# Each: (code + 1) * 10  →  FourierFeatures  →  MLP  →  [32-dim vector]
```

The `(t + 1) * 10` scaling maps treatment codes from {0, 1} to {10, 20}, giving the Fourier features more range to work with.

**Step 4: Combine treatment + days into treat_day_sum (Paper: "sum the learned treatment embedding with its day embedding")**
```python
treat_day_sum = th.cat([(t + d).unsqueeze(1) for t, d in zip(treat_feat, days_feat)], dim=1)
# Shape: (batch, 4_sessions, 32)
# Each session gets one combined treatment-day vector
```

**Step 5: Compute RELATIVE differences (Paper: "relative (difference) distance between source and target treatment times")**
```python
target = treat_day_sum[:, i_tg, :]    # Extract the target session's embedding
treat_day_diff = treat_day_sum - target[:, None, :]  # Subtract target from all
```

This is a key insight from the paper. Instead of absolute embeddings, the network sees *how different* each source session is from the target. If all sessions had the same treatment and similar days, the differences would be small (small changes expected). If the target is far in the future with different treatment, the differences would be large (big changes expected).

**Step 6: Special conditioning for the bottleneck**
```python
middle_emb = emb + target  # Timestep + target treatment-day info only
```

The middle block (bottleneck) gets only the timestep and target information, not the relative differences. The paper explains this gives a "smooth and fast learning process when blending the target-treatment messages into the diffusion timesteps."

**Step 7: Inject target position info into treat_day_diff**
```python
for i, j in zip(range(b), i_tg):
    treat_day_diff[i, j, :] = treat_day_diff[i, j, :] + middle_emb[i]
```

The target session's slot in treat_day_diff gets enriched with the timestep embedding. This tells the network which position contains the noisy image being denoised.

**Step 8: Flatten and run through U-Net**
```python
treat_day_diff = treat_day_diff.view(b, -1)  # (batch, 4*32) = (batch, 128)

# Encoder
for module in self.input_blocks:
    h = module(h, treat_day_diff)  # Each ResBlock is conditioned by treat_day_diff
    hs.append(h)                    # Save for skip connections

# Bottleneck
h = self.middle_block(h, middle_emb)  # Conditioned by time + target only

# Decoder
for module in self.output_blocks:
    h = th.cat([h, hs.pop()], dim=1)  # Concatenate skip connection
    h = module(h, treat_day_diff)       # Conditioned by treat_day_diff
```

#### ResBlock — The Workhorse (Paper: based on [30] Ho et al.)

```python
class ResBlock(TimestepBlock):
    def _forward(self, x, emb):
        # Main path
        h = self.in_layers(x)          # GroupNorm → SiLU → Conv3x3
        emb_out = self.emb_layers(emb) # SiLU → Linear (project conditioning)
        h = h + emb_out                # ADD the conditioning
        h = self.out_layers(h)         # GroupNorm → SiLU → Dropout → Conv3x3(zero-init)

        # Skip connection
        return self.skip_connection(x) + h
```

**The residual pattern:** `output = skip(x) + transform(x)`. This is the fundamental building block of deep networks. The skip connection lets gradients flow directly backward through the network, preventing the vanishing gradient problem. The `zero_module` initialization of the last conv means the transform initially outputs zeros, so the block starts as an identity function.

**How conditioning is injected:** The conditioning vector `emb` (128-dim) is projected to match the channel dimension and simply ADDED to the feature map after the first convolution. This is a lightweight conditioning method — the conditioning doesn't change the convolution weights, it just adds a bias that shifts features based on treatment-day-time information.

#### AttentionBlock — Self-Attention (Paper: Figure 2, at lower resolutions)

```python
class AttentionBlock(nn.Module):
    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)        # Flatten spatial dims: (B,C,H*W)
        qkv = self.qkv(self.norm(x))   # Compute Q, K, V from normalized features
        h = self.attention(qkv)          # Multi-head attention
        h = self.proj_out(h)            # Project back (zero-initialized)
        return (x + h).reshape(b, c, *spatial)  # Residual connection
```

**What self-attention does here:** At low resolutions (8x8 and 4x4 feature maps), the receptive field of convolutions is limited. Self-attention lets every spatial position attend to every other position, capturing long-range dependencies. For brain tumors, this means the network can learn relationships between distant parts of the brain.

**`softmax_one` — A modified softmax:**
```python
def softmax_one(x, dim=None):
    exp_x = th.exp(x - x.max(dim=dim, keepdim=True).values)
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))  # Note the +1
```

Standard softmax: `exp(x_i) / sum(exp(x_j))`
Softmax_one: `exp(x_i) / (1 + sum(exp(x_j)))`

The `+1` in the denominator means attention weights can sum to less than 1. This allows the attention mechanism to "attend to nothing" — useful when some spatial positions have no relevant content.

---

## 6. The Diffusion Process

### 6.1 `diffusion.py` — Forward & Reverse Diffusion

**Purpose:** Implements the mathematical diffusion process described in Section II of the paper.

**Paper Reference:** Section II (Preliminary), Equations 1-12

#### The Forward Process (Adding Noise) — Paper Eq. 2

The forward process gradually adds Gaussian noise to a clean image over T timesteps:

**x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon**

```python
class GaussianDiffusion:
    def __init__(self, T, schedule):
        self.T = T  # Number of steps (1000 for training, 600 for testing)

        # Linear beta schedule: Paper says beta_1=1e-4, beta_T=2e-2
        b0 = 1e-4
        bT = 2e-2
        self.beta = torch.linspace(b0, bT, T)  # T linearly spaced values

        # alpha_t = 1 - beta_t
        self.alpha = 1 - self.beta

        # alpha_bar_t = product of all alphas up to t
        self.alphabar = torch.cumprod(self.alpha, dim=0)
```

**What these variables mean physically:**
- `beta[t]` — the variance of noise added at step t. Starts tiny (1e-4), grows to 0.02
- `alpha[t]` — how much of the original signal survives step t (close to 1 early, drops later)
- `alphabar[t]` — cumulative product: how much of the original signal survives after ALL steps up to t

At t=0: `alphabar ≈ 1.0` → image is almost clean
At t=500: `alphabar ≈ 0.05` → image is mostly noise
At t=1000: `alphabar ≈ 0.0` → image is pure noise

```python
def sample(self, x0, t):
    atbar = self.alphabar[t-1]
    epsilon = torch.randn_like(x0)                         # Random noise
    xt = torch.sqrt(atbar) * x0 + torch.sqrt(1-atbar) * epsilon   # Paper Eq. 2
    return xt, epsilon
```

**Coding pattern — `torch.randn_like(x0)`:** Creates a tensor of the same shape and device as x0, filled with random numbers from N(0,1). The `_like` suffix is a common PyTorch pattern that copies shape and device from an existing tensor.

#### The Reverse Process (DDPM Sampling) — Paper Algorithm 2, Eq. 12

```python
def TaDiff_inverse(self, net, x, intv, treat_cond, i_tg,
                   steps, start_t, step_mask=10, device='cpu'):
```

This is the core inference algorithm. Starting from noise, it iteratively denoises to produce a clean image.

**Step by step through the reverse loop:**

```python
for t in range(start_t, start_t-steps, -1):   # Count down: 600, 599, ..., 1
    at = self.alpha[t-1]                        # alpha at current step
    atbar = self.alphabar[t-1]                  # cumulative alpha at current step

    # Prepare noise for stochastic sampling
    if t > 1:
        z = torch.randn_like(x0)               # Fresh random noise
        beta_tilde = self.beta[t-1] * (1 - self.alphabar[t-2]) / (1 - atbar)  # Eq. 8
    else:
        z = torch.zeros_like(x0)               # No noise at final step
        beta_tilde = 0

    # Ask the neural network: "given this noisy image, what noise was added?"
    with torch.no_grad():                       # No gradient computation (inference only)
        pred = net(x, t, intv_t=intv, treat_code=treat_cond, i_tg=i_tg)
        img_p = pred[:, 4:7, :, :]             # Predicted noise (3 channels)
        mask = pred[:, 0:4, :, :]              # Predicted mask (4 channels)

    # Apply the denoising formula: Paper Eq. 12
    # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alphabar_t) * epsilon) + sqrt(beta_tilde) * z
    x0 = (1/sqrt(at)) * (xt - ((1-at)/sqrt(1-atbar)) * img_p) + sqrt(beta_tilde) * z
```

**The output split — `pred[:, 4:7]` vs `pred[:, 0:4]`:**
The network outputs 7 channels total:
- Channels 0-3: mask predictions (4 channels: one per session)
- Channels 4-6: noise prediction (3 channels: one per modality — T1, T1c, FLAIR)

#### Mask Fusion — Paper Algorithm 2, lines 4-9

```python
T_m = step_mask  # Default: 10 (last 10 steps only)
w_p = (self.alphabar[:T_m] / torch.sum(self.alphabar[:T_m]))  # Weights

# During the last T_m steps:
if t <= T_m:
    y += mask * w_p[t-1]  # Weighted accumulation of mask predictions
```

**Why only the last 10 steps?** Early in the reverse process, the image is still very noisy, so mask predictions are unreliable. The last few steps produce the cleanest images and best masks. The weighting by `alphabar` gives higher weight to the cleanest steps (highest alpha_bar).

This corresponds to Paper Algorithm 2 line 9: "Fuse the last T_m predicted masks"

#### DDIM Sampling (Faster Alternative)

```python
def ddim_inverse(self, net, x, eta=0.0, steps, ...):
```

DDIM (Denoising Diffusion Implicit Models) is a deterministic variant of DDPM that can produce good results in fewer steps. When `eta=0`, it's completely deterministic (same input → same output). When `eta>0`, it adds stochasticity.

The inference script uses DDIM with only 50 steps (vs 600 for DDPM), which is ~12x faster.

---

## 7. Training

### 7.1 `tadiff_model.py` — The Training Loop

**Purpose:** This file contains the complete training logic, implementing **Algorithm 1** from the paper.

**Paper Reference:** Algorithm 1 (TaDiff Training), Section III-C (Joint Loss Function), Equations 14-16

**Framework: PyTorch Lightning**

```python
class Tadiff_model(LightningModule):
```

PyTorch Lightning is a framework that handles the boilerplate of training (GPU management, logging, checkpointing, distributed training). You just need to implement:
- `__init__` — set up the model
- `configure_optimizers` — set up the optimizer
- `training_step` — what happens each training iteration
- `validation_step` — what happens each validation iteration

#### `__init__` — Model Setup

```python
def __init__(self, config):
    self._model = TaDiff_Net(...)                      # The U-Net
    self.diffusion = GaussianDiffusion(T=1000, ...)    # The diffusion process
    self.alphabar = np.cumprod(1-np.linspace(1e-4, 2e-2, 1000))  # Precomputed alpha_bar
    self.dilation_filters = torch.ones(1,1,11,11) / 10.  # For DF-weighting (Eq. 15)
    self.dice = DiceLoss(...)                           # Dice loss for segmentation
```

#### `get_loss()` — The Heart of Training (Paper Algorithm 1)

This is the most complex function in the codebase. Let's trace through it step by step.

**Step 1: Unpack the batch**
```python
imgs, label, days, treatments = batch["image"], batch["label"], batch["days"], batch["treatments"]
# imgs:       (batch, sessions, channels, H, W) — e.g., (1, 4, 3, 192, 192)
# label:      (batch, sessions, H, W) — e.g., (1, 4, 192, 192)
# days:       (batch, 4) — e.g., [[0, 36, 64, 127]]
# treatments: (batch, 4) — e.g., [[0, 0, 1, 1]]
```

**Step 2: Randomly choose which session to predict (Paper Algorithm 1, line 2)**
```python
if mode == 'train' and np.random.random_sample() > 0.5:
    i_tg = torch.randint(0, s, (b,))   # Random target among all sessions
else:
    i_tg = -torch.ones((b,), dtype=torch.int8)  # -1 means "last session" (future)
```

The paper says the model is trained to predict three scenarios: future (50%), middle (30%), and past (20%). The `i_tg` variable stores which session is the target. When `i_tg = -1`, Python indexing makes this the last element (the future session). This data augmentation strategy is described in Section IV-B.

**Step 3: Extract the target image and add noise (Paper Algorithm 1, lines 3-5)**
```python
gt_img = torch.cat([imgs[[i], j, :, :, :] for i, j in zip(range(b), i_tg)])  # Ground truth target
t = torch.randint(1, self.diffusion.T + 1, [gt_img.shape[0]])    # Random timestep
xt, epsilon = self.diffusion.sample(gt_img, t)   # Add noise → noisy target (Eq. 2)
```

**Step 4: Replace the target position with the noisy version**
```python
for i, j in zip(range(b), i_tg):
    imgs[i, j, :, :, :] = xt[i, :, :, :]    # Swap clean target with noisy version
```

Now the input to the network is: 3 clean source MRIs + 1 noisy target MRI.

**Step 5: Feed through the network (Paper Algorithm 1, line 6)**
```python
out = self.forward(xt, t, intv_t=intvs, treat_code=treat_cond, i_tg=i_tg)
img_pred = out[:, 4:7, :, :]    # Predicted noise (epsilon_hat)
mask_pred = out[:, 0:4, :, :]   # Predicted masks for all 4 sessions
```

**Step 6: Compute DF-Weighting (Paper Eq. 15)**
```python
loss_weights = torch.sum(label, dim=1, keepdim=True)  # Sum masks across sessions → tumor area
loss_weights = loss_weights * torch.exp(-loss_weights)  # w = m * e^(-m)
loss_weights = F.conv2d(loss_weights, self.dilation_filters, padding='same') + 1.
```

**What this does:** Creates a spatial weight map `omega` that emphasizes the region AROUND the tumor:
1. Sum all masks → highlights voxels that are tumor in any session
2. `m * e^(-m)` → creates a hump function: zero for no-tumor (m=0), peaks at m=1, decreases for high m
3. Convolve with an 11x11 filter → dilates (expands) the region outward
4. Add 1 → ensures minimum weight of 1 everywhere

**Why?** The paper explains this prevents the model from ignoring the tumor region. Most of the brain doesn't change between sessions, so without this weighting, the model would optimize for the easy background pixels and ignore the dynamic tumor region.

**Step 7: Diffusion loss — Weighted MSE on predicted noise (Paper Eq. 16, first term)**
```python
loss1 = torch.mean(loss_weights * (img_pred - epsilon)**2)
# Weighted MSE: error is penalized more near the tumor region
```

**Step 8: Segmentation loss — Dice loss on predicted masks (Paper Eq. 14)**
```python
dice_loss = self.dice(mask_pred, label)  # Dice loss for all 4 session masks

# Weight the future mask's dice loss by sqrt(alpha_bar)
for i, j in zip(range(b), i_tg):
    dice_loss[i, j] = dice_loss[i, j] * torch.sqrt(w_tg[i])
```

**Why `sqrt(alpha_bar)` weighting?** At high noise levels (large t), `alpha_bar` is small, meaning the image is mostly noise. Predicting a good mask from pure noise is impossible, so we downweight the dice loss for noisy inputs. As the image gets cleaner (denoising progresses, alpha_bar → 1), the mask prediction becomes more important and gets full weight. This is Paper Eq. 14.

**Step 9: Combine losses (Paper Eq. 16)**
```python
loss = loss1 + torch.mean(dice_loss) * self.cfg.aux_loss_w  # aux_loss_w = lambda = 0.01
```

**Paper Eq. 16:** `L_TaDiff = ||omega(epsilon - epsilon_hat)||^2 + lambda * L_seg`

The lambda=0.01 means segmentation loss is weighted much less than diffusion loss. This is deliberate: the primary task is generating good MRIs (diffusion), and segmentation is a helpful auxiliary task.

#### `configure_optimizers` — Optimizer Setup

```python
optimizer = AdamW(self.trainer.model.parameters(), lr=5e-3, weight_decay=3e-5)
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=100, t_total=total_steps)
```

**AdamW:** An improved version of Adam optimizer with proper weight decay. It's the standard choice for diffusion models.

**WarmupCosineSchedule:** The learning rate starts at 0, linearly increases to 5e-3 over 100 steps (warmup), then follows a cosine curve down to near-zero. This prevents instability at the start of training when weights are random.

---

## 8. Testing & Evaluation

### 8.1 `test.py` — Testing Pipeline

**Purpose:** Load a trained model, run it on test patients, compute evaluation metrics, and save visualizations.

**Paper Reference:** Section IV-C (Test Results), Tables I and II

**The testing workflow:**

```
For each test patient:
    For each session (time point):
        1. Find the 3 slices with the most tumor (by volume)
        2. For each slice:
            a. Create 5 copies with different random noise (num_samples=5)
            b. Run 600 reverse diffusion steps on each
            c. Average the 5 predictions (ensemble)
            d. Compute standard deviation (uncertainty)
            e. Calculate metrics: Dice, SSIM, PSNR, MAE, RAVD
            f. Save visualizations
```

**Key function — `process_slice()`:**

```python
# Step 1: Select 4 consecutive sessions
session_indices = [session_idx - 3, session_idx - 2, session_idx - 1, session_idx]
session_indices[session_indices < 0] = 0  # Clamp to valid range

# Step 2: Extract the 2D slice from the 3D volume
seq_imgs = images[0, session_indices, :, :, :, :]
seq_imgs = seq_imgs[:, :, :, :, [slice_idx]*num_samples]  # Replicate slice 5 times

# Step 3: Replace target with noise
noise = torch.randn((num_samples, 3, h, w))
for i, j in zip(range(num_samples), i_tg):
    x_t[i, j, :, :, :] = noise[i, :, :, :]  # Each copy gets different noise

# Step 4: Run reverse diffusion
diffusion = GaussianDiffusion(T=600, schedule="linear")
pred_img, seg_seq = diffusion.TaDiff_inverse(model, steps=600, x=x_t, ...)
```

**Why 5 samples?** Each sample starts from different random noise. By averaging 5 predictions, we get a more robust estimate. The standard deviation across the 5 predictions gives an uncertainty map — showing where the model is confident vs uncertain.

**Coding pattern — `torch.no_grad()`:**
```python
with torch.no_grad():
    pred = net(x, t, ...)
```
During inference, we don't need to compute gradients (no backpropagation). `no_grad()` tells PyTorch to skip gradient computation, saving ~50% memory and running faster.

### 8.2 `metrics.py` — Evaluation Metrics

**Purpose:** Compute quantitative evaluation metrics.

**Paper Reference:** Section IV-A (Metric)

```python
class MetricsCalculator:
    def __init__(self, device, dice_thresholds=[0.25, 0.5, 0.75]):
        self.dice_metrics = {
            f'dice_{int(t*100)}': Dice(threshold=t).to(device)
            for t in dice_thresholds
        }
        self.mae = MAE().to(device)
        self.psnr = PSNR().to(device)
        self.ssim = SSIM(win_size=11, win_sigma=1.5, data_range=255, ...)
```

**Coding pattern — dictionary comprehension:**
```python
{f'dice_{int(t*100)}': Dice(threshold=t) for t in [0.25, 0.5, 0.75]}
```
This creates: `{'dice_25': Dice(0.25), 'dice_50': Dice(0.50), 'dice_75': Dice(0.75)}`

It's a compact way to create multiple similar objects. The `f'dice_{int(t*100)}'` is an f-string that formats the threshold as a name.

**What each metric measures:**

| Metric | What it measures | Higher or lower is better? | Used for |
|--------|-----------------|---------------------------|----------|
| **DSC (Dice)** | Overlap between predicted and true tumor masks | Higher (max 1.0) | Segmentation accuracy |
| **SSIM** | Structural similarity between predicted and true MRI | Higher (max 1.0) | MRI generation quality |
| **PSNR** | Peak signal-to-noise ratio | Higher | MRI generation quality |
| **MAE** | Average absolute pixel error | Lower (min 0.0) | MRI generation quality |
| **RAVD** | Relative absolute volume difference | Lower (min 0.0) | Tumor size prediction |

**RAVD computation:**
```python
@staticmethod
def ravd(pred, truth, threshold=0.5):
    pred = (pred >= threshold).astype(bool)  # Binarize prediction
    truth = truth.astype(bool)                # Binarize ground truth
    vol1 = np.count_nonzero(pred) + 1         # +1 prevents division by zero
    vol2 = np.count_nonzero(truth) + 1
    return (vol1 - vol2) / float(vol2)        # Relative difference
```

RAVD tells you: "how wrong is the predicted tumor volume?" A positive value means over-prediction (predicted tumor is larger), negative means under-prediction. Zero means perfect volume match.

---

## 9. Inference

### 9.1 `inference.py` — Prediction Without Ground Truth

**Purpose:** Generate predictions for new/future time points where no ground truth exists.

**Paper Reference:** This extends Algorithm 2 for clinical use.

**Key difference from testing:** In testing, we have ground truth to compare against. In inference, we just generate predictions for arbitrary future time points.

```bash
python inference.py --patient_ids 17 --slice_idx 102 --input_day 10000 --input_treatment 1
```

This command says: "For patient 17, at slice 102, predict what the brain will look like 10000 days from the last known scan, under TMZ treatment (code=1)."

**Key code differences from test.py:**

1. **DDIM sampling instead of DDPM:**
```python
pred_img, seg_seq = diffusion.ddim_inverse(...)  # Faster (50 steps vs 600)
```

2. **No metric computation:** Since there's no ground truth, we can't compute Dice, SSIM, etc.

3. **Concatenation of historical + future:**
```python
days = torch.cat([hist_days, input_days.unsqueeze(0)], dim=1)
# Historical: [0, 36, 64, 127]  +  Future: [10000]  →  [0, 36, 64, 127, 10000]
```

4. **Saves predictions as numpy arrays** for downstream analysis:
```python
np.save(f'prediction-slice-{slice_idx:03d}.npy', pred_img.cpu().numpy())
np.save(f'segmentation-slice-{slice_idx:03d}.npy', seg_seq.cpu().numpy())
```

---

## 10. End-to-End Data Flow

Here is the complete journey of data through the system, from raw MRI files to final prediction:

```
                            PREPROCESSING (preprocess_sailor.py)
                            ====================================
Raw NIfTI files              →   .npy arrays
(sub-01/ses-01/T1.nii.gz)       (sub-01_image.npy)
                                 (sub-01_label.npy)
                                 (sub-01_days.npy)
                                 (sub-01_treatment.npy)


                            DATA LOADING (data_loader.py)
                            ============================
.npy arrays                  →   PyTorch tensors in batches
CropForeground               →   Remove empty space around brain
CenterSpatialCrop(192^3)     →   Standard size cube
SpatialPad(192^3)             →   Pad if too small
MergeMultiLabels              →   Binarize masks


                            TRAINING (tadiff_model.py → get_loss)
                            =====================================
batch = {                    →   Algorithm 1 from the paper
  image: (B, 4, 3, H, W)         1. Pick random target session
  label: (B, 4, H, W)            2. Add noise to target image at random timestep t
  days:  (B, 4)                   3. Feed [3 clean + 1 noisy] through U-Net
  treatment: (B, 4)               4. Network predicts: noise (3ch) + masks (4ch)
}                                 5. Loss = DF-weighted MSE + 0.01 * Dice


                            THE U-NET (tadiff_unet_arch.py)
                            ==============================
Input: (B, 12, H, W)        →   Output: (B, 7, H, W)
  12 = 4 sessions x 3 mods       7 = 4 mask channels + 3 noise channels

Conditioning:
  timestep t  →  time_embed  →  emb (32-dim)
  days [d1..d4]  →  days_embed  →  4x (32-dim)
  treats [t1..t4]  →  treats_embed  →  4x (32-dim)
                         ↓
  treat_day_sum = treat + day per session  →  (B, 4, 32)
  target = treat_day_sum[i_tg]             →  (B, 32)
  treat_day_diff = all - target            →  (B, 4, 32) → flatten → (B, 128)
  middle_emb = emb + target                →  (B, 32)

  Encoder & Decoder: conditioned by treat_day_diff (128-dim)
  Bottleneck:        conditioned by middle_emb (32-dim)


                            REVERSE DIFFUSION (diffusion.py → TaDiff_inverse)
                            ================================================
Start: pure noise in target slot    →   End: predicted clean MRI + masks
Loop t = 600 down to 1:
  1. Network predicts noise + masks
  2. Subtract predicted noise (Eq. 12)
  3. Add small fresh noise (stochastic)
  4. Accumulate mask predictions (last 10 steps, weighted by alphabar)


                            EVALUATION (test.py + metrics.py)
                            =================================
5 samples (different noise seeds)   →   Ensemble average + uncertainty
                                        Metrics: Dice, SSIM, PSNR, MAE, RAVD
                                        Visualizations: overlays, contours, heatmaps
```

---

## 11. Key Python & PyTorch Patterns Explained

This section explains coding patterns you'll encounter throughout the codebase. Understanding these will help you read and modify the code confidently.

### 11.1 Tensor Shapes and Reshaping

Throughout the code, you'll see frequent reshaping. Here's a reference:

```python
# Adding a dimension
x.unsqueeze(0)    # (3, H, W)  →  (1, 3, H, W)     — add batch dim
x.unsqueeze(-1)   # (B, C)     →  (B, C, 1)          — add spatial dim
x[:, None, :]     # Same as unsqueeze(1)

# Removing a dimension
x.squeeze(0)      # (1, 3, H, W)  →  (3, H, W)

# Reshaping
x.view(b, 4, 3, h, w)    # Reinterpret flat channels as sessions x modalities
x.reshape(b, 12, h, w)   # Flatten sessions x modalities back to channels
x.contiguous()            # Ensure memory layout is correct after view/permute

# Reordering dimensions
x.permute(0, 2, 1, 3, 4)  # (B,S,C,H,W) → (B,C,S,H,W)  — swap sessions and channels
```

### 11.2 List Comprehensions

Used extensively for creating lists of embeddings:
```python
# Create an embedding for each session's days
d_embs = [timestep_embedding(days, dim) for days in intv_t]
# If intv_t = [36, 64, 127, 225], this creates 4 embedding vectors

# Extract a specific slice from each batch element
xt = [x[[i], j, :, :, :] for i, j in zip(range(b), i_tg)]
# For batch of 2 with i_tg=[3, 2]: extracts session 3 from item 0, session 2 from item 1
```

### 11.3 The `zip` and `enumerate` Patterns

```python
# zip: iterate over two lists in parallel
for treat_feat, day_feat in zip(treat_features, day_features):
    combined = treat_feat + day_feat

# enumerate: get both index and value
for idx, session_id in enumerate(session_ids):
    print(f"Session {idx}: {session_id}")

# zip(range(b), i_tg): pair each batch index with its target index
for i, j in zip(range(b), i_tg):
    x[i, j] = noisy_image[i]
    # Batch item i → modify session j
```

### 11.4 PyTorch Lightning Callbacks

```python
class MyCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Called automatically after each validation epoch
        # Generate sample predictions for visual inspection
        preds, aux_out = self.diffusion.TaDiff_inverse2(pl_module, ...)
        trainer.logger.log_image(key="Flair", images=[...])
```

Callbacks are a "hook" system. You define functions with specific names (like `on_validation_epoch_end`), and PyTorch Lightning calls them at the right time. This keeps the main training code clean.

### 11.5 Broadcasting in PyTorch

```python
treat_day_diff = treat_day_sum - target[:, None, :]
# treat_day_sum: (B, 4, 32)
# target:        (B, 32)
# target[:, None, :]: (B, 1, 32)  — add a dimension
# Broadcasting: (B, 4, 32) - (B, 1, 32)  →  (B, 4, 32)
# The (1) dimension is automatically expanded to match (4)
```

### 11.6 `torch.no_grad()` Context Manager

```python
with torch.no_grad():
    pred = model(x, t, ...)
```

During inference, we don't need gradients (no learning happening). This context manager:
- Saves ~50% GPU memory (no gradient tensors stored)
- Runs ~20% faster (no gradient computation overhead)
- MUST be used during testing/inference

### 11.7 Device Management

```python
# Check what hardware is available
if torch.cuda.is_available():
    device = "cuda:0"           # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"              # Apple Silicon GPU
else:
    device = "cpu"              # Fallback

# Move tensors to the device
x = x.to(device)                # Move tensor
model = model.to(device)        # Move all model parameters
```

All tensors that interact must be on the same device. Moving data between CPU and GPU is slow, so you want to keep everything on GPU during computation.

### 11.8 `nn.Sequential` — Chaining Layers

```python
self.time_embed = nn.Sequential(
    linear(32, 64),    # Layer 1: 32 → 64
    nn.SiLU(),         # Layer 2: activation
    linear(64, 32),    # Layer 3: 64 → 32
)
# When called: output = self.time_embed(input)
# Equivalent to: output = linear2(SiLU(linear1(input)))
```

`nn.Sequential` chains layers in order. Input flows through each layer sequentially. It's syntactic sugar for writing out each step manually.

### 11.9 F-strings for Formatting

```python
print(f"Patient {patient_id}: {n_sessions} sessions, days={days}")
# Output: "Patient sub-01: 8 sessions, days=[0, 36, 64, 127, 183, 225, 280, 350]"

f"sub-{i:02d}"    # Format with leading zeros: "sub-01", "sub-02", ..., "sub-27"
f"{val:.4f}"       # 4 decimal places: "0.9192"
f"{name:>10}"      # Right-align in 10 characters
```

### 11.10 Assertions for Debugging

```python
assert image_arr.shape[0] == 4 * n_sessions, (
    f"Image stack has {image_arr.shape[0]} slabs but expected "
    f"4 modalities x {n_sessions} sessions = {4*n_sessions}"
)
```

Assertions are sanity checks. If the condition is False, Python stops immediately with the error message. They're invaluable for catching shape mismatches, which are the #1 source of bugs in deep learning code.

---

## Summary

The TaDiff system can be understood as four major components working together:

1. **Preprocessing** converts raw brain scans into standardized arrays
2. **The U-Net** takes those arrays (with noise injected in the target position) and predicts the noise + segmentation masks, conditioned on treatment information
3. **The Diffusion Process** orchestrates the gradual addition and removal of noise
4. **The Joint Loss** trains both tasks (MRI generation + segmentation) simultaneously, with smart weighting to focus on the tumor region

The key innovation is the treatment-aware conditioning: three parallel embedding streams for time, days, and treatment type, combined through relative differences. This lets the model understand not just "where is the tumor now" but "how will treatment X affect the tumor over Y days."
